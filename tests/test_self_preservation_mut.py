"""Mutation-hardening tests for self_preservation — kill surviving mutants in
the AST lethal-call detection, lockdown gate, size-reduction / stub heuristics
and the emergency memory-cleanup threshold."""
from aegis.layers.self_preservation import SelfPreservation


class _FakeMemory:
    def __init__(self, n_episodic):
        self.episodic = [{"importance": 0.5, "event": f"e{i}"} for i in range(n_episodic)]
        self.semantic = {}


class _FakeSubstrate:
    def __init__(self, memory):
        self.memory = memory


# ── AST lethal-call detection: And->Or would crash on non-Name funcs ──

def test_raise_attribute_call_does_not_crash(tmp_path):
    # Kills L69 And->Or: `raise obj.method()` is a Call whose func is an
    # Attribute; the mutant would evaluate `exc.func.id` and raise AttributeError.
    sp = SelfPreservation(base_dir=tmp_path)
    safe, rep = sp.is_modification_safe("x.py", "def f():\n    raise obj.method()\n")
    assert isinstance(safe, bool)  # completed without crashing
    # Not a lethal pattern -> should be considered safe.
    assert safe is True


def test_nested_attribute_call_does_not_crash(tmp_path):
    # Kills L79 And->Or: `a.b.c()` reaches the bare-name branch with func being
    # an Attribute; the mutant would evaluate `func.id` and raise.
    sp = SelfPreservation(base_dir=tmp_path)
    safe, rep = sp.is_modification_safe("y.py", "def f():\n    a.b.c()\n")
    assert isinstance(safe, bool)
    assert safe is True


def test_lethal_attr_call_still_detected(tmp_path):
    # Guard: a genuine lethal call must still be caught (os.kill).
    sp = SelfPreservation(base_dir=tmp_path)
    safe, rep = sp.is_modification_safe("z.py", "def f():\n    os.kill(1, 9)\n")
    assert safe is False
    assert any("os.kill" in c for c in rep["critical"])


# ── Lockdown gate (L169) ──────────────────────────────────────────────

def test_lockdown_marks_report_unsafe(tmp_path):
    # Kills the `report["safe"] = False` -> True mutant in the lockdown branch.
    sp = SelfPreservation(base_dir=tmp_path)
    sp.activate_lockdown()
    safe, rep = sp.is_modification_safe("x.py", "def f():\n    pass\n")
    assert safe is False
    assert rep["safe"] is False
    assert rep["trust_score"] == 0.0


# ── Size-reduction heuristic (L204: And->Or and Mult->Div) ────────────

def test_no_size_warning_when_reduction_under_half(tmp_path):
    # Kills both L204 mutants. old=100 chars, new=60 chars: 60 is NOT < 0.5*100,
    # so there must be NO size-reduction warning. The And->Or mutant (fires on
    # old_size>0 alone) and the Mult->Div mutant (compares against old*2=200)
    # would both wrongly warn.
    sp = SelfPreservation(base_dir=tmp_path)
    (tmp_path / "x.py").write_text("a" * 100, encoding="utf-8")
    safe, rep = sp.is_modification_safe("x.py", "b" * 60)
    assert not any("reduced" in w.lower() for w in rep["warnings"])


def test_size_warning_when_reduction_over_half(tmp_path):
    # Guard: a genuine >50% shrink DOES warn (old=100, new=40).
    sp = SelfPreservation(base_dir=tmp_path)
    (tmp_path / "x.py").write_text("a" * 100, encoding="utf-8")
    safe, rep = sp.is_modification_safe("x.py", "b" * 40)
    assert any("reduced" in w.lower() for w in rep["warnings"])


# ── Empty-stub heuristic (L211: And->Or) ──────────────────────────────

def test_no_stub_warning_without_def(tmp_path):
    # Kills L211 And->Or: many `pass` but no `def ` must NOT warn.
    sp = SelfPreservation(base_dir=tmp_path)
    safe, rep = sp.is_modification_safe("x.py", "pass\npass\npass\npass\n")
    assert not any("stub" in w.lower() for w in rep["warnings"])


def test_stub_warning_with_def_and_many_pass(tmp_path):
    # Guard: real stub replacement (def + >3 pass) warns.
    sp = SelfPreservation(base_dir=tmp_path)
    code = "def a(): pass\ndef b(): pass\ndef c(): pass\ndef d(): pass\n"
    safe, rep = sp.is_modification_safe("x.py", code)
    assert any("stub" in w.lower() for w in rep["warnings"])


# ── Emergency memory cleanup threshold (L275: Gt->LtE) ────────────────

def test_emergency_cleanup_trims_over_threshold(tmp_path):
    # Kills the `len(episodic) > 500` Gt->LtE mutant: with 600 low-importance
    # episodes, cleanup must trim down to the recent-200 window.
    sp = SelfPreservation(base_dir=tmp_path)
    sub = _FakeSubstrate(_FakeMemory(600))
    sp._emergency_memory_cleanup(sub)
    assert len(sub.memory.episodic) == 200  # important(0) + recent(200)


def test_emergency_cleanup_leaves_small_memory(tmp_path):
    # Guard: under the threshold, nothing is trimmed.
    sp = SelfPreservation(base_dir=tmp_path)
    sub = _FakeSubstrate(_FakeMemory(300))
    sp._emergency_memory_cleanup(sub)
    assert len(sub.memory.episodic) == 300
