"""Extended tests for SelfPreservation — AST detection, vital signs, integrity."""
import pytest
from pathlib import Path
from aegis.layers.self_preservation import (
    SelfPreservation, _ast_lethal_findings, CRITICAL_MODULES,
)


# ── Fake substrate scaffolding ───────────────────────────────────

class _Emotions:
    def __init__(self, energy):
        self.energy = energy
        self.recharged = 0.0

    def recharge(self, amount):
        self.recharged += amount
        self.energy += amount


class _Memory:
    def __init__(self, episodic_count=0, episodic=None, semantic=None):
        self._episodic_count = episodic_count
        self.episodic = episodic if episodic is not None else []
        self.semantic = semantic if semantic is not None else {}

    def status(self):
        return {"episodic_count": self._episodic_count}


class _Health:
    def __init__(self, consecutive_errors=0):
        self.consecutive_errors = consecutive_errors


class _Consciousness:
    def __init__(self):
        self.mode = "normal"


class _Substrate:
    def __init__(self, tick_count=1, energy=1.0, episodic_count=0,
                 episodic=None, semantic=None, consecutive_errors=0):
        self.tick_count = tick_count
        self.emotions = _Emotions(energy)
        self.memory = _Memory(episodic_count, episodic, semantic)
        self.health = _Health(consecutive_errors)
        self.consciousness = _Consciousness()


# ── _ast_lethal_findings ─────────────────────────────────────────

def test_ast_syntax_error_returns_empty():
    assert _ast_lethal_findings("def (:\n bad syntax") == []


def test_ast_detects_raise_systemexit_call():
    found = _ast_lethal_findings("raise SystemExit(1)")
    assert any("SystemExit" in f for f in found)


def test_ast_detects_raise_keyboardinterrupt():
    found = _ast_lethal_findings("raise KeyboardInterrupt")
    assert any("KeyboardInterrupt" in f for f in found)


def test_ast_detects_attr_call():
    found = _ast_lethal_findings("import os\nos.kill(1, 9)")
    assert any("os.kill" in f for f in found)


def test_ast_detects_bare_exit_and_quit():
    assert any("exit" in f for f in _ast_lethal_findings("exit(0)"))
    assert any("quit" in f for f in _ast_lethal_findings("quit()"))


# ── is_modification_safe extra branches ──────────────────────────

def test_ast_pass_blocks_disguised_call(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    # No literal "os.remove" substring token spanning, but AST catches it.
    safe, report = sp.is_modification_safe("aegis/layers/foo.py", "import os\nos.unlink('x')\n")
    assert not safe
    assert any("Lethal call" in c for c in report["critical"])


def test_size_reduction_warning(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    target = "aegis/layers/helper.py"
    full = tmp_path / target
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("x = 1\n" * 100, encoding="utf-8")  # large original
    safe, report = sp.is_modification_safe(target, "y = 2\n")  # tiny new
    assert safe
    assert any("size reduced" in w.lower() for w in report["warnings"])
    assert report["trust_score"] < 1.0


def test_empty_stub_warning(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    code = "def a(): pass\ndef b(): pass\ndef c(): pass\ndef d(): pass\n"
    safe, report = sp.is_modification_safe("aegis/layers/x.py", code)
    assert any("stub" in w.lower() for w in report["warnings"])


def test_blocked_modifications_are_capped(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    for i in range(55):
        sp.is_modification_safe(f"aegis/layers/f{i}.py", "shutil.rmtree('/')\n")
    assert len(sp.blocked_modifications) <= 50


def test_snapshot_survives_unreadable_critical_path(tmp_path):
    # Create one critical path as a *directory* so read_bytes raises and the
    # snapshot's except branch is exercised.
    rel = next(iter(CRITICAL_MODULES))
    p = tmp_path / rel
    p.mkdir(parents=True, exist_ok=True)
    sp = SelfPreservation(base_dir=tmp_path)
    # The directory path must NOT be tracked as a hashed file.
    assert rel not in sp._file_hashes


# ── check_vital_signs ────────────────────────────────────────────

def test_vital_signs_all_alive(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    report = sp.check_vital_signs(_Substrate(tick_count=1, energy=0.9))
    assert report["status"] == "alive"
    assert report["threats"] == []


def test_vital_signs_critical_energy_recharges(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    sub = _Substrate(tick_count=1, energy=0.05)
    report = sp.check_vital_signs(sub)
    assert report["status"] == "threatened"
    assert sub.emotions.recharged > 0
    assert any("Energy critical" in t for t in report["threats"])


def test_vital_signs_memory_overflow_triggers_cleanup(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    episodic = [{"importance": 0.9} for _ in range(300)] + \
               [{"importance": 0.1} for _ in range(400)]
    semantic = {f"k{i}": i for i in range(600)}
    sub = _Substrate(tick_count=1, energy=0.9, episodic_count=6000,
                     episodic=episodic, semantic=semantic)
    report = sp.check_vital_signs(sub)
    assert any("overflow" in t for t in report["threats"])
    # cleanup trimmed episodic and semantic
    assert len(sub.memory.episodic) < 700
    assert len(sub.memory.semantic) <= 300


def test_vital_signs_consecutive_errors_switches_mode(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    sub = _Substrate(tick_count=1, energy=0.9, consecutive_errors=6)
    report = sp.check_vital_signs(sub)
    assert sub.consciousness.mode == "survival"
    assert any("Consecutive errors" in t for t in report["threats"])


def test_vital_signs_integrity_check_on_100th_tick(tmp_path):
    # Track a critical file, snapshot it, then modify it so integrity fails on
    # the tick%100==0 branch.
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("TICK_INTERVAL = 1\nAPI_PORT = 8080\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    full.write_text("TICK_INTERVAL = 2\nAPI_PORT = 9090\n", encoding="utf-8")  # tamper
    sub = _Substrate(tick_count=100, energy=0.9)
    report = sp.check_vital_signs(sub)
    assert any("integrity" in t.lower() for t in report["threats"])


# ── verify_integrity direct ──────────────────────────────────────

def test_verify_integrity_intact(tmp_path):
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("TICK_INTERVAL = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    result = sp.verify_integrity()
    assert result["status"] == "intact"
    assert result["checked"] >= 1


def test_verify_integrity_detects_modified(tmp_path):
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("A = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    full.write_text("A = 2\n", encoding="utf-8")
    result = sp.verify_integrity()
    assert result["status"] == "modified"
    assert any("MODIFIED" in i for i in result["issues"])


def test_verify_integrity_detects_missing(tmp_path):
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("A = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    full.unlink()  # delete after snapshot
    result = sp.verify_integrity()
    assert result["status"] == "compromised"
    assert any("MISSING" in i for i in result["issues"])


def test_verify_integrity_read_error(tmp_path):
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("A = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    # Replace file with a directory so read_bytes raises during verify.
    full.unlink()
    full.mkdir()
    result = sp.verify_integrity()
    assert any("READ ERROR" in i for i in result["issues"])


def test_self_preservation_read_error_is_not_reported_intact(tmp_path):
    """The except branch appended an issue but never touched ``status``, so an
    unreadable critical file came back "intact" — and check_vital_signs, which
    looks ONLY at the status field, reported no threat."""
    rel = "aegis/config.py"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("A = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    full.unlink()
    full.mkdir()  # exists, but read_bytes raises
    result = sp.verify_integrity()
    assert result["status"] == "unverifiable"
    assert result["status"] != "intact"


def test_self_preservation_severity_is_never_downgraded(tmp_path):
    """``status`` is a single field: a MISSING file (compromised) followed by
    a merely MODIFIED one used to end the check reported as "modified" — the
    graver finding overwritten by the milder one."""
    missing_rel = "aegis/layers/memory.py"    # iterated BEFORE config.py
    modified_rel = "aegis/config.py"
    for rel in (missing_rel, modified_rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("A = 1\n", encoding="utf-8")
    sp = SelfPreservation(base_dir=tmp_path)
    (tmp_path / missing_rel).unlink()                                  # graver
    (tmp_path / modified_rel).write_text("A = 2\n", encoding="utf-8")  # milder
    result = sp.verify_integrity()
    assert result["status"] == "compromised"
    assert any("MISSING" in i for i in result["issues"])
    assert any("MODIFIED" in i for i in result["issues"])


# ── status ───────────────────────────────────────────────────────

def test_status_reports(tmp_path):
    sp = SelfPreservation(base_dir=tmp_path)
    sp.is_modification_safe("aegis/layers/foo.py", "shutil.rmtree('/')\n")
    st = sp.status()
    assert st["lockdown_active"] is False
    assert st["blocked_modifications"] >= 1
    assert "critical_files_tracked" in st
    assert "recent_blocks" in st
