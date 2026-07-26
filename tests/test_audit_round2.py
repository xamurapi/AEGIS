"""Tests for the second-round audit fixes (security bypasses + persistence)."""
import json

import pytest


# ── sandbox: scoped param exemption (CRITICAL RCE) ────────────────────

def test_sandbox_param_scope_rce_blocked():
    from aegis.eval.sandbox import check_safe, run_skill
    # A throwaway lambda param must NOT un-block the real builtin at module scope.
    exploit = "_ = lambda eval: None\ndef solve(p):\n    return eval(\"1+1\")\n"
    safe, reasons = check_safe(exploit)
    assert safe is False
    assert any("eval" in r for r in reasons)
    out = run_skill(exploit, "solve", None, timeout=3.0)
    assert out["ok"] is False


def test_sandbox_legit_param_shadow_still_allowed():
    from aegis.eval.sandbox import check_safe
    # `input` shadowed as a real parameter of solve must still pass.
    safe, _ = check_safe("def solve(input):\n    return sorted(input)\n")
    assert safe is True


def test_sandbox_breakpoint_blocked():
    from aegis.eval.sandbox import check_safe
    safe, _ = check_safe("def solve(p):\n    breakpoint()\n    return p\n")
    assert safe is False


# ── code_modifier: from-import / open / wildcard bypasses ─────────────

@pytest.fixture
def cm(tmp_path):
    from aegis.layers.code_modifier import CodeModifier
    base = tmp_path / "pkg"
    (base / "layers").mkdir(parents=True)
    (base / "layers" / "toy.py").write_text("x = 1\n", encoding="utf-8")
    return CodeModifier(base_dir=base, backups_dir=tmp_path / "b")


def _blocked(cm, code):
    safe, _ = cm.validate_safety(code, "layers/toy.py")
    return not safe


def test_from_os_import_aliased_call_blocked(cm):
    assert _blocked(cm, "from os import system as s\ndef f():\n    s('x')\n")
    assert _blocked(cm, "from os import remove\ndef f():\n    remove('/x')\n")


def test_wildcard_import_from_dangerous_module_blocked(cm):
    assert _blocked(cm, "from os import *\ndef f():\n    pass\n")


def test_io_open_write_blocked(cm):
    assert _blocked(cm, "import io\ndef f():\n    io.open('/x', 'w')\n")


def test_os_open_blocked(cm):
    assert _blocked(cm, "import os\ndef f():\n    os.open('/x', 1)\n")


def test_legit_os_read_still_allowed(cm):
    safe, _ = cm.validate_safety("import os\ndef f():\n    return os.getcwd()\n", "layers/toy.py")
    assert safe


# ── self_preservation: from-import lethal alias ───────────────────────

def test_self_preservation_from_import_lethal_blocked():
    from aegis.layers.self_preservation import _ast_lethal_findings
    assert _ast_lethal_findings("from os import kill\ndef f():\n    kill(1, 9)\n")
    assert _ast_lethal_findings("from shutil import rmtree\ndef f():\n    rmtree('/')\n")
    assert not _ast_lethal_findings("def f(x):\n    return x * 2\n")


# ── FeedbackLoop: counters restored on reload (M1) ────────────────────

def test_feedback_loop_counters_restored(tmp_path):
    from aegis.layers.feedback_loop import FeedbackLoop
    p = tmp_path / "exp.jsonl"
    fb = FeedbackLoop(store_path=p)
    for i in range(4):
        eid = fb.record_situation(f"s{i}", "d")
        fb.record_result(eid, success=(i < 3), metric=0.5)
    assert fb.resolved == 4 and fb.successes == 3

    fb2 = FeedbackLoop(store_path=p)  # fresh instance, same file
    assert fb2.resolved == 4
    assert fb2.successes == 3
    assert fb2.failures == 1
    assert fb2.success_rate() == 0.75
    # New ids continue past the stored sequence (no collision).
    new_id = fb2.record_situation("s", "d")
    assert new_id == "exp_00000005"


# ── Evolution parameters persisted across restart (H1) ────────────────

def test_evolution_parameters_survive_checkpoint(tmp_path):
    import aegis.layers.substrate as sub

    def _fresh():
        s = sub.Substrate()

        async def _na():
            return []
        s.agent_system.run_due_agents = _na
        s.llm.enabled = False
        s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
        s._checkpoint_path = tmp_path / "latest.json"
        return s

    s1 = _fresh()
    s1.self_mod.parameters["curiosity_weight"] = 0.777
    s1._save_checkpoint()

    s2 = _fresh()
    s2._restore_checkpoint()
    assert s2.self_mod.parameters["curiosity_weight"] == 0.777


# ── CognitiveGraph ingest survives episodic forgetting (M2) ───────────

def test_cognitive_graph_ingest_after_forgetting(tmp_path):
    from aegis.layers.cognitive_graph import CognitiveGraph

    class _Mem:
        def __init__(self):
            self.semantic = {"robotics": {"relations": {"type": "x"}}}
            self.episodic = []

    cg = CognitiveGraph(store_path=tmp_path / "g.json")
    mem = _Mem()
    # Ingest some events, then simulate forgetting (list shrinks from the front),
    # then add a NEW event — it must still be ingested (old index would skip it).
    mem.episodic = [{"event": f"robotics event {i}", "importance": 0.6} for i in range(10)]
    cg.ingest_memory(mem)
    mem.episodic = mem.episodic[5:]  # forgetting removed the first 5
    mem.episodic.append({"event": "robotics brand new discovery", "importance": 0.9})
    cg.ingest_memory(mem)
    assert any("brand new discovery" in n for n in cg.nodes)
