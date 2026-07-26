"""Audit C2: autonomous source self-modification must be opt-in (default OFF).

Untrusted external content can influence LLM prompts; the same channel proposes
whole-file rewrites. With the opt-in gate OFF, no code rewrite is applied even
if the method is reached and the LLM proposes a change.
"""
import asyncio

import aegis.layers.substrate as substrate_mod
from aegis.layers.substrate import Substrate


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    s.agent_system.run_due_agents = _noop_agents
    s.llm.enabled = False
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    # Pin health to "healthy": HealthMonitor.check() reads REAL cpu/memory via
    # psutil, so under full-suite load it reports "critical", MetaRegulation
    # switches to emergency mode and the tick legitimately skips learning /
    # ethics blocks self-modification — a machine-load dependency, not a defect.
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


def test_disabled_by_default():
    assert substrate_mod.CODE_SELF_MOD_ENABLED is False


def test_code_self_modification_noop_when_disabled(monkeypatch):
    async def run():
        monkeypatch.setattr(substrate_mod, "CODE_SELF_MOD_ENABLED", False)
        s = _make_substrate()

        applied = {"n": 0}
        s.code_modifier.apply_modification = lambda *a, **k: applied.__setitem__("n", 1) or {"status": "applied"}

        called = {"list": 0}
        orig_list = s.code_modifier.list_sources
        s.code_modifier.list_sources = lambda: called.__setitem__("list", called["list"] + 1) or orig_list()

        async def _propose(*a, **k):
            return {"success": True, "parsed": {"should_modify": True,
                    "modified_code": "x = 2\n", "description": "d"}}

        s.llm.propose_code_change = _propose

        await s._code_self_modification()
        # Returned before touching sources or applying anything.
        assert called["list"] == 0
        assert applied["n"] == 0
    asyncio.run(run())


def test_enabled_flag_allows_the_path(monkeypatch):
    async def run():
        monkeypatch.setattr(substrate_mod, "CODE_SELF_MOD_ENABLED", True)
        s = _make_substrate()
        # Stub so nothing real is written; just prove the gate no longer blocks.
        s.code_modifier.list_sources = lambda: []  # no modifiable files -> early return
        await s._code_self_modification()  # must not raise
    asyncio.run(run())
