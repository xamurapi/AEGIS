"""The core runs with no model at all (spec §M8.1, M8.8, Appendix K stage 1).

This is the load-bearing claim of the whole cortex design: a model layer that
*improves* judgement rather than *providing* liveness. If every provider is
gone — no keys, no local server, every circuit open — the system must keep
ticking, every contour must take its deterministic path, and nothing may raise.

500 ticks, because a failure that only appears once the caches, buffers and
retention rules have turned over is exactly the failure worth catching.
"""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.cortex.router import Cortex, Role
from aegis.layers.substrate import Substrate
from tests.cortex_fakes import ScriptedProvider

OFFLINE_TICKS = 500


def _offline_substrate() -> Substrate:
    """A substrate with nothing to talk to and no host-dependent readings."""
    substrate = Substrate()

    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.sensors.read_all = lambda: {"pinned": True}
    substrate.health.check = lambda: {
        "status": "healthy", "warnings": [], "critical": [], "metrics": {},
    }
    # Total darkness: no legacy client, no cortex route.
    substrate.llm.enabled = False
    substrate.llm.deepseek.enabled = False
    substrate.llm.claude.enabled = False
    substrate.llm.local.enabled = False
    substrate.llm.cortex.configure_routes({})
    return substrate


@pytest.fixture
def offline(isolated_state):
    return _offline_substrate()


# ── the router degrades rather than failing ──────────────────────────

def test_a_cortex_with_no_providers_reports_no_roles():
    cortex = Cortex(providers={}, routes={})
    assert cortex.available_roles() == []
    assert cortex.enabled is False


def test_every_role_returns_none_when_nothing_is_configured():
    cortex = Cortex(providers={}, routes={})
    for role in Role:
        assert asyncio.run(cortex.call(role, [{"role": "user", "content": "q"}])) is None


def test_structured_output_returns_none_rather_than_a_default():
    # A fabricated "default" answer would be indistinguishable from a real one
    # downstream; None forces the caller onto its deterministic branch.
    cortex = Cortex(providers={}, routes={})
    assert asyncio.run(cortex.structured(
        Role.DEEP, [{"role": "user", "content": "q"}], "decision")) is None


def test_going_offline_mid_run_is_survivable():
    provider = ScriptedProvider("a", responses=["ok"])
    cortex = Cortex(providers={"a": provider}, routes={"deep": ["a"]})
    assert asyncio.run(cortex.call(Role.DEEP, [{"role": "user", "content": "q"}])) is not None

    provider._available = False
    cortex.configure_routes({"deep": ["a"]})
    assert asyncio.run(cortex.call(Role.DEEP, [{"role": "user", "content": "r"}])) is None


# ── the facade degrades too ──────────────────────────────────────────

def test_the_llm_facade_reports_an_error_instead_of_raising(offline):
    result = asyncio.run(offline.llm.think("anything"))
    assert result["success"] is False
    assert result["response"] == ""


def test_every_structured_method_survives_full_darkness(offline):
    calls = [
        offline.llm.evaluate_state({"tick": 1}),
        offline.llm.make_decision(["a", "b"], {"tick": 1}),
        offline.llm.reflect({"tick": 1}),
        offline.llm.generate_curiosity(["topic"]),
    ]
    for coro in calls:
        assert asyncio.run(coro)["success"] is False


def test_skill_and_coding_proposals_return_none_offline(offline):
    assert asyncio.run(offline.llm.propose_skill("calc", [])) is None
    assert asyncio.run(offline.llm.propose_coding_solution("f", "spec", [])) is None


# ── 500 ticks in the dark ────────────────────────────────────────────

def test_five_hundred_ticks_without_a_model(offline):
    errors_before = offline.health.error_count
    with frozen() as clock:
        async def _drive():
            for _ in range(OFFLINE_TICKS):
                await offline.tick()
                clock.advance(3.0)

        asyncio.run(_drive())

    assert offline.tick_count == OFFLINE_TICKS
    assert offline.health.error_count == errors_before, "a tick failed while offline"


def test_the_deterministic_contours_still_did_work(offline):
    with frozen() as clock:
        async def _drive():
            for _ in range(120):
                await offline.tick()
                clock.advance(3.0)

        asyncio.run(_drive())

    # Not merely "did not crash": the causal model, the graph, the experience
    # log and the value table all have to keep learning without any model.
    assert offline.world_model.total_observations > 0
    assert offline.feedback_loop.resolved > 0
    assert offline.goal_intelligence.values
    assert len(offline.memory.episodic) > 0


def test_telemetry_keeps_recording_offline(offline):
    from aegis.telemetry import metrics as M
    with frozen() as clock:
        async def _drive():
            for _ in range(20):
                await offline.tick()
                clock.advance(3.0)

        asyncio.run(_drive())
    offline.telemetry.flush()
    assert len(offline.telemetry.series(M.TICK_DURATION_MS)) == 20


def test_status_is_still_assemblable_offline(offline):
    asyncio.run(offline.tick())
    status = offline.full_status()
    assert status["llm"]["enabled"] is False
    assert status["llm"]["cortex"]["available_roles"] == []
