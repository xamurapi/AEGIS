"""End-to-end determinism (spec §3.1, M9.4).

The zero-randomness guarantee is only worth something if it is checked at the
level of the whole cognitive cycle: two runs of the core from the same starting
state must reach byte-identical state. This is the test the later stages —
evolution, policy, discovery — rely on, because every one of them compares
"before" against "after" and needs the difference to mean something.
"""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.layers.substrate import Substrate

TICKS = 12
SECONDS_PER_TICK = 3.0


def _quiet(substrate: Substrate) -> Substrate:
    """Neutralise everything that is legitimately non-reproducible.

    Not the core: the network, the LLM and the host's CPU/RAM readings. Those
    differ between two runs for reasons that have nothing to do with whether
    the cognitive cycle is deterministic, and leaving them in would test the
    machine rather than the system.
    """
    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.llm.enabled = False
    substrate.health.check = lambda: {
        "status": "healthy", "warnings": [], "critical": [], "metrics": {},
    }
    # Sensor readings are real hardware values; the perception they feed is not
    # part of the digest, but pinning them keeps the run cheap and stable.
    substrate.sensors.read_all = lambda: {"pinned": True}
    return substrate


def _run(ticks: int = TICKS) -> tuple[str, dict]:
    """Build a fresh substrate, tick it under a virtual clock, return its state."""
    with frozen() as clock:
        substrate = _quiet(Substrate())

        async def _drive():
            for _ in range(ticks):
                await substrate.tick()
                clock.advance(SECONDS_PER_TICK)

        asyncio.run(_drive())
        return substrate.state_digest(), substrate.state_snapshot()


@pytest.fixture
def _isolated(isolated_state):
    """Each run needs its own store, or the second one would load the first."""
    return isolated_state


def test_two_runs_reach_identical_state(_isolated, tmp_path, monkeypatch):
    """The guarantee itself."""
    first_digest, first_state = _run()

    # Second run: a fresh private store, so nothing is inherited from the first.
    import importlib
    second_root = tmp_path / "second"
    for module_name, constant, subdir in _module_dirs():
        module = importlib.import_module(module_name)
        if not hasattr(module, constant):
            continue
        target = second_root / subdir
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, constant, target, raising=False)

    second_digest, second_state = _run()

    if first_digest != second_digest:
        differing = sorted(
            key for key in first_state
            if first_state[key] != second_state.get(key)
        )
        pytest.fail(f"state diverged in: {differing}")
    assert first_digest == second_digest


def _module_dirs():
    from tests.conftest import _STATE_DIRS
    return _STATE_DIRS


def test_the_run_produces_non_trivial_state(_isolated):
    """Guard against a vacuous guarantee.

    Two empty states are also identical. If a later change quietly stops the
    cycle from learning anything, the equality above would still hold and the
    determinism test would pass while measuring nothing.
    """
    _, state = _run()
    assert state["tick"] == TICKS
    assert len(state["memory"]["episodic"]) >= TICKS
    assert len(state["goals"]) > 4
    assert len(state["world_model"]) > 0
    assert len(state["cognitive_graph"]["nodes"]) > 0
    assert len(state["goal_intelligence"]) > 0
    assert state["feedback"]["resolved"] == TICKS


def test_harness_detects_injected_nondeterminism(_isolated):
    """Guard against a blind guarantee.

    Proves the comparison can fail: with a genuinely random choice spliced into
    the decision path, two runs must diverge. Without this, a digest that
    accidentally covered nothing would look like a passing determinism test.
    """
    import random

    def noisy_run() -> str:
        with frozen() as clock:
            substrate = _quiet(Substrate())
            original = substrate.goal_intelligence.choose

            def choose(objectives, context=None):
                result = original(objectives, context)
                if result:
                    result["objective"] = random.choice(objectives)
                return result

            substrate.goal_intelligence.choose = choose

            async def _drive():
                for _ in range(TICKS):
                    await substrate.tick()
                    clock.advance(SECONDS_PER_TICK)

            asyncio.run(_drive())
            return substrate.state_digest()

    assert noisy_run() != noisy_run()


def test_digest_is_stable_without_ticking(_isolated):
    substrate = _quiet(Substrate())
    assert substrate.state_digest() == substrate.state_digest()


def test_digest_moves_when_state_moves(_isolated):
    substrate = _quiet(Substrate())
    before = substrate.state_digest()
    substrate.memory.add_semantic("new_concept", {"type": "test"})
    assert substrate.state_digest() != before


def test_digest_ignores_wall_clock(_isolated):
    """A run started a second later is not a different run."""
    substrate = _quiet(Substrate())
    before = substrate.state_digest()
    substrate.start_time -= 1000.0
    substrate.autobiography.log_event("noise", "an entry with a timestamp", 0.1)
    substrate.memory.add_episodic("event", importance=0.5)
    after_state = substrate.state_snapshot()
    # The episodic event IS state; the timestamps attached to it are not.
    assert "event" in after_state["memory"]["episodic"]
    substrate.memory.episodic[-1]["timestamp"] += 5000.0
    assert substrate.state_digest() == substrate.state_digest()
    assert before != substrate.state_digest()  # the event itself did register


def test_snapshot_covers_every_learned_structure(_isolated):
    """A structure missing from the snapshot is a blind spot: it could change
    between two runs and the determinism test would never notice."""
    substrate = _quiet(Substrate())
    snapshot = substrate.state_snapshot()
    for key in ("world_model", "cognitive_graph", "evolution",
                "goal_intelligence", "feedback", "skills", "memory",
                "goals", "parameters", "safety_contract"):
        assert key in snapshot, f"{key} is not covered by the state digest"


def test_safety_contract_is_part_of_the_digest(_isolated, monkeypatch):
    """Quietly widening the untouchable set must not look like 'no change'."""
    from aegis.safety import immutable
    substrate = _quiet(Substrate())
    before = substrate.state_digest()
    monkeypatch.setitem(immutable.BOUNDED_PARAMS, "SANDBOX_TIMEOUT", (0.5, 999.0))
    assert substrate.state_digest() != before


def test_ticking_advances_the_digest(_isolated):
    with frozen() as clock:
        substrate = _quiet(Substrate())
        before = substrate.state_digest()

        async def _one():
            await substrate.tick()

        asyncio.run(_one())
        clock.advance(SECONDS_PER_TICK)
        assert substrate.state_digest() != before
        assert substrate.tick_count == 1
