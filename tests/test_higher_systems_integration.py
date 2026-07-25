"""Integration tests: the 5 higher-order systems wired into the Substrate tick.

Verifies that a real PERCEIVE..REFLECT cycle drives World Model, Cognitive
Graph, Goal Intelligence, Feedback Loop and (on cadence) the Evolution Engine,
without blocking or crashing the loop. Network/LLM/sandbox are neutralized so
the test is deterministic and offline.
"""
import asyncio

from aegis.layers.substrate import Substrate


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.llm.enabled = False
    # A solved env step so the World Model gets real cause->effect data.
    s.environment.step = lambda: {
        "reward": 1.0, "solved": True, "task": "t1",
        "kind": "arith", "winning_skill": "adder",
    }
    return s


def test_all_five_systems_present():
    s = _make_substrate()
    assert s.world_model is not None
    assert s.cognitive_graph is not None
    assert s.evolution is not None
    assert s.goal_intelligence is not None
    assert s.feedback_loop is not None


def test_full_status_exposes_five_systems():
    s = _make_substrate()
    st = s.full_status()
    for key in ("world_model", "cognitive_graph", "evolution",
                "goal_intelligence", "feedback_loop"):
        assert key in st, f"{key} missing from full_status()"


def test_evolution_champion_seeded_on_boot():
    s = _make_substrate()
    assert s.evolution.champion is not None
    # Genome mirrors the tunable self-mod parameters.
    assert set(s.evolution.champion["genome"]) == set(s.self_mod.parameters)


def test_tick_drives_world_model_and_feedback():
    async def run():
        s = _make_substrate()
        # Run enough ticks to cross the env-step + world-model cadences.
        for _ in range(6):
            await s.tick()
        # Env steps recorded cause->effect observations.
        assert s.world_model.total_observations > 0
        # Every decide opened an experience; every reflect closed one.
        assert s.feedback_loop.resolved > 0
        # Goal intelligence made value-driven choices.
        assert s.goal_intelligence.values
    asyncio.run(run())


def test_tick_populates_cognitive_graph():
    async def run():
        s = _make_substrate()
        for _ in range(9):  # cross COGNITIVE_GRAPH_EVERY_N_TICKS (8)
            await s.tick()
        assert len(s.cognitive_graph.nodes) > 0
    asyncio.run(run())


def test_evolution_proposes_and_is_judged():
    async def run():
        s = _make_substrate()
        # Force the evolution cadence on the next tick.
        from aegis.config import EVOLUTION_EVERY_N_TICKS
        s.tick_count = EVOLUTION_EVERY_N_TICKS - 1
        await s.tick()  # lands on cadence -> proposes a mutation
        assert s.evolution.candidate is not None
        param = s.evolution.candidate["mutated_param"]
        old = s.evolution.candidate["old_value"]

        # Simulate a benchmark that makes the mutation look worse -> rejected
        # and the parameter reverted to its old value. Stub the evaluator so
        # _run_benchmark uses a deterministic, low score.
        s.evaluator.run = lambda: {"score": 0.0, "passed": 0, "total": 1}
        await s._run_benchmark()
        assert s.evolution.candidate is None
        assert s.self_mod.parameters[param] == old
    asyncio.run(run())


def test_tick_never_raises_out():
    """The higher-systems hooks are guarded — even a broken sub-system must not
    escape tick() (it records a failed tick instead)."""
    async def run():
        s = _make_substrate()

        def _boom(*a, **k):
            raise RuntimeError("injected")

        s.world_model.observe = _boom
        s.cognitive_graph.ingest_memory = _boom
        # Should not raise despite the injected failures.
        for _ in range(4):
            await s.tick()
    asyncio.run(run())
