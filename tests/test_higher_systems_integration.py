"""Integration tests: the 5 higher-order systems wired into the Substrate tick.

Verifies that a real PERCEIVE..REFLECT cycle drives World Model, Cognitive
Graph, Goal Intelligence, Feedback Loop and (on cadence) the Evolution Engine,
without blocking or crashing the loop. Network/LLM/sandbox are neutralized so
the test is deterministic and offline.
"""
import asyncio
import tempfile
from pathlib import Path

from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.substrate import Substrate


def _make_substrate():
    s = Substrate()

    # Substrate() reads the developer's working `data/` directory, so an
    # evolution candidate left there by ANY earlier run — a real session, a
    # soak, someone poking at a REPL — is restored on construction, and the
    # cadence tick then finds a candidate already pending and proposes nothing.
    # The tests below assert on evolution state, so per the rule in docs/QA.md
    # §6 they have to own that state. This gave a genuine false failure once
    # the round-5 fix made persisted candidates survive a restart, which is
    # exactly when a shared-state test starts lying.
    s.evolution = EvolutionEngine(
        store_path=Path(tempfile.mkdtemp(prefix="aegis_evo_test_")) / "lineage.json")
    # Re-do what Substrate.__init__ does for a fresh engine: without a champion
    # there is nothing for a mutation to be proposed against.
    s.evolution.evaluator.pool = s.eval_pool
    s.evolution.register_champion(s._genome.to_dict(), 0.0)

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
    # Pin health to "healthy": HealthMonitor.check() reads REAL cpu/memory via
    # psutil, so under full-suite load it reports "critical", MetaRegulation
    # switches to emergency mode and the tick legitimately skips learning /
    # ethics blocks self-modification — a machine-load dependency, not a defect.
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
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
    from aegis.layers.evolution.genome import GENE_NAMES, RETIRED_GENES

    s = _make_substrate()
    assert s.evolution.champion is not None
    # The genome is the fixed schema of Appendix C, not a mirror of the LoRA
    # parameters: those are still tunable through `parametric_self_mod`, but
    # nothing the benchmark measures reads them, so evolution no longer selects
    # on them (§M5.3).
    assert set(s.evolution.champion["genome"]) == set(GENE_NAMES)
    assert RETIRED_GENES & set(s.self_mod.parameters)
    assert not (RETIRED_GENES & set(GENE_NAMES))


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
        before = s.current_genome()
        await s.tick()  # lands on cadence -> proposes a mutation
        assert s.evolution.candidate is not None
        param = s.evolution.candidate["mutated_param"]
        assert s.current_genome() != before      # the variant is live

        # Simulate a benchmark that makes the mutation look worse -> rejected
        # and the parameter reverted to its old value. Stub the evaluator so
        # _run_benchmark uses a deterministic, low score.
        s.evaluator.run = lambda: {"score": 0.0, "passed": 0, "total": 1}
        await s._run_benchmark()
        assert s.evolution.candidate is None
        # The WHOLE genome goes back, not just the gene that moved furthest: a
        # coordinate mutation moves every gene at once (§M5.4), so reverting one
        # would leave the rest of a rejected variant in place for good.
        assert s.current_genome() == before, (
            f"a rejected variant left {param!r} and possibly others behind")
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
