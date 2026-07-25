"""Tests for the Substrate ACT-phase orchestration branches (offline).

Drives external learning, agent knowledge ingestion, motor logging, background
benchmark/skill-synthesis scheduling and weight-training scheduling by placing
the tick on the right cadence with all heavy/network work stubbed.
"""
import asyncio

from aegis.layers.substrate import Substrate
from aegis.config import TRAIN_EVERY_N_TICKS, EVAL_EVERY_N_TICKS


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.llm.enabled = False
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    return s


def test_external_learning_success_adds_concepts():
    async def run():
        s = _make_substrate()

        async def _learn(source, topic):
            return {"success": True, "concepts": ["quantum_x", "entropy_y", "graph_z"]}

        s.external_learning.learn_from_source = _learn
        s.tick_count = 39  # -> 40, hits the `% 40 == 0` learning cadence
        await s.tick()
        # At least one learned concept made it into semantic memory.
        assert any(c in s.memory.semantic for c in ("quantum_x", "entropy_y", "graph_z"))
    asyncio.run(run())


def test_agent_knowledge_ingested_into_memory():
    async def run():
        s = _make_substrate()

        async def _agents():
            return [{"agent": "arxiv_bot", "source": "arxiv", "items": 2}]

        s.agent_system.run_due_agents = _agents
        s.agent_system.get_recent_knowledge = lambda n: [
            {"agent": "arxiv_bot", "source": "arxiv",
             "data": {"title": "New Attention Variant", "summary": "faster transformer"}}]
        await s.tick()
        assert "New Attention Variant" in s.memory.semantic
    asyncio.run(run())


def test_benchmark_task_scheduled_on_cadence():
    async def run():
        s = _make_substrate()
        ran = {"n": 0}

        def _fake_eval_run():
            ran["n"] += 1
            return {"score": 0.5, "passed": 1, "total": 2}

        s.evaluator.run = _fake_eval_run
        s.tick_count = EVAL_EVERY_N_TICKS - 1
        await s.tick()
        assert s._eval_task is not None
        await s._eval_task  # let the detached benchmark finish
        assert ran["n"] == 1
        assert s._last_benchmark_score == 0.5
    asyncio.run(run())


def test_weight_training_scheduled_and_detached():
    async def run():
        s = _make_substrate()
        s.ethics.evaluate_weight_modification = lambda info: {"status": "approved", "score": 1.0}

        started = {"n": 0}

        async def _fake_training(*a, **k):
            started["n"] += 1
            return {"status": "applied", "train_loss": 0.1}

        s.self_mod.propose_weight_modification = _fake_training
        s.tick_count = TRAIN_EVERY_N_TICKS - 1
        await s.tick()
        assert s._weight_training_task is not None
        await s._weight_training_task
        assert started["n"] == 1
    asyncio.run(run())


def test_weight_training_blocked_by_ethics_does_not_start():
    async def run():
        s = _make_substrate()
        s.ethics.evaluate_weight_modification = lambda info: {"status": "blocked", "score": 0.0}
        s.tick_count = TRAIN_EVERY_N_TICKS - 1
        await s.tick()
        # Ethics blocked -> no background training task spawned this tick.
        assert s._weight_training_task is None
    asyncio.run(run())


def test_motor_log_tick_runs():
    async def run():
        s = _make_substrate()
        logged = {"n": 0}
        orig = s.motor.execute
        s.motor.execute = lambda *a, **k: logged.__setitem__("n", logged["n"] + 1) or orig(*a, **k)
        s.tick_count = 9  # -> 10, hits the `% 10 == 0` motor-log cadence
        await s.tick()
        assert logged["n"] >= 1
    asyncio.run(run())


def test_checkpoint_saved_on_cadence(tmp_path):
    async def run():
        s = _make_substrate()
        s._checkpoint_path = tmp_path / "latest.json"
        s.tick_count = 9  # -> 10, hits CHECKPOINT_EVERY_N_TICKS
        await s.tick()
        assert s._checkpoint_path.exists()
    asyncio.run(run())
