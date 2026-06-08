"""The scheduled LoRA training must NOT block the cognitive tick loop.

Regression test for the audit follow-up: training is spawned as a detached
background task so ticks (and WS updates) keep flowing while it runs.
"""
import asyncio
import time

from aegis.layers.substrate import Substrate
from aegis.config import TRAIN_EVERY_N_TICKS


def _make_substrate():
    s = Substrate()

    # Neutralize network-bound helpers for a deterministic offline test.
    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.llm.enabled = False
    # Neutralize the capability-layer env step (it runs a sandbox subprocess) so
    # the tick-timing assertion isolates training scheduling, not env work.
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    # Deterministically approve weight modification so the test exercises the
    # scheduling/threading behavior, not the ethics gate.
    s.ethics.evaluate_weight_modification = lambda info: {"status": "approved", "score": 1.0}
    return s


def test_training_does_not_block_tick():
    async def run():
        s = _make_substrate()
        started = asyncio.Event()
        finished = {"done": False}

        async def fake_training(*args, **kwargs):
            started.set()
            await asyncio.sleep(1.5)  # stand in for a long LoRA run
            finished["done"] = True
            return {"status": "applied", "train_loss": 0.1}

        s.self_mod.propose_weight_modification = fake_training
        # Force the next tick to hit the training cadence.
        s.tick_count = TRAIN_EVERY_N_TICKS - 1

        t0 = time.time()
        await s.tick()  # increments to a multiple of TRAIN_EVERY_N_TICKS
        elapsed = time.time() - t0

        # The tick must return quickly — NOT wait the 1.5s "training".
        assert elapsed < 1.0, f"tick blocked for {elapsed:.2f}s"
        # A detached task was spawned and is still running (the long sleep).
        assert s._weight_training_task is not None, "training task was not spawned"
        assert not s._weight_training_task.done(), "training ran synchronously (blocked the tick)"
        assert finished["done"] is False

        # The detached task completes on its own afterwards.
        await s._weight_training_task
        assert started.is_set()
        assert finished["done"] is True

    asyncio.run(run())


def test_no_overlapping_training_runs():
    async def run():
        s = _make_substrate()
        calls = {"n": 0}

        async def fake_training(*args, **kwargs):
            calls["n"] += 1
            await asyncio.sleep(0.5)
            return {"status": "applied"}

        s.self_mod.propose_weight_modification = fake_training

        # First trigger spawns a run.
        s.tick_count = TRAIN_EVERY_N_TICKS - 1
        await s.tick()
        task_a = s._weight_training_task
        assert task_a is not None, "first training run was not spawned"

        # Make the very next tick also land on the cadence while the first run
        # is still in flight; it must NOT start a second overlapping run — the
        # task handle must be the same object.
        s.tick_count = (2 * TRAIN_EVERY_N_TICKS) - 1
        await s.tick()
        assert s._weight_training_task is task_a, "a second overlapping run was spawned"

        await task_a
        assert calls["n"] == 1, f"expected exactly 1 training run, got {calls['n']}"

    asyncio.run(run())
