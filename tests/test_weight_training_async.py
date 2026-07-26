"""The scheduled LoRA training must NOT block the cognitive tick loop.

Regression test for the audit follow-up: training is spawned as a detached
background task so ticks (and WS updates) keep flowing while it runs.
"""
import asyncio

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
    # Pin health to "healthy". HealthMonitor.check() reads REAL cpu/memory via
    # psutil: while the full suite runs, CPU crosses the critical threshold,
    # MetaRegulation switches to emergency/eco and sets `skip_learning`, and the
    # tick then legitimately declines to start training — making this file fail
    # intermittently on machine load rather than on a defect.
    s.health.check = lambda: {"status": "healthy", "warnings": [], "critical": [], "metrics": {}}
    return s


def test_training_does_not_block_tick():
    async def run():
        s = _make_substrate()
        started = asyncio.Event()
        release = asyncio.Event()
        finished = {"done": False}

        async def fake_training(*args, **kwargs):
            started.set()
            # Stands in for a long LoRA run: it does not end until the test says
            # so. A fixed sleep would race a slow tick and make this flaky under
            # suite load; here, a tick that BLOCKS can never finish, so blocking
            # surfaces as a clean timeout instead of a timing-dependent verdict.
            await release.wait()
            finished["done"] = True
            return {"status": "applied", "train_loss": 0.1}

        s.self_mod.propose_weight_modification = fake_training
        # Force the next tick to hit the training cadence.
        s.tick_count = TRAIN_EVERY_N_TICKS - 1

        # increments to a multiple of TRAIN_EVERY_N_TICKS
        await asyncio.wait_for(s.tick(), timeout=10)

        # A detached task was spawned and is still running (training is held).
        assert s._weight_training_task is not None, "training task was not spawned"
        assert not s._weight_training_task.done(), "training ran synchronously (blocked the tick)"
        assert finished["done"] is False

        # The detached task completes on its own once released.
        release.set()
        await s._weight_training_task
        assert started.is_set()
        assert finished["done"] is True

    asyncio.run(run())


def test_no_overlapping_training_runs():
    async def run():
        s = _make_substrate()
        calls = {"n": 0}
        # The first run is held open by an Event the TEST releases, not by a
        # wall-clock sleep. With a sleep, a slow tick (the suite runs many
        # Substrates concurrently) could outlast it, the first run would finish
        # legitimately, and the second tick would then be entitled to spawn a
        # new run — making this test intermittently fail on load rather than on
        # a real defect.
        release = asyncio.Event()

        async def fake_training(*args, **kwargs):
            calls["n"] += 1
            await release.wait()
            return {"status": "applied"}

        s.self_mod.propose_weight_modification = fake_training

        # First trigger spawns a run.
        s.tick_count = TRAIN_EVERY_N_TICKS - 1
        await s.tick()
        task_a = s._weight_training_task
        assert task_a is not None, "first training run was not spawned"
        assert not task_a.done(), "the first run must still be in flight"

        # Make the very next tick also land on the cadence while the first run
        # is still in flight; it must NOT start a second overlapping run — the
        # task handle must be the same object.
        s.tick_count = (2 * TRAIN_EVERY_N_TICKS) - 1
        await s.tick()
        assert s._weight_training_task is task_a, "a second overlapping run was spawned"

        release.set()
        await task_a
        assert calls["n"] == 1, f"expected exactly 1 training run, got {calls['n']}"

    asyncio.run(run())
