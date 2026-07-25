"""Audit M6: shutdown must cancel detached background tasks."""
import asyncio

from aegis.layers.substrate import Substrate


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    s.agent_system.run_due_agents = _noop_agents
    s.llm.enabled = False
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    return s


def test_cancel_background_tasks_cancels_pending():
    async def run():
        s = _make_substrate()

        async def _long():
            await asyncio.sleep(30)

        s._eval_task = asyncio.create_task(_long())
        s._skill_synth_task = asyncio.create_task(_long())
        s._weight_training_task = asyncio.create_task(_long())

        await s.cancel_background_tasks()

        assert s._eval_task.done()
        assert s._skill_synth_task.done()
        assert s._weight_training_task.done()
        assert s._eval_task.cancelled()
    asyncio.run(run())


def test_cancel_background_tasks_tolerates_none_and_done():
    async def run():
        s = _make_substrate()
        s._eval_task = None

        async def _quick():
            return 1

        s._skill_synth_task = asyncio.create_task(_quick())
        await s._skill_synth_task  # already done
        s._weight_training_task = None
        # Must not raise on None / already-completed tasks.
        await s.cancel_background_tasks()
    asyncio.run(run())
