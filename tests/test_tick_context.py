"""TickContext and the phase decomposition (spec §3.9)."""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.layers.phases import act, decide, evaluate, perceive, reflect
from aegis.layers.phases.context import PHASE_ORDER, TickContext
from aegis.layers.substrate import Substrate


# ── the context object ───────────────────────────────────────────────

def test_fresh_context_starts_empty():
    ctx = TickContext(tick=7)
    assert ctx.tick == 7
    assert ctx.new_concepts == ctx.new_episodic == ctx.llm_insights == 0
    assert ctx.external == set()
    assert ctx.durations_ms == {}


def test_external_marking_is_per_phase():
    ctx = TickContext()
    ctx.mark_external("act")
    assert ctx.did_external("act")
    assert not ctx.did_external("decide")


def test_marking_the_same_phase_twice_is_idempotent():
    ctx = TickContext()
    ctx.mark_external("act")
    ctx.mark_external("act")
    assert ctx.external == {"act"}


def test_durations_are_recorded_per_phase():
    ctx = TickContext()
    ctx.record_duration("decide", 12.5)
    assert ctx.durations_ms["decide"] == 12.5


@pytest.mark.parametrize("concepts,insights,expected", [
    (0, 0, False), (1, 0, True), (0, 1, True), (2, 3, True),
])
def test_learned_something_reflects_the_accumulators(concepts, insights, expected):
    ctx = TickContext(new_concepts=concepts, llm_insights=insights)
    assert ctx.learned_something() is expected


def test_summary_is_json_friendly():
    ctx = TickContext(tick=3, new_concepts=1)
    ctx.mark_external("act")
    ctx.record_duration("act", 1.23456)
    summary = ctx.summary()
    assert summary["tick"] == 3
    assert summary["external_phases"] == ["act"]
    assert summary["durations_ms"]["act"] == 1.235


def test_phase_order_matches_the_cycle():
    assert PHASE_ORDER == ("perceive", "evaluate", "decide", "act", "reflect")


# ── substrate integration ────────────────────────────────────────────

def _quiet(substrate: Substrate) -> Substrate:
    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.llm.enabled = False
    substrate.sensors.read_all = lambda: {"pinned": True}
    substrate.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    return substrate


def test_legacy_accumulator_names_still_work(isolated_state):
    """The phases, the dashboard and the existing suite address these by their
    original names; moving them onto the context must be invisible."""
    substrate = _quiet(Substrate())
    substrate._tick_new_concepts = 4
    assert substrate._ctx.new_concepts == 4
    substrate._ctx.llm_insights = 2
    assert substrate._tick_llm_insights == 2
    substrate._regulation_directives = {"skip_llm": True}
    assert substrate._ctx.regulation_directives["skip_llm"]
    substrate._pending_experiences["decide"] = "exp_1"
    assert substrate._ctx.pending_experiences["decide"] == "exp_1"


def test_each_tick_gets_a_fresh_context(isolated_state):
    substrate = _quiet(Substrate())

    async def _two():
        await substrate.tick()
        first = substrate._ctx
        await substrate.tick()
        return first, substrate._ctx

    first, second = asyncio.run(_two())
    assert first is not second
    assert second.tick == first.tick + 1


def test_accumulators_do_not_leak_between_ticks(isolated_state):
    substrate = _quiet(Substrate())

    async def _run():
        await substrate.tick()
        substrate._tick_new_concepts = 99
        await substrate.tick()
        return substrate._tick_new_concepts

    assert asyncio.run(_run()) != 99


def test_phase_modules_expose_a_uniform_entry_point():
    for module in (perceive, evaluate, decide, act, reflect):
        assert asyncio.iscoroutinefunction(module.run)


def test_wrappers_delegate_to_the_phase_modules(isolated_state, monkeypatch):
    """The substrate's _perceive.._reflect must be pure delegation — if one of
    them grew its own logic the decomposition would be undone silently."""
    substrate = _quiet(Substrate())
    called = []

    async def _spy(name):
        async def _run(sub, ctx):
            called.append((name, sub is substrate, ctx is substrate._ctx))
        return _run

    for name, module in (("perceive", perceive), ("evaluate", evaluate),
                         ("decide", decide), ("act", act), ("reflect", reflect)):
        monkeypatch.setattr(module, "run", asyncio.run(_spy(name)))

    async def _cycle():
        await substrate._perceive()
        await substrate._evaluate()
        await substrate._decide()
        await substrate._act()
        await substrate._reflect()

    asyncio.run(_cycle())
    assert [c[0] for c in called] == list(PHASE_ORDER)
    assert all(c[1] and c[2] for c in called)


def test_tick_measures_every_phase(isolated_state):
    with frozen() as clock:
        substrate = _quiet(Substrate())

        async def _one():
            await substrate.tick()

        asyncio.run(_one())
        clock.advance(3.0)

    assert set(substrate._ctx.durations_ms) == set(PHASE_ORDER)


def test_a_failing_phase_is_still_measured(isolated_state):
    """Timing lives in the runner's finally block: a phase that raises must not
    disappear from the latency record, or a crash loop would look fast."""
    substrate = _quiet(Substrate())

    async def _boom():
        raise RuntimeError("phase exploded")

    async def _run():
        await substrate._run_phase("decide", _boom)

    with pytest.raises(RuntimeError):
        asyncio.run(_run())
    assert "decide" in substrate._ctx.durations_ms
    assert substrate.health.phase_report()["decide"]["samples_all"] == 1
