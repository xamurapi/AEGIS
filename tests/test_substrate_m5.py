"""Audit M5: malformed-but-plausible LLM JSON must not crash a tick.

The model may return numbers as strings, an object where a scalar was asked
for, or a scalar where an object was asked for. A tick that consumes those must
degrade gracefully, not raise (which would drop the rest of the cognitive
cycle and inflate the error rate).
"""
import asyncio

from aegis.layers.substrate import Substrate, _coerce_int, _coerce_float
from aegis.config import LLM_THINK_EVERY_N_TICKS


def _make_substrate():
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    s.llm.enabled = True

    # Malformed-but-plausible payloads for every LLM call the tick makes.
    async def _eval_state(state):
        return {"success": True, "response": "r", "parsed": {
            "assessment": "ok",
            "insight": "an insight",
            # a number and an object where strings were expected, mixed with a
            # valid one — all within the first two the tick consumes:
            "suggested_goals": [123, "valid_goal", {"x": 1}]}}

    async def _make_decision(options, ctx):
        return {"success": True, "response": "r", "parsed": {
            "chosen": "2",          # string, not int
            "confidence": "high",   # non-numeric
            "reasoning": 42}}       # non-string

    async def _reflect(ep):
        return {"success": True, "response": "r", "parsed": {
            "learning": 99,                 # non-string
            "knowledge": "not an object"}}  # string, not dict

    s.llm.evaluate_state = _eval_state
    s.llm.make_decision = _make_decision
    s.llm.reflect = _reflect
    # Pin health to "healthy": HealthMonitor.check() reads REAL cpu/memory via
    # psutil, so under full-suite load it reports "critical", MetaRegulation
    # switches to emergency mode and the tick legitimately skips learning /
    # ethics blocks self-modification — a machine-load dependency, not a defect.
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


def test_coerce_helpers():
    assert _coerce_int("2", 1) == 2
    assert _coerce_int("nope", 7) == 7
    assert _coerce_int(None, 3) == 3
    assert _coerce_float("0.5", 0.0) == 0.5
    assert _coerce_float("high", 0.9) == 0.9


def test_tick_survives_malformed_llm_json():
    async def run():
        s = _make_substrate()
        s.tick_count = LLM_THINK_EVERY_N_TICKS - 1  # next tick is an LLM tick
        before_fail = s.health.failed_ticks
        await s.tick()
        # The tick completed successfully despite the malformed JSON.
        assert s.health.failed_ticks == before_fail
        assert s.health.successful_ticks >= 1
    asyncio.run(run())


def test_valid_suggested_goal_still_used_amid_bad_ones():
    async def run():
        s = _make_substrate()
        s.tick_count = LLM_THINK_EVERY_N_TICKS - 1
        await s.tick()
        # The one valid string goal was added; the int and dict were skipped.
        names = [g.name for g in s.goals.goals]
        assert any("valid_goal" in n for n in names)
    asyncio.run(run())


def test_string_chosen_is_coerced_not_crashed():
    async def run():
        s = _make_substrate()
        s.tick_count = LLM_THINK_EVERY_N_TICKS - 1
        # Should not raise; "chosen":"2" coerces to index 1.
        await s.tick()
        assert s.health.failed_ticks == 0
    asyncio.run(run())
