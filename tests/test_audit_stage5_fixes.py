"""Regressions for the defects the stage-5 audit found in the tick phases.

Each test here failed before its fix and would fail again if the fix were
reverted.

* **EVALUATE / list-shaped reply** — ``_parse_json_response`` returns whatever
  ``json.loads`` produced, and a truthy list sails through the ``parsed or
  {...}`` fallback in ``llm.evaluate_state``. EVALUATE then called ``.get`` on
  a list, and the AttributeError aborted DECIDE/ACT/REFLECT for the tick.
  reflect.py and decide.py both carried the ``isinstance(..., dict)`` guard;
  evaluate.py did not.
* **EVALUATE / non-string insight** — ``insight[:100]`` sliced whatever the
  model returned; a numeric insight raised TypeError with the same
  phase-aborting effect.
* **ACT / curiosity lease** — the topic was extracted with ``(... or
  {}).get("topic")`` *inside the settle argument*, so a list-shaped reply
  raised before ``settle`` ran: the lease stayed open, ``finalize_tick``
  charged the full estimate, and the rest of ACT was skipped.
"""
import asyncio

from aegis.config import ENV_STEP_EVERY_N_TICKS, LLM_THINK_EVERY_N_TICKS
from aegis.layers.substrate import Substrate


def _make_substrate() -> Substrate:
    s = Substrate()

    async def _noop_agents():
        return []

    async def _noop_learn(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _noop_agents
    s.external_learning.learn_from_source = _noop_learn
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    # Real psutil readings make health flap under full-suite load; pin it so the
    # test measures the branch it is about rather than the machine.
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


def _count_env_steps(s):
    stepped = {"n": 0}

    def _step():
        stepped["n"] += 1
        return {"reward": 0.0, "solved": False, "task": None}

    s.environment.step = _step
    return stepped


# ── EVALUATE: a malformed model reply must not abort the tick ────────

def test_evaluate_survives_a_list_shaped_llm_reply():
    async def run():
        s = _make_substrate()
        s.llm.enabled = True

        async def _eval(state, lease=None):
            # json.loads of a JSON array — a real failure mode, not a fake one.
            return {"success": True, "parsed": ["odd", "list"], "response": ""}

        s.llm.evaluate_state = _eval
        stepped = _count_env_steps(s)
        # A tick that is both an LLM tick and an environment-step tick, so a
        # crash in EVALUATE is observable as ACT never happening.
        cadence = LLM_THINK_EVERY_N_TICKS * max(1, ENV_STEP_EVERY_N_TICKS)
        s.tick_count = cadence - 1
        await s.tick()
        assert stepped["n"] == 1          # ACT still ran
        assert s.health.consecutive_errors == 0
    asyncio.run(run())


def test_evaluate_coerces_a_non_string_insight():
    async def run():
        s = _make_substrate()
        s.llm.enabled = True

        async def _eval(state, lease=None):
            return {"success": True, "parsed": {"insight": 42}, "response": ""}

        s.llm.evaluate_state = _eval
        stepped = _count_env_steps(s)
        cadence = LLM_THINK_EVERY_N_TICKS * max(1, ENV_STEP_EVERY_N_TICKS)
        s.tick_count = cadence - 1
        await s.tick()
        # The insight landed (coerced to text) AND the phase survived the
        # slice that used to raise TypeError on the way to the autobiography.
        assert any("LLM Insight: 42" in e["event"] for e in s.memory.episodic)
        assert stepped["n"] == 1
        assert s.health.consecutive_errors == 0
    asyncio.run(run())


# ── ACT: the curiosity lease is settled even for a malformed reply ───

def test_act_settles_the_curiosity_lease_on_a_list_shaped_reply():
    async def run():
        s = _make_substrate()
        s.llm.enabled = True

        async def _curious(known, lease=None):
            return {"success": True, "parsed": ["not", "an", "object"]}

        s.llm.generate_curiosity = _curious
        stepped = _count_env_steps(s)

        settled_purposes = []
        original_settle = s.settle

        def _spy(lease, actual=None, value=0.0):
            settled_purposes.append(lease.purpose)
            return original_settle(lease, actual, value)

        s.settle = _spy

        # The curiosity block needs an LLM tick that is also a multiple of
        # LLM_THINK_EVERY_N_TICKS * 5, and the tick must reach the env step.
        cadence = LLM_THINK_EVERY_N_TICKS * 5 * max(1, ENV_STEP_EVERY_N_TICKS)
        s.tick_count = cadence - 1
        await s.tick()

        # Before the fix the AttributeError fired while the settle argument
        # was being evaluated: the lease was never settled and the rest of
        # ACT was skipped.
        assert "curiosity_explore" in settled_purposes
        assert stepped["n"] == 1
        assert s.health.consecutive_errors == 0
    asyncio.run(run())
