"""Regressions for the defects the stage-4 audit found.

Each test here failed before its fix and would fail again if the fix were
reverted — that is the only reason a regression test earns its place.

* **A1** — ``ACT`` published the curiosity event through ``Event``/``Layer``
  without importing either. Every *successful* curiosity call therefore raised
  ``NameError`` and aborted the rest of the phase (environment step, benchmark,
  evolution, skill synthesis). It only bit on success, which is why nothing
  caught it: the failure path was the covered one.
* **A2** — ``acquire()`` charged the rate limit at *reservation* time and never
  put it back when the lease was handed in unused. A single ethics veto on
  ``train_weights`` silenced it for 1000 ticks it had never run in.
  ``min_interval`` counts executions, not intentions.
* **A7** — a planned action owned by one of ACT's scheduled blocks kept its
  reservation, and ``finalize_tick`` committed it at the full estimate. An
  entirely offline run therefore "spent" 12 000 tokens over 2000 ticks, which
  exhausted the token budget, produced 3816 denied leases, and cost the planner
  a third of its measured advantage in the acceptance harness.
"""
import asyncio

from aegis.config import LLM_THINK_EVERY_N_TICKS
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


# ── A1: the curiosity path must survive its own success ──────────────

def test_successful_curiosity_does_not_abort_the_act_phase():
    async def run():
        s = _make_substrate()
        s.llm.enabled = True

        async def _curious(known, lease=None):
            return {"success": True,
                    "parsed": {"topic": "phase transitions",
                               "question": "why do they sharpen?",
                               "connection": "entropy"}}

        s.llm.generate_curiosity = _curious
        # The curiosity block needs an LLM tick that is also a multiple of
        # LLM_THINK_EVERY_N_TICKS * 5.
        cadence = LLM_THINK_EVERY_N_TICKS * 5
        s.tick_count = cadence - 1

        published = []
        original = s.event_bus.publish

        async def _capture(event):
            published.append(event)
            return await original(event)

        s.event_bus.publish = _capture
        await s.tick()

        # The concept landed AND the event went out — before the fix the store
        # happened and the publish raised, so this second half was never true.
        assert "phase transitions" in s.memory.semantic
        assert any(e.event_type == "llm_curiosity" for e in published)
    asyncio.run(run())


def test_curiosity_success_still_reaches_the_end_of_act():
    """The whole point of A1: work scheduled after the publish still happens."""
    async def run():
        s = _make_substrate()
        s.llm.enabled = True

        async def _curious(known, lease=None):
            return {"success": True, "parsed": {"topic": "renormalisation",
                                                "question": "q", "connection": "c"}}

        s.llm.generate_curiosity = _curious
        stepped = {"n": 0}

        def _step():
            stepped["n"] += 1
            return {"reward": 0.0, "solved": False, "task": None}

        s.environment.step = _step
        # A tick that is both a curiosity tick and an environment-step tick.
        from aegis.config import ENV_STEP_EVERY_N_TICKS
        cadence = LLM_THINK_EVERY_N_TICKS * 5 * max(1, ENV_STEP_EVERY_N_TICKS)
        s.tick_count = cadence - 1
        await s.tick()
        assert stepped["n"] == 1
    asyncio.run(run())


# ── A2: a released lease releases the rate limit too ─────────────────

def test_released_lease_restores_a_first_time_rate_limit():
    s = _make_substrate()
    s.tick_count = 500
    assert "train_weights" not in s.actions.last_run

    lease = s.resources.reserve(
        s.actions.by_name["train_weights"].reservation_cost(),
        "train_weights", 0.5, safety_critical=False)
    assert lease is not None
    s._rate_limit_undo[lease.id] = s.actions.last_run.get("train_weights")
    s.actions.mark_run("train_weights", s.tick_count)

    s.release(lease)
    # Never ran, so nothing to wait for.
    assert "train_weights" not in s.actions.last_run
    assert not s.actions.rate_limited(s.actions.by_name["train_weights"], 501)


def test_released_lease_restores_the_previous_run_tick():
    s = _make_substrate()
    spec = s.actions.by_name["consolidate_memory"]
    s.actions.mark_run(spec, 100)

    s.tick_count = 400
    lease = s.acquire("consolidate_memory")
    assert lease is not None
    assert s.actions.last_run["consolidate_memory"] == 400

    s.release(lease)
    assert s.actions.last_run["consolidate_memory"] == 100


def test_settled_lease_keeps_the_rate_limit():
    """The action ran, so the interval stands — the undo entry is spent."""
    s = _make_substrate()
    s.tick_count = 400
    lease = s.acquire("consolidate_memory")
    assert lease is not None
    s.settle(lease, value=1.0)
    assert s.actions.last_run["consolidate_memory"] == 400
    assert lease.id not in s._rate_limit_undo


def test_an_ethics_veto_does_not_cost_the_action_its_interval():
    """The integration this defect actually harmed.

    ``_choose`` reserves, ethics refuses, the lease goes back. Before the fix
    the refused action was also rate-limited for ``min_interval`` ticks despite
    never having run.
    """
    async def run():
        from aegis.layers.phases import decide as decide_phase

        s = _make_substrate()
        s.llm.enabled = False
        s.tick_count = 700

        s.ethics.evaluate_action = lambda info: {
            "status": "blocked", "score": 0.0, "violations": ["test"]}

        ctx = s._ctx
        ctx.tick = s.tick_count
        await decide_phase.run(s, ctx)

        # Every plan the gate refused is still offerable next tick.
        assert not any(s.actions.rate_limited(spec, s.tick_count + 1)
                       for spec in s.actions.specs)
    asyncio.run(run())


# ── A7: a reservation the scheduled block does not use goes back ─────

def test_a_scheduled_action_hands_its_planning_lease_back(isolated_state):
    """The planner may pick an action a scheduled block owns. That block takes
    its own lease, so the planner's must be returned — anything still open when
    the tick ends is committed at its full estimate, and `curiosity_explore`
    estimates 900 tokens nobody asked a model to spend."""
    async def run():
        from aegis.layers.phases import act as act_phase

        s = _make_substrate()
        s.llm.enabled = False
        s._ctx.action = "curiosity_explore"
        s._ctx.lease = s.acquire("curiosity_explore")
        assert s._ctx.lease is not None
        before = s.resources.spent("llm_tokens")
        assert before > 0                      # reserved

        await act_phase._run_planned_action(s, s._ctx)

        assert s._ctx.lease is None
        assert s.resources.spent("llm_tokens") == 0
        # The interval stands: the action was selected, and the block that owns
        # it runs it on its own cadence.
        assert s.actions.last_run["curiosity_explore"] == s.tick_count
    asyncio.run(run())


def test_no_phantom_token_spend_over_a_run(isolated_state):
    """The whole point: an offline system that never calls a model must finish
    a run having spent nothing."""
    async def run():
        s = _make_substrate()
        s.llm.enabled = False
        for _ in range(25):
            await s.tick()
        assert s.resources.spent("llm_tokens") == 0
        assert s.resources.spent("llm_calls") == 0
    asyncio.run(run())


def test_undo_map_does_not_grow_across_ticks():
    async def run():
        s = _make_substrate()
        s.llm.enabled = False
        for _ in range(3):
            await s.tick()
        assert s._rate_limit_undo == {}
    asyncio.run(run())
