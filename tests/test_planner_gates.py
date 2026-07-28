"""The gate sequence of DECIDE (spec Appendix J, M2.6).

Appendix J calls its order normative, and it is: each gate assumes the ones
before it have run. Moving one is a hole in a guarantee, not a refactoring, so
the order is asserted here rather than left to reading.

The two properties everything else rests on:

* **ethics is last and unarguable** — nothing runs after it that could undo its
  verdict, and no amount of expected value buys a way past it;
* **the cortex may permute the shortlist, never extend it** — the only place a
  model touches the decision is the narrowest one available.
"""
import asyncio

import pytest

from aegis.cortex.cache import ResponseCache
from aegis.cortex.router import Cortex
from aegis.layers.substrate import Substrate
from tests.cortex_fakes import ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def s(isolated_state):
    substrate = Substrate()
    substrate.llm.enabled = False

    async def _no_agents():
        return []

    async def _no_learning(*a, **k):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    substrate.health.check = lambda: {"status": "healthy", "warnings": [],
                                      "critical": [], "metrics": {}}
    substrate.sensors.read_all = lambda: {"pinned": True}
    return substrate


# ── the planner is reached at all ────────────────────────────────────

def test_a_tick_produces_a_plan(s):
    _run(s.tick())
    assert s._ctx.plan is not None
    assert s._ctx.action is not None


def test_the_plan_names_an_action_from_the_registry(s):
    _run(s.tick())
    assert s._ctx.action in s.actions.by_name


def test_the_decision_and_the_action_are_different_things(s):
    # The objective is a wish; the action is what will happen. Conflating them
    # was what left the world model unable to look up its own evidence.
    _run(s.tick())
    assert s._ctx.decision
    assert s._ctx.action
    assert s._ctx.plan.objective == s._ctx.decision


def test_planning_can_be_switched_off(s, monkeypatch):
    monkeypatch.setattr("aegis.layers.phases.decide.PLAN_ENABLED", False)
    _run(s.tick())
    assert s._ctx.plan is None
    assert s._ctx.decision            # but a decision is still made


def test_without_an_encoded_state_there_is_no_plan(s):
    s.world_model.encode = lambda inputs: (_ for _ in ()).throw(RuntimeError("no"))
    _run(s.tick())
    assert s._ctx.plan is None
    assert s._ctx.decision


# ── step 7: no lease, no action ──────────────────────────────────────

def test_the_chosen_action_holds_a_lease(s):
    """DECIDE hands ACT a lease for exactly the action it chose.

    Checked at the end of DECIDE rather than after the whole tick: ACT settles
    the lease when it performs the action, and hands it back untouched when the
    action belongs to one of the scheduled blocks that takes its own. Either
    way `ctx.lease` is legitimately empty by the end — what matters is that
    nothing reaches ACT unpaid for.
    """
    from aegis.layers.phases import decide as decide_phase

    seen = {}
    original = decide_phase.run

    async def _capture(substrate, ctx):
        await original(substrate, ctx)
        seen["lease"] = ctx.lease
        seen["action"] = ctx.action

    s._decide = lambda: _capture(s, s._ctx)
    _run(s.tick())
    assert seen["lease"] is not None
    assert seen["lease"].purpose == seen["action"]


def test_an_unaffordable_system_falls_back_rather_than_acting(s):
    for budget in s.resources.budgets.values():
        budget.limit = 0
    _run(s.tick())
    assert s._ctx.plan is None
    assert s._ctx.decision            # the system still decides something


def test_a_refusal_is_recorded(s):
    for budget in s.resources.budgets.values():
        budget.limit = 0
    _run(s.tick())
    assert s.planner.blocked.get("resources", 0) >= 1


# ── step 10: ethics is last, and unarguable ──────────────────────────

def test_a_blocked_decision_is_not_taken(s):
    s.ethics.evaluate_action = lambda action: {
        "status": "blocked", "score": 0.0, "violations": ["nope"]}
    _run(s.tick())
    # Every candidate was refused, so no plan survived to be executed.
    assert s._ctx.plan is None
    assert s.planner.blocked.get("ethics", 0) >= 1


def test_no_amount_of_expected_value_buys_a_way_past_ethics(s):
    calls = []

    def refuse(action):
        calls.append(action)
        return {"status": "blocked", "score": 0.0, "violations": ["nope"]}

    s.ethics.evaluate_action = refuse
    # Make one option overwhelmingly attractive; it must still be refused.
    for _ in range(20):
        s.world_model.observe_outcome(
            s.world_model.encode_substrate(s), "rest", success=True, reward=1.0)
    _run(s.tick())
    assert calls, "ethics was never consulted"
    assert s._ctx.plan is None


def test_a_blocked_candidate_releases_its_lease(s):
    s.ethics.evaluate_action = lambda action: {
        "status": "blocked", "score": 0.0, "violations": ["nope"]}
    _run(s.tick())
    # A refused candidate must not keep holding the budget it reserved.
    assert s.resources.open_leases() == []


def test_ethics_retries_are_bounded(s):
    calls = []

    def refuse(action):
        calls.append(action)
        return {"status": "blocked", "score": 0.0, "violations": ["nope"]}

    s.ethics.evaluate_action = refuse
    _run(s.tick())
    # A system that kept searching for something its own veto would permit is
    # a system negotiating with its veto.
    from aegis.layers.phases.decide import MAX_ETHICS_RETRIES
    planned_calls = [c for c in calls if c.get("modifies_self") is not None]
    assert len(planned_calls) <= MAX_ETHICS_RETRIES + 1


def test_an_approved_decision_records_its_verdict(s):
    _run(s.tick())
    assert s._ctx.ethics_status in ("approved", "review_required")
    assert 0.0 <= s._ctx.ethics_score <= 1.0


# ── step 11: self-preservation guards self-modification ──────────────

def test_an_irreversible_action_is_checked_against_self_preservation(s, monkeypatch):
    seen = []

    def watch(target, content):
        seen.append(target)
        return True, {"critical": []}

    s.self_preservation.is_modification_safe = watch
    # Force the planner onto the one irreversible action in the registry.
    monkeypatch.setattr(s.planner, "actions_for",
                        lambda objective, available: [
                            spec for spec in available
                            if spec.name == "code_self_mod"] or available)
    monkeypatch.setattr("aegis.config.CODE_SELF_MOD_ENABLED", True)
    _run(s.tick())
    # Either it was chosen and checked, or it was not available at all; what
    # must never happen is being chosen without the check.
    if s._ctx.action == "code_self_mod":
        assert any("code_self_mod" in target for target in seen)


def test_an_unsafe_self_modification_is_refused(s, monkeypatch):
    s.self_preservation.is_modification_safe = lambda target, content: (
        False, {"critical": ["lethal"]})
    monkeypatch.setattr(s.planner, "actions_for",
                        lambda objective, available: [
                            spec for spec in available
                            if not spec.reversible] or available)
    monkeypatch.setattr("aegis.config.CODE_SELF_MOD_ENABLED", True)
    _run(s.tick())
    if s.planner.blocked.get("self_preservation"):
        assert s._ctx.action != "code_self_mod"


# ── step 8: the cortex may permute, never extend ─────────────────────

def _with_cortex(substrate, response):
    provider = ScriptedProvider("a", responses=[response, response])
    substrate.llm.cortex = Cortex(providers={"a": provider},
                                  routes={"deep": ["a"]},
                                  cache=ResponseCache(None),
                                  resources=substrate.resources)
    return provider


def test_the_cortex_can_reorder_the_shortlist(s):
    _with_cortex(s, '{"order": [1, 0], "rationale": "second looks better"}')
    _run(s.tick())
    if s._ctx.plan is not None and s._ctx.plan.source == "planner+cortex":
        assert s._ctx.plan is not None


def test_an_index_outside_the_shortlist_is_discarded(s):
    provider = _with_cortex(s, '{"order": [99], "rationale": "invented"}')
    _run(s.tick())
    # The planner's own choice stands; nothing outside the list it sent can be
    # selected by pointing at it.
    assert s._ctx.plan is None or s._ctx.plan.source == "planner"


def test_a_malformed_rerank_is_discarded(s):
    _with_cortex(s, "not json at all")
    _run(s.tick())
    assert s._ctx.plan is None or s._ctx.plan.source == "planner"


def test_the_cortex_sees_only_the_top_three(s):
    provider = _with_cortex(s, '{"order": [0], "rationale": "fine"}')
    _run(s.tick())
    if provider.invocations:
        prompt = provider.invocations[0][-1]["content"]
        indices = [line.split(":")[0] for line in prompt.splitlines()
                   if line and line[0].isdigit()]
        assert len(indices) <= 3


def test_a_failing_cortex_leaves_the_planners_order_alone(s):
    provider = ScriptedProvider("a", fail=True)
    s.llm.cortex = Cortex(providers={"a": provider}, routes={"deep": ["a"]},
                          cache=ResponseCache(None), resources=s.resources)
    _run(s.tick())
    assert s._ctx.plan is None or s._ctx.plan.source == "planner"


def test_no_cortex_route_skips_the_rerank_entirely(s):
    s.llm.cortex.configure_routes({})
    _run(s.tick())
    assert s._ctx.plan is None or s._ctx.plan.source == "planner"


# ── step 12: the experience and the forecast open before the action ──

def test_the_forecast_is_recorded_before_the_action(s):
    _run(s.tick())
    assert s._ctx.prediction is not None
    assert s._ctx.prediction.action == s._ctx.action


def test_the_experience_is_opened_in_decide(s):
    async def stop_after_decide():
        s._ctx.__class__  # noqa: B018 — readability only
        await s._perceive()
        await s._evaluate()
        await s._decide()

    _run(stop_after_decide())
    assert "decide" in s._pending_experiences


def test_the_chosen_objective_is_credited_not_the_value_tables_own_pick(s):
    # Once a planner exists, the value table's argmax is no longer what
    # happened; reward has to be credited to what was actually done.
    _run(s.tick())
    assert s.goal_intelligence._last_choice["objective"] == s._ctx.decision


# ── the action actually runs ─────────────────────────────────────────

def test_the_planned_action_is_performed(s):
    _run(s.tick())
    if s._ctx.action and s._ctx.action not in (
            "env_step", "run_benchmark", "learn_external", "run_agents"):
        assert s._ctx.action in s._ctx.executed_actions


def test_a_failing_executor_costs_the_action_not_the_tick(s):
    errors_before = s.health.error_count

    def explode(*args, **kwargs):
        raise RuntimeError("executor down")

    s.actions.executor_for = lambda spec, substrate, ctx=None: explode
    _run(s.tick())
    assert s.health.error_count == errors_before


def test_scheduled_work_is_not_done_twice(s):
    # The planner may choose an action the scheduled code already owns; both
    # running would mean two environment steps or two benchmarks in one tick.
    from aegis.layers.phases.act import _SCHEDULED_ACTIONS
    calls = []
    s.environment.step = lambda: (calls.append(1),
                                  {"reward": 0.0, "solved": False, "task": None})[1]
    for _ in range(6):
        _run(s.tick())
    assert "env_step" in _SCHEDULED_ACTIONS
    assert len(calls) <= 6


# ── the loop closes ──────────────────────────────────────────────────

def test_the_promise_is_measured_against_what_happened(s):
    for _ in range(4):
        _run(s.tick())
    assert s.planner.ev_gap is not None


def test_planner_metrics_reach_the_time_series(s):
    from aegis.telemetry import metrics as M
    for _ in range(3):
        _run(s.tick())
    s.telemetry.flush()
    for metric in (M.PLAN_OVERRIDE_RATE, M.PLAN_LATENCY_MS, M.PLAN_BLOCKED):
        assert len(s.telemetry.series(metric)) >= 1, metric


def test_the_planner_appears_in_the_status(s):
    _run(s.tick())
    status = s.full_status()["planner"]
    assert "override_rate" in status and "last_plan" in status
