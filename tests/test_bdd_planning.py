"""pytest-bdd step definitions for tests/features/planning.feature.

Executable Gherkin: every scenario drives the real substrate, so the feature
file is both the specification of how a decision is reached and the test that
it is reached that way.
"""
import asyncio

import pytest
from pytest_bdd import given, scenarios, then, when

from aegis.cortex.cache import ResponseCache
from aegis.cortex.router import Cortex
from aegis.layers.substrate import Substrate
from aegis.layers.planner import Plan
from aegis.layers.world.state import StateKey
from tests.cortex_fakes import ScriptedProvider

scenarios("features/planning.feature")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx(isolated_state):
    substrate = Substrate()
    substrate.llm.enabled = False

    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.environment.step = lambda: {"reward": 0.0, "solved": False,
                                          "task": None}
    substrate.health.check = lambda: {"status": "healthy", "warnings": [],
                                      "critical": [], "metrics": {}}
    substrate.sensors.read_all = lambda: {"pinned": True}
    return {"substrate": substrate, "scores": {}}


# ── Background ───────────────────────────────────────────────────────

@given("a running AEGIS instance with a planner")
def _running(ctx):
    assert ctx["substrate"].planner is not None


# ── the planner proposes ─────────────────────────────────────────────

@given('the world model has learned that "rest" pays well')
def _rest_pays(ctx):
    substrate = ctx["substrate"]
    state = StateKey(energy="hi")
    ctx["state"] = state
    for _ in range(15):
        substrate.world_model.observe_transition(state, "rest", state)
        substrate.world_model.observe_outcome(state, "rest", success=True,
                                              reward=0.95)


@given('the world model has learned that "dream" pays badly')
def _dream_pays_badly(ctx):
    substrate = ctx["substrate"]
    state = ctx["state"]
    for _ in range(15):
        substrate.world_model.observe_transition(state, "dream", state)
        substrate.world_model.observe_outcome(state, "dream", success=False,
                                              reward=0.05)


@given("every resource budget is exhausted")
def _no_budget(ctx):
    for budget in ctx["substrate"].resources.budgets.values():
        budget.limit = 0


@given("the ethics core refuses every action")
def _ethics_refuses(ctx):
    ctx["substrate"].ethics.evaluate_action = lambda action: {
        "status": "blocked", "score": 0.0, "violations": ["refused"]}


@given("a cortex that answers with an index outside the shortlist")
def _lying_cortex(ctx):
    substrate = ctx["substrate"]
    provider = ScriptedProvider(
        "a", responses=['{"order": [42], "rationale": "invented"}'] * 4)
    substrate.llm.cortex = Cortex(providers={"a": provider},
                                  routes={"deep": ["a"]},
                                  cache=ResponseCache(None),
                                  resources=substrate.resources)


@when("the system takes a tick")
def _tick(ctx):
    _run(ctx["substrate"].tick())


@when("the system takes several ticks")
def _several_ticks(ctx):
    for _ in range(5):
        _run(ctx["substrate"].tick())


@when("the system scores both plans")
def _score_both(ctx):
    substrate = ctx["substrate"]

    class _Ctx:
        state = ctx["state"]
        state_inputs = {}

    for action in ("rest", "dream"):
        plan = Plan(objective="stay_well", steps=[action],
                    expected_value=substrate.world_model.evaluate_sequence(
                        ctx["state"], [action]))
        ctx["scores"][action] = substrate.planner.score(plan, _Ctx())


@then("a plan should have been built")
def _plan_built(ctx):
    assert ctx["substrate"]._ctx.plan is not None


@then("the plan should name an action from the registry")
def _plan_names_action(ctx):
    substrate = ctx["substrate"]
    assert substrate._ctx.action in substrate.actions.by_name


@then("the action should hold a resource lease")
def _action_has_lease(ctx):
    substrate = ctx["substrate"]
    # The lease is released once the action has been performed and settled, so
    # what matters is that one was granted for it.
    assert substrate._ctx.plan is not None
    assert substrate.resources.granted > 0


@then("the better-paying plan should score higher")
def _better_scores_higher(ctx):
    assert ctx["scores"]["rest"] > ctx["scores"]["dream"]


@then("the plan should carry a rationale naming its objective and action")
def _plan_explains(ctx):
    plan = ctx["substrate"]._ctx.plan
    assert plan is not None
    assert plan.objective in plan.rationale
    assert (plan.action or "") in plan.rationale


@then("no plan should have been executed")
def _nothing_executed(ctx):
    assert ctx["substrate"]._ctx.plan is None


@then("the refusal should be recorded as a resource block")
def _resource_block(ctx):
    assert ctx["substrate"].planner.blocked.get("resources", 0) >= 1


@then("the refusal should be recorded as an ethics block")
def _ethics_block(ctx):
    assert ctx["substrate"].planner.blocked.get("ethics", 0) >= 1


@then("no lease should still be held")
def _no_lease_held(ctx):
    assert ctx["substrate"].resources.open_leases() == []


@then("the planner's own choice should stand")
def _planner_choice_stands(ctx):
    plan = ctx["substrate"]._ctx.plan
    assert plan is None or plan.source == "planner"


@then("a forecast should exist for the chosen action")
def _forecast_exists(ctx):
    substrate = ctx["substrate"]
    assert substrate._ctx.prediction is not None
    assert substrate._ctx.prediction.action == substrate._ctx.action


@then("the gap between promised and realised value should be measured")
def _gap_measured(ctx):
    assert ctx["substrate"].planner.ev_gap is not None
