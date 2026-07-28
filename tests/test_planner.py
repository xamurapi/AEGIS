"""The planner: decisions by comparing plans (spec M2).

What separates this from the ranking it replaced is that the score consults
what the system has *learned*, not only what it wants. So the tests that matter
are the ones showing evidence changing the choice, and the ones showing the
planner staying inside its remit — it proposes, and every gate that can refuse
runs afterwards.
"""
import pytest

import aegis.config as cfg
from aegis.layers.actions import ActionSpec, ActionSpace
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.planner import COST_SCALE, DEFAULT_WEIGHTS, Plan, Planner
from aegis.layers.world.state import StateKey
from aegis.layers.world_model import PredictiveWorldModel


class _Ctx:
    """The slice of a tick context the planner reads."""

    def __init__(self, state):
        self.state = state
        self.state_inputs = {}


class _Values:
    def __init__(self, values=None, drives=None):
        self.values = values or {}
        self.drives = drives or {}

    def expected_value(self, objective, context=None):
        return self.values.get(objective, 0.0)

    def _classify_drive(self, objective):
        return self.drives.get(objective, "knowledge")


def spec(name, drive="knowledge", tok=0, ms=10, safety=False):
    return ActionSpec(name=name, drive=drive,
                      cost=ResourceCost(llm_tokens=tok, wall_ms=ms),
                      executor="world.perceive", safety_critical=safety)


@pytest.fixture
def world(tmp_path):
    return PredictiveWorldModel(store_path=tmp_path / "model.json")


@pytest.fixture
def ctx():
    return _Ctx(StateKey(energy="hi"))


def make_planner(world, values=None, drives=None, policy=None):
    return Planner(world_model=world, actions=ActionSpace(),
                   goal_intelligence=_Values(values, drives), policy=policy)


def teach(world, state, action, reward, success=True, times=10, cost=0.0):
    for _ in range(times):
        world.observe_transition(state, action, state)
        world.observe_outcome(state, action, success=success, reward=reward,
                              cost=cost)


# ── the weights are genes ────────────────────────────────────────────

def test_default_weights_match_the_appendix():
    assert DEFAULT_WEIGHTS == {"ev": 1.0, "val": 0.6, "exp": 0.3,
                               "cost": 0.4, "risk": 0.5}


def test_the_genome_can_retune_the_scoring(world):
    planner = make_planner(world)
    planner.set_weights({"w_ev": 2.0, "w_risk": 0.1, "plan_depth": 2,
                         "plan_beam": 7, "plan_discount": 0.5})
    assert planner.weights["ev"] == 2.0
    assert planner.weights["risk"] == 0.1
    assert planner.depth == 2 and planner.beam == 7
    assert planner.discount == 0.5


def test_an_unusable_gene_is_ignored(world):
    planner = make_planner(world)
    planner.set_weights({"w_ev": "lots", "plan_depth": "deep"})
    assert planner.weights["ev"] == 1.0
    assert planner.depth == cfg.WM_DEPTH


def test_an_unrelated_gene_is_ignored(world):
    planner = make_planner(world)
    planner.set_weights({"something_else": 3})
    assert planner.weights == DEFAULT_WEIGHTS


# ── the score ────────────────────────────────────────────────────────

def test_the_score_is_the_weighted_sum_the_spec_declares(world, ctx):
    planner = make_planner(world, values={"goal": 0.5})
    plan = Plan(objective="goal", steps=["act"], expected_value=0.8, risk=0.2,
                expected_cost=ResourceCost(llm_tokens=int(COST_SCALE * 1000 / 2)))
    score = planner.score(plan, ctx)

    expected = (0.8 * 1.0          # expected value
                + 0.5 * 0.6        # learned value
                + 1.0 * 0.3        # nothing known yet, so all to learn
                - 0.5 * 0.4        # cost, half the scale
                - 0.2 * 0.5)       # risk
    assert score == pytest.approx(expected)


def test_the_score_records_how_it_was_reached(world, ctx):
    planner = make_planner(world)
    plan = Plan(objective="goal", steps=["act"], expected_value=0.5)
    planner.score(plan, ctx)
    assert set(plan.breakdown) >= {"expected_value", "value", "explore",
                                   "cost_norm", "risk", "policy_delta"}


def test_cost_counts_against_the_score(world, ctx):
    planner = make_planner(world)
    cheap = Plan(objective="a", steps=["act"], expected_value=0.5)
    dear = Plan(objective="a", steps=["act"], expected_value=0.5,
                expected_cost=ResourceCost(llm_tokens=100_000))
    assert planner.score(dear, ctx) < planner.score(cheap, ctx)


def test_risk_counts_against_the_score(world, ctx):
    planner = make_planner(world)
    safe = Plan(objective="a", steps=["act"], expected_value=0.5, risk=0.0)
    risky = Plan(objective="a", steps=["act"], expected_value=0.5, risk=1.0)
    assert planner.score(risky, ctx) < planner.score(safe, ctx)


def test_the_exploration_bonus_points_at_the_unknown(world, ctx):
    planner = make_planner(world)
    teach(world, ctx.state, "known", reward=0.5)
    assert planner.explore_bonus(ctx, "untried") == 1.0
    assert planner.explore_bonus(ctx, "known") == 0.0


def test_no_action_means_no_exploration_bonus(world, ctx):
    assert make_planner(world).explore_bonus(ctx, None) == 0.0


def test_a_broken_value_source_does_not_break_scoring(world, ctx):
    class Exploding:
        def expected_value(self, objective, context=None):
            raise RuntimeError("value store down")

        def _classify_drive(self, objective):
            return "knowledge"

    planner = Planner(world_model=world, actions=ActionSpace(),
                      goal_intelligence=Exploding())
    assert isinstance(planner.score(Plan(objective="a", steps=["x"]), ctx), float)


# ── the policy contributes nothing until it exists ───────────────────

def test_without_a_policy_there_is_no_delta(world, ctx):
    assert make_planner(world).policy_delta(ctx, "act") == 0.0


def test_a_policy_delta_moves_the_score(world, ctx):
    class Policy:
        def delta(self, state, action):
            return 0.5 if action == "favoured" else -0.5

    planner = make_planner(world, policy=Policy())
    favoured = Plan(objective="a", steps=["favoured"], expected_value=0.5)
    other = Plan(objective="a", steps=["other"], expected_value=0.5)
    assert planner.score(favoured, ctx) > planner.score(other, ctx)


def test_a_broken_policy_does_not_break_scoring(world, ctx):
    class Policy:
        def delta(self, state, action):
            raise RuntimeError("rules down")

    assert make_planner(world, policy=Policy()).policy_delta(ctx, "act") == 0.0


# ── which actions serve which objective ──────────────────────────────

def test_actions_are_matched_to_the_objectives_drive(world):
    planner = make_planner(world, drives={"learn": "knowledge"})
    available = [spec("study", "knowledge"), spec("practise", "competence")]
    assert [s.name for s in planner.actions_for("learn", available)] == ["study"]


def test_safety_critical_work_is_not_relevant_to_every_objective(world):
    # It is protected, not universally on-topic; folding it into every set
    # would let a checkpoint win a tie for an objective about reasoning.
    planner = make_planner(world, drives={"learn": "knowledge"})
    available = [spec("study", "knowledge"), spec("checkpoint", "stability",
                                                  safety=True)]
    assert [s.name for s in planner.actions_for("learn", available)] == ["study"]


def test_an_objective_no_action_serves_still_gets_a_plan(world):
    # Otherwise it would silently drop out of consideration.
    planner = make_planner(world, drives={"odd": "coherence"})
    available = [spec("study", "knowledge")]
    assert planner.actions_for("odd", available) == available


def test_an_unclassifiable_objective_defaults_to_knowledge(world):
    planner = Planner(world_model=world, actions=ActionSpace())
    assert planner.drive_of("whatever") == "knowledge"


# ── building a plan ──────────────────────────────────────────────────

def test_a_plan_names_the_action_it_would_take(world, ctx):
    planner = make_planner(world)
    teach(world, ctx.state, "study", reward=0.9)
    plan = planner.plan_for("learn", ctx, [spec("study", "knowledge")])
    assert plan is not None
    assert plan.action == "study"
    assert plan.steps[0] == "study"


def test_a_plan_carries_the_cost_of_its_steps(world, ctx):
    planner = make_planner(world)
    teach(world, ctx.state, "study", reward=0.9)
    plan = planner.plan_for("learn", ctx, [spec("study", "knowledge", tok=100)])
    assert plan.expected_cost.llm_tokens >= 100


def test_no_available_actions_means_no_plan(world, ctx):
    assert make_planner(world).plan_for("learn", ctx, []) is None


def test_a_plan_reports_its_confidence(world, ctx):
    planner = make_planner(world)
    unknown = planner.plan_for("learn", ctx, [spec("untried", "knowledge")])
    teach(world, ctx.state, "known", reward=0.5)
    known = planner.plan_for("learn", ctx, [spec("known", "knowledge")])
    assert known.confidence > unknown.confidence


def test_known_failures_raise_a_plans_risk(world, ctx):
    planner = make_planner(world)
    for _ in range(5):
        world.observe("dangerous_thing", "broke", success=False)
    assert planner.risk_of("dangerous_thing", ctx, "act") > 0


def test_reward_spread_raises_a_plans_risk(world, ctx):
    planner = make_planner(world)
    for index in range(10):
        world.observe_outcome(ctx.state, "swingy", success=(index % 2 == 0),
                              reward=float(index % 2))
        world.observe_outcome(ctx.state, "steady", success=True, reward=0.5)
    assert planner.risk_of("x", ctx, "swingy") > planner.risk_of("x", ctx, "steady")


def test_risk_is_bounded(world, ctx):
    planner = make_planner(world)
    for _ in range(50):
        world.observe("terrible", "broke", success=False)
    assert planner.risk_of("terrible", ctx, "act") <= 1.0


# ── evidence changes the choice ──────────────────────────────────────

def test_the_better_paying_action_wins_a_tie_on_wishes(world, ctx):
    planner = make_planner(world, values={"a": 0.5, "b": 0.5},
                           drives={"a": "knowledge", "b": "knowledge"})
    teach(world, ctx.state, "rich", reward=0.9)
    teach(world, ctx.state, "poor", reward=0.05)

    rich = planner.plan_for("a", ctx, [spec("rich", "knowledge")])
    poor = planner.plan_for("b", ctx, [spec("poor", "knowledge")])
    assert planner.score(rich, ctx) > planner.score(poor, ctx)


def test_learned_value_can_outweigh_a_thin_expected_value(world, ctx):
    planner = make_planner(world, values={"valued": 1.0, "ignored": 0.0},
                           drives={"valued": "knowledge", "ignored": "knowledge"})
    teach(world, ctx.state, "act", reward=0.5)
    valued = planner.plan_for("valued", ctx, [spec("act", "knowledge")])
    ignored = planner.plan_for("ignored", ctx, [spec("act", "knowledge")])
    assert planner.score(valued, ctx) > planner.score(ignored, ctx)


# ── explaining ───────────────────────────────────────────────────────

def test_a_plan_explains_itself_from_its_own_numbers(world, ctx):
    planner = make_planner(world, values={"goal": 0.5})
    plan = Plan(objective="goal", steps=["act"], expected_value=0.8)
    planner.score(plan, ctx)
    assert "goal" in plan.rationale and "act" in plan.rationale
    assert "score" in plan.rationale


def test_a_thinly_evidenced_plan_says_so(world, ctx):
    planner = make_planner(world)
    plan = Plan(objective="goal", steps=["act"], expected_value=0.5, confidence=0.1)
    planner.score(plan, ctx)
    assert "thin evidence" in plan.rationale


def test_a_well_evidenced_plan_does_not(world, ctx):
    planner = make_planner(world)
    plan = Plan(objective="goal", steps=["act"], expected_value=0.5, confidence=0.9)
    planner.score(plan, ctx)
    assert "thin evidence" not in plan.rationale


def test_an_unscored_plan_still_explains_itself(world):
    assert "goal" in make_planner(world).explain(Plan(objective="goal", steps=["act"]))


def test_a_plan_with_no_steps_explains_itself(world):
    assert make_planner(world).explain(Plan(objective="goal"))


# ── pricing someone else's proposal ──────────────────────────────────

def test_a_proposed_sequence_is_priced_by_the_same_yardstick(world, ctx):
    planner = make_planner(world)
    teach(world, ctx.state, "good", reward=0.9)
    teach(world, ctx.state, "bad", reward=0.05)
    assert planner.evaluate(ctx, ["good"]) > planner.evaluate(ctx, ["bad"])


def test_pricing_survives_a_broken_model(world, ctx):
    planner = make_planner(world)

    def explode(state, sequence):
        raise RuntimeError("model down")

    planner.world_model.evaluate_sequence = explode
    assert planner.evaluate(ctx, ["a"]) == 0.0


# ── measuring whether it helped ──────────────────────────────────────

def test_agreeing_with_the_greedy_pick_is_not_an_override(world):
    planner = make_planner(world)
    planner.record_choice(Plan(objective="same"), "same")
    assert planner.override_rate() == 0.0


def test_choosing_differently_is_an_override(world):
    planner = make_planner(world)
    planner.record_choice(Plan(objective="planned"), "greedy")
    assert planner.override_rate() == 1.0


def test_the_override_rate_is_a_fraction(world):
    planner = make_planner(world)
    planner.record_choice(Plan(objective="a"), "a")
    planner.record_choice(Plan(objective="b"), "a")
    assert planner.override_rate() == 0.5


def test_no_decisions_yet_means_no_rate(world):
    assert make_planner(world).override_rate() == 0.0


def test_the_promise_is_closed_against_what_happened(world):
    planner = make_planner(world)
    planner.record_outcome(Plan(objective="a", expected_value=1.0), realised=0.4)
    assert planner.ev_gap == pytest.approx(0.6)


def test_repeated_outcomes_smooth_the_gap(world):
    planner = make_planner(world)
    for _ in range(50):
        planner.record_outcome(Plan(objective="a", expected_value=1.0), 1.0)
    first = planner.ev_gap
    for _ in range(50):
        planner.record_outcome(Plan(objective="a", expected_value=1.0), 0.0)
    assert planner.ev_gap > first


def test_blocking_reasons_are_counted(world):
    planner = make_planner(world)
    planner.note_blocked("ethics")
    planner.note_blocked("ethics")
    planner.note_blocked("resources")
    assert planner.status()["blocked"] == {"ethics": 2, "resources": 1}


def test_status_reports_the_settings_in_force(world):
    status = make_planner(world).status()
    assert set(status) >= {"weights", "depth", "beam", "discount",
                           "plans_built", "override_rate", "blocked"}
