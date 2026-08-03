"""Mutation-killing tests for the planner (spec §3.7).

The harness mutates one operator at a time and asks whether any test notices.
Eighteen mutants survived the first run, and they were all the same kind of
gap: the suite checked that a number came out and that it ordered plans
correctly, but never that the number was *this* number. A planner whose
`/10` became `*10` still ranked things — just wrongly, and by a factor of a
hundred.

So these tests pin arithmetic rather than behaviour. Each one names the
constant it defends, because a test that asserts `0.1` without saying why is
the reason people delete failing assertions.
"""
import pytest

from aegis.layers.actions import ActionSpec, ActionSpace
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.planner import COST_SCALE, Plan, Planner
from aegis.layers.world.state import StateKey


class _Ctx:
    def __init__(self, state=None, inputs=None):
        self.state = state or StateKey(energy="hi")
        self.state_inputs = inputs if inputs is not None else {}


class _Values:
    def __init__(self, values=None, drives=None):
        self.values = values or {}
        self.drives = drives or {}

    def expected_value(self, objective, context=None):
        return self.values.get(objective, 0.0)

    def _classify_drive(self, objective):
        return self.drives.get(objective, "knowledge")


class _Model:
    """A world model with dials, so a score can be predicted exactly."""

    def __init__(self, *, sequence=("rest",), value=1.0, failures=(),
                 reward_sd=0.0, p_pess=0.5, knows=0.5):
        self.sequence = list(sequence)
        self.value = value
        self.failures = list(failures)
        self.reward_sd = reward_sd
        self.p_pess = p_pess
        self._knows = knows

    def rollout(self, state, actions, depth, beam):
        return type("R", (), {"sequence": list(self.sequence),
                              "value": self.value})()

    def risks_for(self, tokens):
        return [{"failure_rate": rate} for rate in self.failures]

    def predict_outcome(self, state, action):
        return type("O", (), {"reward_sd": self.reward_sd,
                              "p_success_pessimistic": self.p_pess})()

    def knows(self, state, action):
        return self._knows

    def evaluate_sequence(self, state, sequence):
        return self.value


def _spec(name, drive="knowledge", ms=0, safety=False):
    return ActionSpec(name=name, drive=drive, cost=ResourceCost(wall_ms=ms),
                      executor="world.perceive", safety_critical=safety)


def _planner(model, **kw):
    kw.setdefault("goal_intelligence", _Values())
    return Planner(world_model=model, actions=ActionSpace(), **kw)


# ── the Plan record's own defaults ───────────────────────────────────

def test_a_plan_is_not_safety_critical_unless_it_says_so():
    """The default has to be the *safe* one: an ordinary plan that claimed to
    be safety-critical would be immune to policy suppression and could spend
    the reserved floor."""
    plan = Plan(objective="obj")
    assert plan.safety_critical is False
    assert plan.as_dict()["safety_critical"] is False


def test_a_plan_inherits_criticality_from_the_action_it_starts_with():
    model = _Model(sequence=("rest",))
    planner = _planner(model)
    ordinary = planner.plan_for("obj", _Ctx(), [_spec("rest", safety=False)])
    assert ordinary.safety_critical is False

    critical = planner.plan_for("obj", _Ctx(), [_spec("rest", safety=True)])
    assert critical.safety_critical is True


# ── objective collection: which goals count ──────────────────────────

def test_only_active_non_axiom_goals_become_objectives():
    planner = _planner(_Model())

    class _Goal:
        def __init__(self, name, status, level):
            self.name, self.status, self.level = name, status, level

    class _Substrate:
        goals = type("G", (), {
            "goals": [_Goal("live_one", "active", "strategy"),
                      _Goal("done_one", "completed", "strategy"),
                      _Goal("axiom_one", "active", "axiom")],
            "get_current_focus": staticmethod(lambda: None)})()
        cognitive_graph = type("C", (), {"related": staticmethod(lambda n: [])})()
        meta_goals = type("M", (), {"active_meta_goals": [], "generated_goals": []})()

    found = planner.collect_objectives(_Substrate())
    assert "live_one" in found
    assert "done_one" not in found          # completed goals are not pursued
    assert "axiom_one" not in found         # axioms are constraints, not tasks


def test_planner_reads_meta_goals_from_a_real_generator():
    """collect_objectives read ``meta_goals.goals`` — an attribute the real
    MetaGoalGenerator never had (it exposes ``active_meta_goals`` and
    ``generated_goals``). The AttributeError was swallowed by the defensive
    try, so meta-goals never reached the shortlist. The fakes above cannot
    catch that class of defect; only the real class can."""
    from aegis.layers.meta_goal_generator import MetaGoalGenerator

    planner = _planner(_Model())
    meta = MetaGoalGenerator()
    # learning_sessions < 5 triggers the knowledge_expansion domain.
    generated = meta.generate_goals({"learning_sessions": 0})
    assert generated, "the trigger context must actually produce a meta-goal"

    class _Substrate:
        goals = type("G", (), {"goals": [],
                               "get_current_focus": staticmethod(lambda: None)})()
        cognitive_graph = type("C", (), {"related": staticmethod(lambda n: [])})()
        meta_goals = meta

    found = planner.collect_objectives(_Substrate())
    assert any(g["description"][:60] in found for g in meta.active_meta_goals)


# ── risk: two sources, each with a fixed scale ───────────────────────

def test_known_failures_contribute_a_tenth_of_their_summed_rate():
    """`sum(rate) / 10` — the divisor is what keeps a handful of remembered
    failures from saturating risk at the cap on its own."""
    planner = _planner(_Model(failures=(1.0,), reward_sd=0.0))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.1)

    planner = _planner(_Model(failures=(1.0, 1.0, 1.0), reward_sd=0.0))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.3)


def test_the_causal_half_of_risk_is_capped_at_six_tenths():
    planner = _planner(_Model(failures=(1.0,) * 20, reward_sd=0.0))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.6)


def test_reward_spread_contributes_in_proportion_to_pessimism():
    """`sd * (1 − p_pessimistic)` — an action that varies wildly *and* is not
    reliably successful is the risky one; either alone is much less so."""
    planner = _planner(_Model(reward_sd=0.5, p_pess=0.2))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.4)

    planner = _planner(_Model(reward_sd=0.5, p_pess=0.6))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.2)

    # A perfectly reliable action carries no spread risk at all.
    planner = _planner(_Model(reward_sd=0.5, p_pess=1.0))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.0)


def test_the_spread_half_of_risk_is_capped_at_four_tenths():
    planner = _planner(_Model(reward_sd=9.0, p_pess=0.0))
    assert planner.risk_of("obj", _Ctx(), "rest") == pytest.approx(0.4)


# ── the score: every term, with its weight ───────────────────────────

def test_each_term_enters_the_score_multiplied_by_its_weight():
    """The whole scoring formula, pinned at once.

    expected_value 2.0·w_ev 1.5 + value 0.5·w_val 2.0 + explore 0.25·w_exp 0.4
    − cost_norm 0.3·w_cost 1.0 − risk 0.2·w_risk 0.5 + policy 0.05
    = 3.0 + 1.0 + 0.1 − 0.3 − 0.1 + 0.05 = 3.75
    """
    model = _Model(knows=0.75)
    planner = Planner(
        world_model=model, actions=ActionSpace(),
        goal_intelligence=_Values({"obj": 0.5}),
        policy=type("P", (), {"delta": staticmethod(lambda s, a: 0.05)})(),
        weights={"w_ev": 1.5, "w_val": 2.0, "w_exp": 0.4,
                 "w_cost": 1.0, "w_risk": 0.5})

    plan = Plan(objective="obj", steps=["rest"], expected_value=2.0,
                # wall_ms is weighted 1/10_000 by normalize_cost, so 30 s of
                # wall clock is a normalised cost of 3.0 -> 0.3 after COST_SCALE.
                expected_cost=ResourceCost(wall_ms=30_000),
                risk=0.2)
    # normalize_cost turns wall_ms back into the 0..1 magnitude used above.
    from aegis.layers.motivation.roi import normalize_cost
    assert normalize_cost(plan.expected_cost) / COST_SCALE == pytest.approx(0.3)

    assert planner.score(plan, _Ctx()) == pytest.approx(3.75)


def test_the_cost_contribution_is_capped_at_one():
    planner = _planner(_Model(knows=1.0), weights={"w_ev": 0.0, "w_val": 0.0,
                                                   "w_exp": 0.0, "w_cost": 1.0,
                                                   "w_risk": 0.0})
    plan = Plan(objective="obj", steps=["rest"],
                expected_cost=ResourceCost(wall_ms=10_000_000))
    assert planner.score(plan, _Ctx()) == pytest.approx(-1.0)


# ── policy delta: the guard is a disjunction ─────────────────────────

def test_a_plan_with_no_action_never_consults_the_policy():
    """`policy is None or action is None` — either one alone is enough to skip.

    A policy asked about ``None`` would be asked to price "doing nothing",
    which it has no evidence about and no business scoring.
    """
    asked = []
    policy = type("P", (), {"delta": staticmethod(
        lambda state, action: asked.append(action) or 0.5)})()
    planner = _planner(_Model(), policy=policy)

    assert planner.policy_delta(_Ctx(), None) == 0.0
    assert asked == []
    assert planner.policy_delta(_Ctx(), "rest") == 0.5
    assert asked == ["rest"]


def test_context_metrics_are_passed_through_not_emptied():
    """`state_inputs or {}` — the fallback replaces a missing dict, it does not
    replace a present one."""
    planner = _planner(_Model())
    assert planner._context_metrics(_Ctx(inputs={"energy": 0.5})) == {"energy": 0.5}
    assert planner._context_metrics(_Ctx(inputs=None)) == {}
    assert planner._context_metrics(object()) == {}


# ── latency: measured, in milliseconds ───────────────────────────────

def test_planning_latency_is_elapsed_seconds_in_milliseconds(monkeypatch):
    """`(end − start) · 1000`. The budget in §3.4 is 30 ms, so a factor of a
    thousand in either direction is the difference between passing and failing
    a gate nobody would then trust."""
    import aegis.layers.planner as planner_module

    ticks = iter([1.0, 1.5])

    class _Clock:
        @staticmethod
        def monotonic():
            return next(ticks)

    monkeypatch.setattr(planner_module, "CLOCK", _Clock())
    planner = _planner(_Model())

    class _Substrate:
        goals = type("G", (), {"goals": [],
                               "get_current_focus": staticmethod(lambda: None)})()
        cognitive_graph = type("C", (), {"related": staticmethod(lambda n: [])})()
        meta_goals = type("M", (), {"active_meta_goals": [], "generated_goals": []})()

    planner.build(_Substrate(), _Ctx(), [_spec("rest")])
    assert planner.last_latency_ms == pytest.approx(500.0)


# ── the explanation must not drift from the decision ─────────────────

def _explained(planner, **breakdown):
    base = {"expected_value": 0.0, "value": 0.0, "explore": 0.0,
            "cost_norm": 0.0, "risk": 0.0, "policy_delta": 0.0,
            "weights": {"ev": 1.0, "val": 1.0, "exp": 1.0,
                        "cost": 1.0, "risk": 1.0}}
    base.update(breakdown)
    plan = Plan(objective="obj", steps=["rest"], confidence=1.0)
    plan.breakdown = base
    plan.score = 1.234
    return planner.explain(plan)


def test_the_explanation_reports_each_term_weighted():
    planner = _planner(_Model())
    text = _explained(planner, expected_value=2.0,
                      weights={"ev": 1.5, "val": 2.0, "exp": 0.5,
                               "cost": 1.0, "risk": 1.0})
    assert "expected value +3.000" in text


def test_a_positive_contribution_reads_positive_and_a_cost_reads_negative():
    planner = _planner(_Model())
    # Weights deliberately away from 1.0: with unit weights a multiplication
    # and a division are indistinguishable, and the explanation would be free
    # to disagree with the score that produced it.
    text = _explained(planner, value=0.5, cost_norm=0.25, risk=0.25,
                      weights={"ev": 1.0, "val": 1.0, "exp": 1.0,
                               "cost": 2.0, "risk": 0.5})
    assert "learned value +0.500" in text
    assert "cost -0.500" in text
    assert "risk -0.125" in text


def test_room_to_learn_is_reported_with_its_own_weight():
    planner = _planner(_Model())
    text = _explained(planner, explore=0.4,
                      weights={"ev": 1.0, "val": 1.0, "exp": 0.25,
                               "cost": 1.0, "risk": 1.0})
    assert "room to learn +0.100" in text


def test_a_plan_driven_by_nothing_says_so():
    """All contributions zero — the explanation must admit it rather than
    printing an empty parenthesis that reads like a formatting bug."""
    planner = _planner(_Model())
    text = _explained(planner)
    assert "no clear driver" in text


def test_a_plan_with_real_drivers_does_not_claim_to_have_none():
    planner = _planner(_Model())
    text = _explained(planner, expected_value=1.0, value=0.5)
    assert "no clear driver" not in text
    assert "expected value +1.000" in text


def test_thin_evidence_is_stated_in_the_rationale():
    planner = _planner(_Model())
    plan = Plan(objective="obj", steps=["rest"], confidence=0.2)
    plan.breakdown = {"expected_value": 1.0, "value": 0.0, "explore": 0.0,
                      "cost_norm": 0.0, "risk": 0.0, "policy_delta": 0.0,
                      "weights": {"ev": 1.0, "val": 1.0, "exp": 1.0,
                                  "cost": 1.0, "risk": 1.0}}
    plan.score = 1.0
    assert "on thin evidence" in planner.explain(plan)

    plan.confidence = 0.9
    assert "on thin evidence" not in planner.explain(plan)
