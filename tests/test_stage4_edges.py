"""Stage-4 failure paths: what happens when a contour is unavailable.

The spec asks new modules for ≥95% branch coverage (§3.7), and for the planner
the uncovered half was entirely the *degraded* paths — the world model raising,
the cortex answering nonsense, prioritisation failing, no lease anywhere. Those
are the paths that decide whether an unavailable contour costs the system its
judgement or its ability to act at all, so they are the ones worth pinning.

Every test here asserts the same shape of promise: the failure is absorbed, a
decision still comes out, and no gate is skipped on the way.
"""
import asyncio

import pytest

import aegis.config as cfg
from aegis.layers.actions import ActionSpec, ActionSpace
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.phases import decide as decide_phase
from aegis.layers.planner import Plan, Planner
from aegis.layers.world.state import StateKey
from aegis.layers.world_model import PredictiveWorldModel


class _Ctx:
    def __init__(self, state=None):
        self.state = state
        self.state_inputs = {}
        self.decision = None
        self.action = None
        self.plan = None
        self.lease = None
        self.prediction = None
        self.confidence = 0.0
        self.ethics_status = "ok"
        self.ethics_score = 1.0

    def mark_external(self, phase):
        pass


class _Values:
    def __init__(self, values=None, drives=None):
        self.values = values or {}
        self.drives = drives or {}

    def expected_value(self, objective, context=None):
        return self.values.get(objective, 0.0)

    def _classify_drive(self, objective):
        return self.drives.get(objective, "knowledge")


class _Boom:
    """Every attribute access raises — a subsystem that is present but broken."""

    def __getattr__(self, name):
        def _raise(*a, **k):
            raise RuntimeError(f"{name} exploded")
        return _raise


def spec(name, drive="knowledge", ms=10, safety=False):
    return ActionSpec(name=name, drive=drive,
                      cost=ResourceCost(wall_ms=ms), executor="world.perceive",
                      safety_critical=safety)


@pytest.fixture
def world(tmp_path):
    return PredictiveWorldModel(store_path=tmp_path / "model.json")


@pytest.fixture
def ctx():
    return _Ctx(StateKey(energy="hi"))


def make_planner(world, **kw):
    kw.setdefault("goal_intelligence", _Values())
    return Planner(world_model=world, actions=ActionSpace(), **kw)


# ── Planner: a broken source costs its contribution, not the plan ────

def test_weights_passed_to_the_constructor_are_adopted(world):
    planner = Planner(world_model=world, actions=ActionSpace(),
                      weights={"w_ev": 1.75, "plan_beam": 9})
    assert planner.weights["ev"] == 1.75
    assert planner.beam == 9


def test_unreadable_goals_do_not_stop_objective_collection(world):
    planner = make_planner(world)

    class _Substrate:
        goals = _Boom()
        cognitive_graph = _Boom()
        meta_goals = _Boom()

    # Nothing readable anywhere — the fallback objective is still offered,
    # because a system with no goals must still decide something.
    assert planner.collect_objectives(_Substrate()) == ["idle_exploration"]


def test_graph_neighbours_and_meta_goals_become_objectives(world):
    planner = make_planner(world)

    class _Goals:
        goals = []

        @staticmethod
        def get_current_focus():
            return {"name": "expand_knowledge"}

    class _Graph:
        @staticmethod
        def related(name):
            return [{"type": "concept", "node": "entropy"},
                    {"type": "goal", "node": "ignored"},
                    {"type": "concept", "node": ""}]

    class _Meta:
        goals = [{"name": "reduce_error"}, {"description": "raise coverage"},
                 {"nothing": "usable"}]

    class _Substrate:
        goals = _Goals()
        cognitive_graph = _Graph()
        meta_goals = _Meta()

    found = planner.collect_objectives(_Substrate())
    assert "entropy" in found
    assert "expand_knowledge" in found
    assert "reduce_error" in found and "raise coverage" in found
    assert "ignored" not in found


def test_an_unclassifiable_objective_still_gets_a_drive(world):
    planner = Planner(world_model=world, actions=ActionSpace(),
                      goal_intelligence=_Boom())
    assert planner.drive_of("anything") == "knowledge"


def test_no_goal_intelligence_means_no_value_contribution(world, ctx):
    planner = Planner(world_model=world, actions=ActionSpace())
    assert planner.value_of("whatever", ctx) == 0.0
    assert planner.drive_of("whatever") == "knowledge"


def test_a_broken_value_table_contributes_nothing(world, ctx):
    planner = Planner(world_model=world, actions=ActionSpace(),
                      goal_intelligence=_Boom())
    assert planner.value_of("whatever", ctx) == 0.0


def test_a_rollout_with_no_sequence_yields_no_plan(world, ctx):
    planner = make_planner(world)
    planner.world_model = _Boom()
    # rollout raises -> plan_for cannot price anything, and says so with None
    # rather than inventing a plan out of an error.
    with pytest.raises(RuntimeError):
        planner.plan_for("obj", ctx, [spec("a")])


def test_an_empty_action_set_yields_no_plan(world, ctx):
    planner = make_planner(world)
    assert planner.plan_for("obj", ctx, []) is None


def test_a_broken_world_model_costs_risk_and_confidence_not_the_plan(world, ctx):
    planner = make_planner(world)
    planner.world_model = _Boom()
    assert planner.risk_of("obj", ctx, "a") == 0.0
    assert planner.confidence_of(ctx, "a") == 0.0
    assert planner.explore_bonus(ctx, "a") == 0.0


def test_explore_bonus_of_nothing_is_zero(world, ctx):
    planner = make_planner(world)
    assert planner.explore_bonus(ctx, None) == 0.0


def test_a_broken_policy_contributes_no_delta(world, ctx):
    planner = make_planner(world, policy=_Boom())
    assert planner.policy_delta(ctx, "a") == 0.0
    assert planner.policy_delta(ctx, None) == 0.0


def test_evaluating_a_proposed_sequence_survives_a_broken_model(world, ctx):
    planner = make_planner(world)
    planner.world_model = _Boom()
    assert planner.evaluate(ctx, ["a", "b"]) == 0.0


def test_a_plan_without_a_breakdown_explains_itself_plainly(world):
    planner = make_planner(world)
    text = planner.explain(Plan(objective="obj"))
    assert text == "obj via nothing"


def test_metrics_are_a_no_op_without_telemetry(world):
    planner = make_planner(world)
    planner.publish_metrics(1)          # must not raise


def test_a_failing_telemetry_sink_does_not_break_the_tick(world):
    planner = make_planner(world, telemetry=_Boom())
    planner.last_plan = Plan(objective="obj", steps=["a"])
    planner.ev_gap = 0.1
    planner.publish_metrics(1)          # swallowed, logged, tick continues


def test_override_rate_is_zero_before_any_decision(world):
    assert make_planner(world).override_rate() == 0.0


# ── DECIDE: the gate sequence under failure ──────────────────────────

class _FakePlanner:
    """Enough planner to drive the phase without pricing anything."""

    def __init__(self, plans):
        self.plans = plans
        self.blocked = {}
        self.recorded = []

    def build(self, substrate, ctx, available):
        return list(self.plans)

    def note_blocked(self, reason):
        self.blocked[reason] = self.blocked.get(reason, 0) + 1

    def record_choice(self, chosen, greedy):
        self.recorded.append((chosen.objective, greedy))


def _plan(objective, action, score=1.0):
    plan = Plan(objective=objective, steps=[action] if action else [])
    plan.score = score
    plan.rationale = f"because {objective}"
    return plan


class _Substrate:
    """A stand-in with only what DECIDE's helpers touch."""

    def __init__(self):
        self.tick_count = 5
        self.planner = _FakePlanner([])
        self.actions = ActionSpace()
        self.autobiography = _Boom()
        self.world_model = _Boom()
        self.resources = _Boom()
        self._regulation_directives = {}


def test_plans_are_dropped_when_planning_is_switched_off(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", False)
    assert decide_phase._build_plans(_Substrate(), _Ctx(StateKey())) == []


def test_no_encoded_state_means_no_plans():
    assert decide_phase._build_plans(_Substrate(), _Ctx(None)) == []


def test_no_available_actions_means_no_plans():
    substrate = _Substrate()
    substrate.actions = type("A", (), {"available": lambda *a, **k: []})()
    assert decide_phase._build_plans(substrate, _Ctx(StateKey())) == []


def test_a_planner_that_raises_falls_back_to_the_direct_choice():
    substrate = _Substrate()
    substrate.actions = type("A", (), {"available": lambda *a, **k: [spec("rest")]})()
    substrate.planner = _Boom()
    assert decide_phase._build_plans(substrate, _Ctx(StateKey())) == []


def test_no_policy_leaves_every_plan_standing():
    plans = [_plan("a", "rest")]
    assert decide_phase._apply_rules(_Substrate(), _Ctx(StateKey()), plans) == plans


def test_a_policy_that_raises_leaves_every_plan_standing():
    substrate = _Substrate()
    substrate.policy = _Boom()
    plans = [_plan("a", "rest")]
    assert decide_phase._apply_rules(substrate, _Ctx(StateKey()), plans) == plans


def test_a_policy_may_suppress_a_plan():
    substrate = _Substrate()
    substrate.policy = type("P", (), {
        "apply_rules": staticmethod(lambda state, plans, tick=0: [])})()
    assert decide_phase._apply_rules(substrate, _Ctx(StateKey()), [_plan("a", "rest")]) == []


def test_failed_prioritisation_keeps_the_planner_order():
    substrate = _Substrate()
    substrate.priority = _Boom()
    plans = [_plan("a", "rest"), _plan("b", "dream")]
    assert decide_phase._prioritise(substrate, _Ctx(StateKey()), plans) == plans


def test_a_plan_with_no_action_is_never_reserved():
    substrate = _Substrate()
    taken = []
    substrate.acquire = lambda action: taken.append(action) or object()
    chosen, lease = decide_phase._reserve(substrate, [_plan("a", None), _plan("b", "rest")])
    assert taken == ["rest"]
    assert chosen.objective == "b" and lease is not None


def test_reserve_returns_nothing_when_nothing_is_affordable():
    substrate = _Substrate()
    substrate.acquire = lambda action: None
    assert decide_phase._reserve(substrate, [_plan("a", "rest")]) == (None, None)


# ── DECIDE step 8: the cortex may permute, never extend ──────────────

class _Cortex:
    def __init__(self, answer=None, available=True, explode=False):
        self.answer = answer
        self._available = available
        self.explode = explode
        self.seen = []

    def role_available(self, role):
        return self._available

    async def structured(self, role, messages, schema, lease=None):
        if self.explode:
            raise RuntimeError("cortex down")
        self.seen.append(messages)
        return self.answer


def _rerank_substrate(answer=None, available=True, explode=False):
    substrate = _Substrate()
    substrate.llm = type("L", (), {"cortex": _Cortex(answer, available, explode)})()
    substrate.resources = type("R", (), {"release": staticmethod(lambda lease: None)})()
    substrate.release = lambda lease: None
    substrate.acquire = lambda action: object()
    return substrate


def _run(coro):
    return asyncio.run(coro)


def test_a_shortlist_of_one_is_not_worth_a_model_call():
    substrate = _rerank_substrate()
    chosen = _plan("a", "rest")
    got, lease = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), [chosen], chosen, "L"))
    assert got is chosen and lease == "L"
    assert substrate.llm.cortex.seen == []


def test_an_unavailable_role_leaves_the_order_alone():
    substrate = _rerank_substrate(available=False)
    chosen = _plan("a", "rest")
    ordered = [chosen, _plan("b", "dream")]
    got, _ = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), ordered, chosen, "L"))
    assert got is chosen


def test_regulation_can_switch_the_model_out_of_the_loop():
    substrate = _rerank_substrate(answer={"order": [1, 0]})
    substrate._regulation_directives = {"skip_llm": True}
    chosen = _plan("a", "rest")
    ordered = [chosen, _plan("b", "dream")]
    got, _ = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), ordered, chosen, "L"))
    assert got is chosen
    assert substrate.llm.cortex.seen == []


def test_a_cortex_failure_keeps_the_planner_order():
    substrate = _rerank_substrate(explode=True)
    chosen = _plan("a", "rest")
    ordered = [chosen, _plan("b", "dream")]
    got, _ = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), ordered, chosen, "L"))
    assert got is chosen


@pytest.mark.parametrize("answer", [None, {}, {"order": []},
                                    {"order": ["second"]}, {"order": [7]},
                                    {"order": [-1]}])
def test_an_unusable_answer_changes_nothing(answer):
    """Out-of-range and non-integer indices are discarded, not followed.

    This is the containment property: whatever a model returns, the decision
    stays inside the shortlist the planner built.
    """
    substrate = _rerank_substrate(answer=answer)
    chosen = _plan("a", "rest")
    ordered = [chosen, _plan("b", "dream")]
    got, _ = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), ordered, chosen, "L"))
    assert got is chosen


def test_a_reordering_that_agrees_costs_nothing():
    substrate = _rerank_substrate(answer={"order": [0, 1]})
    chosen = _plan("a", "rest")
    ordered = [chosen, _plan("b", "dream")]
    got, lease = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), ordered, chosen, "L"))
    assert got is chosen and lease == "L"


def test_a_new_preference_must_be_paid_for():
    substrate = _rerank_substrate(answer={"order": [1, 0]})
    leases = []
    substrate.acquire = lambda action: leases.append(action) or f"lease:{action}"
    chosen = _plan("a", "rest")
    other = _plan("b", "dream")
    got, lease = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), [chosen, other], chosen, "L"))
    assert got is other
    assert got.source == "planner+cortex"
    assert lease == "lease:dream" and leases == ["dream"]


def test_an_unaffordable_preference_falls_back_to_what_can_be_paid_for():
    substrate = _rerank_substrate(answer={"order": [1, 0]})
    # The preferred action cannot be leased; the first affordable one wins.
    substrate.acquire = lambda action: None if action == "dream" else "lease:rest"
    chosen = _plan("a", "rest")
    other = _plan("b", "dream")
    got, lease = _run(decide_phase._cortex_rerank(
        substrate, _Ctx(StateKey()), [chosen, other], chosen, "L"))
    assert got is chosen and lease == "lease:rest"


# ── DECIDE steps 9-12: confidence, logging, experience, forecast ─────

def test_confidence_survives_an_unreadable_risk_history():
    substrate = _Substrate()
    substrate._compute_confidence = lambda: 0.8
    confidence, reasoning = decide_phase._adjust_confidence(
        substrate, _Ctx(StateKey()), _plan("a", "rest"))
    assert confidence == 0.8
    assert reasoning == "because a"


def test_a_plan_with_no_rationale_still_explains_the_choice():
    substrate = _Substrate()
    substrate._compute_confidence = lambda: 0.8
    plan = _plan("a", "rest")
    plan.rationale = ""
    _, reasoning = decide_phase._adjust_confidence(substrate, _Ctx(StateKey()), plan)
    assert reasoning == "Planned a"


def test_nothing_is_logged_when_the_plan_agreed_with_the_runner_up():
    substrate = _Substrate()
    logged = []
    substrate.autobiography = type("A", (), {
        "log_event": staticmethod(lambda *a: logged.append(a))})()
    chosen, runner_up = _plan("a", "rest", 1.0), _plan("b", "dream", 1.0)
    decide_phase._log_if_it_changed_anything(substrate, chosen, [chosen, runner_up])
    assert logged == []


def test_a_sole_candidate_has_nothing_to_have_changed():
    substrate = _Substrate()
    logged = []
    substrate.autobiography = type("A", (), {
        "log_event": staticmethod(lambda *a: logged.append(a))})()
    chosen = _plan("a", "rest", 1.0)
    decide_phase._log_if_it_changed_anything(substrate, chosen, [chosen])
    assert logged == []


def test_a_decisive_margin_is_written_to_the_autobiography():
    substrate = _Substrate()
    logged = []
    substrate.autobiography = type("A", (), {
        "log_event": staticmethod(lambda *a: logged.append(a))})()
    chosen = _plan("a", "rest", 1.0)
    runner_up = _plan("b", "dream", 1.0 - cfg.PLAN_LOG_THRESHOLD * 10)
    decide_phase._log_if_it_changed_anything(substrate, chosen, [chosen, runner_up])
    assert logged and "a via rest" in logged[0][1]


def test_a_forecast_needs_a_state_and_a_subject():
    substrate = _Substrate()
    ctx = _Ctx(None)
    decide_phase._record_prediction(substrate, ctx)
    assert ctx.prediction is None

    ctx = _Ctx(StateKey())
    ctx.action = ctx.decision = None
    decide_phase._record_prediction(substrate, ctx)
    assert ctx.prediction is None


def test_a_failing_forecast_is_absorbed():
    substrate = _Substrate()          # world_model raises on everything
    ctx = _Ctx(StateKey())
    ctx.action = "rest"
    decide_phase._record_prediction(substrate, ctx)
    assert ctx.prediction is None


def test_a_failing_causal_chain_is_absorbed():
    substrate = _Substrate()
    substrate.tick_count = cfg.WORLD_MODEL_EVERY_N_TICKS
    substrate.goals = type("G", (), {"goals": []})()
    decide_phase._build_causal_chain(substrate, {"name": "focus"})   # must not raise


def test_no_focus_means_no_chain():
    substrate = _Substrate()
    substrate.tick_count = cfg.WORLD_MODEL_EVERY_N_TICKS
    decide_phase._build_causal_chain(substrate, None)                # must not raise


def test_meta_goal_generation_failure_is_absorbed():
    substrate = _Substrate()
    substrate.tick_count = 30
    substrate.health = _Boom()
    decide_phase._generate_meta_goals(substrate)                     # must not raise


def test_meta_goals_are_only_generated_on_cadence():
    substrate = _Substrate()
    substrate.tick_count = 31
    substrate.health = _Boom()        # would raise if it were reached
    decide_phase._generate_meta_goals(substrate)


def test_opening_the_experience_absorbs_both_halves_failing():
    substrate = _Substrate()
    substrate.goal_intelligence = _Boom()
    substrate.feedback_loop = _Boom()
    substrate.health = _Boom()
    substrate._pending_experiences = {}
    decide_phase._open_experience(substrate, _Ctx(StateKey()), None, "d", None)
    assert substrate._pending_experiences == {}


# ── DECIDE: the path taken when planning cannot happen ───────────────

class _Goals:
    curiosity_level = 0.4
    goals = []

    @staticmethod
    def get_current_focus():
        return {"name": "expand_knowledge"}


class _Fallback:
    """A substrate with the pieces ``_decide_without_a_plan`` reads."""

    def __init__(self):
        self.tick_count = 9
        self.goals = _Goals()
        self.emotions = type("E", (), {"energy": 0.7, "mood": "curious"})()
        self.consciousness = type("C", (), {"mode": "focused"})()
        self.health = type("H", (), {"successful_ticks": 8, "failed_ticks": 2,
                                     "error_count": 1})()
        self.world_model = type("W", (), {
            "risks_for": staticmethod(lambda tokens: [])})()
        self.ethics = type("Et", (), {"evaluate_action": staticmethod(
            lambda info: {"status": "ok", "score": 0.9})})()
        self.goal_intelligence = type("G", (), {"choose": staticmethod(
            lambda options, context: {"objective": options[0]})})()
        self._regulation_directives = {}
        self.settled = []
        self.published = []

    def _compute_confidence(self):
        return 0.6

    def _is_llm_tick(self):
        return False

    def acquire(self, action):
        return None

    def settle(self, lease, value=0.0):
        self.settled.append((lease, value))


def test_a_broken_value_table_keeps_the_heuristic_pick():
    substrate = _Fallback()
    substrate.goal_intelligence = _Boom()
    decision, confidence, reasoning, plan, lease = _run(
        decide_phase._decide_without_a_plan(substrate, _Ctx(), "greedy", ["rest"]))
    assert decision == "greedy" and plan is None and lease is None
    assert confidence == 0.6


def test_known_failure_modes_cost_confidence_on_the_fallback_path():
    substrate = _Fallback()
    substrate.world_model = type("W", (), {"risks_for": staticmethod(
        lambda tokens: [{"failure_rate": 1.0}, {"failure_rate": 1.0}])})()
    _, confidence, reasoning, _, _ = _run(
        decide_phase._decide_without_a_plan(substrate, _Ctx(), "greedy", []))
    assert confidence < 0.6
    assert "known failure mode" in reasoning


def test_an_unreadable_risk_history_leaves_the_fallback_confidence_alone():
    substrate = _Fallback()
    substrate.world_model = _Boom()
    _, confidence, _, _, _ = _run(
        decide_phase._decide_without_a_plan(substrate, _Ctx(), "greedy", []))
    assert confidence == 0.6


def test_the_free_choice_path_needs_a_lease():
    substrate = _Fallback()
    substrate.llm = _Boom()             # would raise if it were reached
    decision, confidence, reasoning = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest"], 0.6, "why"))
    assert (decision, confidence, reasoning) == ("greedy", 0.6, "why")


def test_regulation_can_withhold_the_free_choice_lease():
    substrate = _Fallback()
    substrate._is_llm_tick = lambda: True
    substrate._regulation_directives = {"skip_llm": True}
    substrate.llm = _Boom()
    decision, _, _ = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest"], 0.6, "why"))
    assert decision == "greedy"


def _with_llm(answer, lease="L"):
    substrate = _Fallback()
    substrate._is_llm_tick = lambda: True
    substrate.acquire = lambda action: lease

    async def _make_decision(options, context, lease=None):
        return answer

    substrate.llm = type("L", (), {"make_decision": staticmethod(_make_decision)})()

    async def _publish(event):
        substrate.published.append(event)

    substrate.event_bus = type("B", (), {"publish": staticmethod(_publish)})()
    return substrate


def test_the_model_may_pick_from_the_options_it_was_given():
    substrate = _with_llm({"success": True,
                           "parsed": {"chosen": 2, "confidence": 0.9,
                                      "reasoning": "the second one"}})
    decision, confidence, reasoning = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest", "dream"], 0.6, "why"))
    assert decision == "rest"           # options = [greedy, rest, dream], 1-based
    assert confidence == 0.9 and reasoning == "the second one"
    assert substrate.settled == [("L", 1.0)]
    assert substrate.published and substrate.published[0].event_type == "llm_decision"


def test_an_out_of_range_pick_is_ignored_but_the_rest_is_kept():
    substrate = _with_llm({"success": True,
                           "parsed": {"chosen": 99, "confidence": 0.75,
                                      "reasoning": "off the end"}})
    decision, confidence, _ = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest"], 0.6, "why"))
    assert decision == "greedy" and confidence == 0.75


def test_a_failed_call_changes_nothing_and_scores_zero():
    substrate = _with_llm({"success": False, "parsed": None})
    decision, confidence, reasoning = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest"], 0.6, "why"))
    assert (decision, confidence, reasoning) == ("greedy", 0.6, "why")
    assert substrate.settled == [("L", 0.0)]


def test_a_non_dict_payload_is_ignored():
    substrate = _with_llm({"success": True, "parsed": ["not", "a", "dict"]})
    decision, _, _ = _run(decide_phase._llm_decision(
        substrate, _Ctx(), "greedy", ["rest"], 0.6, "why"))
    assert decision == "greedy"


# ── DECIDE: _choose falling through every gate ───────────────────────

class _Registry:
    """An action space that is always wired, so the fake substrate can plan."""

    def __init__(self):
        real = ActionSpace()
        self.by_name = dict(real.by_name)
        self.blocked = {}

    def available(self, substrate, ctx=None):
        return [self.by_name["rest"], self.by_name["dream"]]

    def note_blocked(self, reason):
        self.blocked[reason] = self.blocked.get(reason, 0) + 1


class _Chooser(_Fallback):
    """A substrate complete enough to run ``_choose`` end to end."""

    def __init__(self, plans, policy=None, affordable=True):
        super().__init__()
        self.planner = _FakePlanner(plans)
        self.actions = _Registry()
        self.priority = type("P", (), {"order": staticmethod(lambda c, ctx: c)})()
        self.autobiography = type("A", (), {"log_event": staticmethod(lambda *a: None)})()
        self.llm = type("L", (), {"cortex": _Cortex(None, available=False)})()
        self.self_preservation = type("SP", (), {
            "is_modification_safe": staticmethod(lambda *a: (True, {}))})()
        self.resources = type("R", (), {"release": staticmethod(lambda lease: None)})()
        self._affordable = affordable
        if policy is not None:
            self.policy = policy

    def release(self, lease):
        pass

    def acquire(self, action):
        return "L" if self._affordable else None


def test_a_policy_that_suppresses_everything_falls_back(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    substrate = _Chooser(
        [_plan("a", "rest")],
        policy=type("P", (), {
            "apply_rules": staticmethod(lambda s, p, tick=0: [])})())
    decision, _, _, plan, lease = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert decision == "greedy" and plan is None and lease is None
    assert substrate.planner.blocked["policy"] == 1


def test_nothing_affordable_falls_back(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    substrate = _Chooser([_plan("a", "rest")], affordable=False)
    decision, _, _, plan, lease = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert decision == "greedy" and plan is None
    assert substrate.planner.blocked["resources"] == 1


def test_an_ethics_veto_walks_the_shortlist_then_gives_up(monkeypatch):
    """Bounded retries: three refusals and the planner stops arguing."""
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    plans = [_plan("a", "rest"), _plan("b", "dream"), _plan("c", "self_inspect"),
             _plan("d", "consolidate_memory")]
    substrate = _Chooser(plans)
    tried = []

    def _evaluate(info):
        tried.append(info["type"])
        return {"status": "blocked", "score": 0.0}

    substrate.ethics = type("Et", (), {"evaluate_action": staticmethod(_evaluate)})()
    decision, _, _, plan, _ = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert plan is None and decision == "greedy"
    # The fallback path evaluates once more, so the plan attempts are bounded
    # by MAX_ETHICS_RETRIES rather than by the number of candidates.
    assert len(tried) == decide_phase.MAX_ETHICS_RETRIES + 1


def test_a_veto_on_the_last_candidate_ends_the_walk(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    substrate = _Chooser([_plan("a", "rest")])
    substrate.ethics = type("Et", (), {"evaluate_action": staticmethod(
        lambda info: {"status": "blocked", "score": 0.0})})()
    _, _, _, plan, _ = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert plan is None


def test_a_veto_then_nothing_affordable_ends_the_walk(monkeypatch):
    """Second candidate exists but cannot be paid for — the loop must stop."""
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    substrate = _Chooser([_plan("a", "rest"), _plan("b", "dream")])
    substrate.ethics = type("Et", (), {"evaluate_action": staticmethod(
        lambda info: {"status": "blocked", "score": 0.0})})()
    calls = {"n": 0}

    def _acquire(action):
        calls["n"] += 1
        return "L" if calls["n"] == 1 else None

    substrate.acquire = _acquire
    _, _, _, plan, _ = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert plan is None


def test_self_preservation_refuses_an_irreversible_action(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    substrate = _Chooser([_plan("a", "code_self_mod")])
    substrate.self_preservation = type("SP", (), {
        "is_modification_safe": staticmethod(lambda *a: (False, {"reason": "no"}))})()
    _, _, _, plan, _ = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert plan is None
    assert substrate.planner.blocked["self_preservation"] >= 1


def test_a_plan_that_clears_every_gate_is_returned(monkeypatch):
    monkeypatch.setattr(decide_phase, "PLAN_ENABLED", True)
    chosen = _plan("a", "rest")
    substrate = _Chooser([chosen])
    decision, confidence, reasoning, plan, lease = _run(
        decide_phase._choose(substrate, _Ctx(StateKey()), "greedy", []))
    assert plan is chosen and lease == "L" and decision == "a"
    assert substrate.planner.recorded == [("a", "greedy")]


# ── executors: the adapters that need tick context ───────────────────

def test_the_dream_adapter_assembles_its_own_material():
    from aegis.layers.executors import adapter_for

    seen = {}

    class _Dreams:
        @staticmethod
        def generate_dream(mood, recent, concepts):
            seen.update(mood=mood, recent=recent, concepts=concepts)
            return {"narrative": "a dream"}

    class _S:
        memory = type("M", (), {
            "episodic": [{"event": "one"}, {"event": "two"}],
            "semantic": {"entropy": {}, "graphs": {}}})()
        emotions = type("E", (), {"mood": "curious"})()
        dreams = _Dreams()

    assert adapter_for("dream")(_S(), None) == {"narrative": "a dream"}
    assert seen["mood"] == "curious"
    assert seen["recent"] == ["one", "two"]
    assert seen["concepts"] == ["entropy", "graphs"]


def test_there_is_no_adapter_for_an_unknown_action():
    from aegis.layers.executors import adapter_for

    assert adapter_for("no_such_action") is None


def test_the_backup_adapter_labels_the_snapshot_as_planned():
    from aegis.layers.executors import adapter_for

    seen = {}

    class _S:
        state_backup = type("B", (), {"save_state": staticmethod(
            lambda status, label: seen.update(status=status, label=label) or "ok")})()

        @staticmethod
        def full_status():
            return {"tick": 3}

    assert adapter_for("backup_state")(_S(), None) == "ok"
    assert seen == {"status": {"tick": 3}, "label": "planned"}


def test_the_inspect_adapter_looks_only_at_recent_decisions():
    from aegis.layers.executors import adapter_for

    seen = {}
    trace = [{"i": i} for i in range(30)]

    class _S:
        introspection = type("I", (), {
            "decision_trace": trace,
            "detect_bias": staticmethod(
                lambda rows: seen.update(n=len(rows)) or {"bias": None})})()

    adapter_for("self_inspect")(_S(), None)
    assert seen["n"] == 20


def test_the_rest_adapter_recovers_a_little_energy():
    from aegis.layers.executors import adapter_for

    seen = {}

    class _S:
        emotions = type("E", (), {"recharge": staticmethod(
            lambda amount: seen.update(amount=amount) or 0.75)})()

    assert adapter_for("rest")(_S(), None) == 0.75
    assert seen["amount"] == 0.05


def test_the_consolidate_adapter_forgets_before_it_ingests():
    from aegis.layers.executors import adapter_for

    order = []

    class _S:
        memory = type("M", (), {"apply_forgetting": staticmethod(
            lambda: order.append("forget"))})()
        cognitive_graph = type("G", (), {"ingest_memory": staticmethod(
            lambda memory: order.append("ingest") or {"nodes": 1})})()

    assert adapter_for("consolidate_memory")(_S(), None) == {"nodes": 1}
    assert order == ["forget", "ingest"]


# ── the planner's own empty cases ────────────────────────────────────

class _NoSequence:
    """A world model that can price nothing — every rollout comes back empty."""

    @staticmethod
    def rollout(state, actions, depth, beam):
        return type("R", (), {"sequence": [], "value": 0.0})()


def test_a_rollout_that_returns_no_sequence_yields_no_plan(ctx):
    planner = Planner(world_model=_NoSequence(), actions=ActionSpace(),
                      goal_intelligence=_Values())
    assert planner.plan_for("obj", ctx, [spec("rest")]) is None


def test_unpriceable_objectives_are_simply_absent_from_the_shortlist(ctx):
    planner = Planner(world_model=_NoSequence(), actions=ActionSpace(),
                      goal_intelligence=_Values())

    class _Substrate:
        goals = _Boom()
        cognitive_graph = _Boom()
        meta_goals = _Boom()

    assert planner.build(_Substrate(), ctx, [spec("rest")]) == []


def test_a_step_outside_the_offered_set_costs_nothing(world, ctx):
    """A rollout may name a step the caller did not offer; it is priced at zero
    rather than crashing the plan it belongs to."""
    planner = make_planner(world)

    class _Model:
        @staticmethod
        def rollout(state, actions, depth, beam):
            return type("R", (), {"sequence": ["rest", "unknown_step"],
                                  "value": 1.0})()

        @staticmethod
        def risks_for(tokens):
            return []

        @staticmethod
        def predict_outcome(state, action):
            return type("O", (), {"reward_sd": 0.0, "p_success_pessimistic": 0.5})()

        @staticmethod
        def knows(state, action):
            return 0.5

    planner.world_model = _Model()
    plan = planner.plan_for("obj", ctx, [spec("rest", ms=10)])
    assert plan.steps == ["rest", "unknown_step"]
    assert plan.expected_cost.wall_ms == 10


def test_no_focus_means_no_graph_neighbours(world):
    planner = make_planner(world)

    class _Substrate:
        goals = type("G", (), {"goals": [],
                               "get_current_focus": staticmethod(lambda: None)})()
        cognitive_graph = _Boom()      # must not be consulted without a focus
        meta_goals = type("M", (), {"goals": []})()

    assert planner.collect_objectives(_Substrate()) == ["idle_exploration"]
