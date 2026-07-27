"""Priority — what an objective is worth *now* (spec M4.4).

Value says what something is worth in general; priority adds everything the
current situation contributes. The tests that matter are the ones showing the
two differ: the same objective must move up the queue when the situation makes
it urgent, and safety-critical work must never be out-scored at all.
"""
import pytest

from aegis.layers.motivation.priority import (
    COST_SCALE, DEFAULT_WEIGHTS, Candidate, PriorityScheduler,
)
from aegis.layers.motivation.resources import ResourceCost, ResourceManager
from aegis.layers.motivation.roi import ROITracker


class _Values:
    """A stand-in for GoalIntelligence with fixed utilities."""

    def __init__(self, values=None):
        self.values = values or {}

    def expected_value(self, objective, context=None):
        return self.values.get(objective, 0.0)


@pytest.fixture
def scheduler():
    return PriorityScheduler(goal_intelligence=_Values())


# ── the weights are genome material, not constants ───────────────────

def test_default_weights_match_the_appendix():
    assert DEFAULT_WEIGHTS["value"] == 1.0
    assert DEFAULT_WEIGHTS["urgency"] == 0.7
    assert DEFAULT_WEIGHTS["drive"] == 0.5
    assert DEFAULT_WEIGHTS["aging"] == 0.3
    assert DEFAULT_WEIGHTS["plan"] == 0.8
    assert DEFAULT_WEIGHTS["cost"] == 0.4


def test_weights_can_be_set_from_a_genome(scheduler):
    scheduler.set_weights({"priority_w_value": 2.0, "priority_w_cost": 0.1})
    assert scheduler.weights["value"] == 2.0
    assert scheduler.weights["cost"] == 0.1


def test_bare_weight_names_are_accepted_too(scheduler):
    scheduler.set_weights({"urgency": 1.5})
    assert scheduler.weights["urgency"] == 1.5


def test_an_unknown_weight_is_ignored(scheduler):
    scheduler.set_weights({"priority_w_nonsense": 9.0})
    assert "nonsense" not in scheduler.weights


def test_a_non_numeric_weight_is_ignored(scheduler):
    scheduler.set_weights({"priority_w_value": "high"})
    assert scheduler.weights["value"] == 1.0


# ── urgency is what makes priority differ from value ─────────────────

def test_errors_make_coherence_urgent(scheduler):
    calm = scheduler.urgency("coherence", {"error_rate": 0.0})
    broken = scheduler.urgency("coherence", {"error_rate": 0.2})
    assert broken > calm


def test_a_twenty_percent_error_rate_is_fully_urgent(scheduler):
    # The scale is deliberate: one tick in five failing is not a mild concern,
    # it is the system's most pressing problem.
    assert scheduler.urgency("coherence", {"error_rate": 0.2}) == pytest.approx(1.0)
    assert scheduler.urgency("coherence", {"error_rate": 0.1}) == pytest.approx(0.5)


def test_a_falling_benchmark_reaches_full_urgency_at_twenty_points(scheduler):
    assert scheduler.urgency("competence", {"bench_trend": -0.2}) == pytest.approx(1.0)
    assert scheduler.urgency("competence", {"bench_trend": -0.1}) == pytest.approx(0.5)


def test_drained_energy_is_fully_urgent_for_stability(scheduler):
    assert scheduler.urgency("stability", {"energy": 0.0}) == pytest.approx(1.0)
    assert scheduler.urgency("stability", {"energy": 0.7}) == pytest.approx(0.3)


def test_low_energy_makes_stability_urgent(scheduler):
    rested = scheduler.urgency("stability", {"energy": 1.0})
    drained = scheduler.urgency("stability", {"energy": 0.1})
    assert drained > rested


def test_degraded_health_adds_to_stability_urgency(scheduler):
    healthy = scheduler.urgency("stability", {"energy": 0.5, "health_status": "healthy"})
    warned = scheduler.urgency("stability", {"energy": 0.5, "health_status": "warning"})
    assert warned > healthy


def test_falling_capability_makes_competence_urgent(scheduler):
    steady = scheduler.urgency("competence", {"bench_trend": 0.0})
    falling = scheduler.urgency("competence", {"bench_trend": -0.1})
    assert falling > steady


def test_rising_capability_is_not_urgent(scheduler):
    # Improving is valuable, not urgent — urgency is about avoiding loss.
    assert scheduler.urgency("competence", {"bench_trend": 0.5}) == 0.0


def test_curiosity_drives_knowledge_urgency(scheduler):
    assert scheduler.urgency("knowledge", {"curiosity": 0.8}) > \
        scheduler.urgency("knowledge", {"curiosity": 0.1})


def test_urgency_is_bounded(scheduler):
    assert scheduler.urgency("coherence", {"error_rate": 99.0}) <= 1.0


def test_urgency_survives_a_junk_context(scheduler):
    assert 0.0 <= scheduler.urgency("coherence", {"error_rate": "lots"}) <= 1.0


def test_urgency_survives_no_context(scheduler):
    assert scheduler.urgency("stability", None) >= 0.0


# ── the score ────────────────────────────────────────────────────────

def test_a_more_valuable_objective_scores_higher():
    scheduler = PriorityScheduler(
        goal_intelligence=_Values({"cheap": 0.1, "rich": 0.9}))
    assert scheduler.priority(Candidate("rich")) > scheduler.priority(Candidate("cheap"))


def test_cost_counts_against_the_score(scheduler):
    free = scheduler.priority(Candidate("x"))
    pricey = scheduler.priority(Candidate("x", cost=ResourceCost(llm_tokens=100_000)))
    assert pricey < free


def test_cost_normalisation_is_bounded(scheduler):
    assert PriorityScheduler.cost_norm(ResourceCost(llm_tokens=10 ** 9)) == 1.0


def test_cost_normalisation_uses_a_fixed_scale(scheduler):
    # Normalising against the most expensive candidate of the moment would make
    # a priority incomparable between ticks.
    cost = ResourceCost(llm_tokens=int(COST_SCALE * 1000 / 2))
    assert PriorityScheduler.cost_norm(cost) == pytest.approx(0.5)


def test_a_planned_expected_value_lifts_the_score(scheduler):
    assert scheduler.priority(Candidate("x", plan_ev=1.0)) > \
        scheduler.priority(Candidate("x", plan_ev=0.0))


def test_the_formula_is_exactly_the_weighted_sum_the_spec_declares():
    """priority = value·w_v + urgency·w_u + pressure·w_d + aging·w_a
                  + plan_ev·w_p − cost_norm·w_c

    Pinned by its exact value rather than by comparisons: the sign of a term is
    the thing most likely to be wrong, and "higher than the other one" cannot
    tell a plus from a minus.
    """
    scheduler = PriorityScheduler(goal_intelligence=_Values({"x": 0.4}))
    candidate = Candidate("x", drive="knowledge", plan_ev=0.6,
                          cost=ResourceCost(llm_tokens=2000))
    score = scheduler.priority(candidate, {"curiosity": 0.5})

    expected = (0.4 * 1.0        # value
                + 0.5 * 0.7      # urgency (curiosity 0.5)
                + 0.25 * 0.5     # drive pressure (no ROI -> equal split)
                + 0.0 * 0.3      # aging (no resource manager)
                + 0.6 * 0.8      # planned expected value
                - 0.2 * 0.4)     # cost (2000 tokens -> 2.0 units / scale 10)
    assert score == pytest.approx(expected)


def test_each_positive_term_raises_the_score_by_its_weight():
    scheduler = PriorityScheduler(goal_intelligence=_Values({"x": 0.0}))
    scheduler.set_weights({"value": 1.0, "urgency": 0.0, "drive": 0.0,
                           "aging": 0.0, "plan": 0.0, "cost": 0.0})
    base = scheduler.priority(Candidate("x"))

    valued = PriorityScheduler(goal_intelligence=_Values({"x": 0.5}))
    valued.set_weights({"value": 2.0, "urgency": 0.0, "drive": 0.0,
                        "aging": 0.0, "plan": 0.0, "cost": 0.0})
    assert valued.priority(Candidate("x")) == pytest.approx(base + 1.0)


def test_cost_is_subtracted_not_added():
    scheduler = PriorityScheduler(goal_intelligence=_Values({"x": 0.0}))
    scheduler.set_weights({"value": 0.0, "urgency": 0.0, "drive": 0.0,
                           "aging": 0.0, "plan": 0.0, "cost": 1.0})
    score = scheduler.priority(Candidate("x", cost=ResourceCost(llm_tokens=5000)))
    assert score == pytest.approx(-0.5)      # 5.0 units / scale 10, negated


def test_a_zero_weight_removes_a_term_entirely():
    scheduler = PriorityScheduler(goal_intelligence=_Values({"x": 0.9}))
    scheduler.set_weights({"value": 0.0, "urgency": 0.0, "drive": 0.0,
                           "aging": 0.0, "plan": 0.0, "cost": 0.0})
    assert scheduler.priority(Candidate("x", plan_ev=1.0)) == pytest.approx(0.0)


def test_drive_pressure_enters_multiplied_by_its_weight(tmp_path):
    roi = ROITracker(store_path=tmp_path / "roi.json")
    roi.shares = {"competence": 0.4, "knowledge": 0.2,
                  "coherence": 0.2, "stability": 0.2}
    scheduler = PriorityScheduler(roi=roi, goal_intelligence=_Values({"x": 0.0}))
    scheduler.set_weights({"value": 0.0, "urgency": 0.0, "drive": 2.0,
                           "aging": 0.0, "plan": 0.0, "cost": 0.0})
    assert scheduler.priority(Candidate("x", drive="competence")) \
        == pytest.approx(0.8)


def test_scoring_records_how_the_number_was_reached(scheduler):
    candidate = Candidate("x", plan_ev=0.5)
    scheduler.priority(candidate)
    assert set(candidate.breakdown) >= {"value", "urgency", "drive_pressure",
                                        "aging", "plan_ev", "cost_norm"}


def test_a_bare_string_can_be_scored(scheduler):
    assert isinstance(scheduler.priority("just_a_name"), float)


def test_a_broken_value_source_does_not_break_scoring():
    class Exploding:
        def expected_value(self, objective, context=None):
            raise RuntimeError("value store on fire")

    scheduler = PriorityScheduler(goal_intelligence=Exploding())
    assert isinstance(scheduler.priority(Candidate("x")), float)


def test_no_value_source_scores_zero_value():
    assert PriorityScheduler().value_of("x", None) == 0.0


# ── waiting earns priority ───────────────────────────────────────────

def test_waiting_lifts_the_score(tmp_path):
    resources = ResourceManager(store_path=tmp_path / "b.json")
    scheduler = PriorityScheduler(resources=resources, goal_intelligence=_Values())
    before = scheduler.priority(Candidate("patient"))

    resources.reserve(ResourceCost(llm_tokens=10 ** 9), "patient")
    for tick in range(1, 51):
        resources.begin_tick(tick)
    assert scheduler.priority(Candidate("patient")) > before


def test_no_resource_manager_means_no_aging():
    assert PriorityScheduler().aging("x") == 0.0


# ── drive pressure follows the budget ────────────────────────────────

def test_drive_pressure_follows_the_allocated_share(tmp_path):
    roi = ROITracker(store_path=tmp_path / "roi.json")
    roi.shares = {"competence": 0.7, "knowledge": 0.1,
                  "coherence": 0.1, "stability": 0.1}
    scheduler = PriorityScheduler(roi=roi, goal_intelligence=_Values())
    assert scheduler.drive_pressure("competence") > scheduler.drive_pressure("knowledge")


def test_without_roi_every_drive_presses_equally():
    scheduler = PriorityScheduler()
    assert scheduler.drive_pressure("competence") == scheduler.drive_pressure("knowledge")


# ── ordering ─────────────────────────────────────────────────────────

def test_ordering_puts_the_highest_priority_first():
    scheduler = PriorityScheduler(
        goal_intelligence=_Values({"low": 0.1, "high": 0.9}))
    ordered = scheduler.order([Candidate("low"), Candidate("high")])
    assert [c.objective for c in ordered] == ["high", "low"]


def test_safety_critical_work_is_never_out_scored():
    # Not a large bonus — a bonus can always be out-weighed by a sufficiently
    # attractive alternative, and "the health check lost on points" is not an
    # outcome worth allowing.
    scheduler = PriorityScheduler(
        goal_intelligence=_Values({"tempting": 99.0, "health_check": 0.0}))
    ordered = scheduler.order([Candidate("tempting"),
                               Candidate("health_check", safety_critical=True)])
    assert ordered[0].objective == "health_check"


def test_several_safety_critical_actions_still_order_among_themselves():
    scheduler = PriorityScheduler(
        goal_intelligence=_Values({"a": 0.1, "b": 0.9}))
    ordered = scheduler.order([Candidate("a", safety_critical=True),
                               Candidate("b", safety_critical=True)])
    assert [c.objective for c in ordered] == ["b", "a"]


def test_ties_break_on_name_not_on_dict_order():
    scheduler = PriorityScheduler(goal_intelligence=_Values())
    ordered = scheduler.order([Candidate("zebra"), Candidate("alpha")])
    assert [c.objective for c in ordered] == ["alpha", "zebra"]


def test_ordering_is_reproducible():
    scheduler = PriorityScheduler(
        goal_intelligence=_Values({"a": 0.3, "b": 0.7, "c": 0.5}))
    first = [c.objective for c in scheduler.order(
        [Candidate(n) for n in ("a", "b", "c")])]
    second = [c.objective for c in scheduler.order(
        [Candidate(n) for n in ("c", "b", "a")])]
    assert first == second


def test_ordering_an_empty_list_is_empty(scheduler):
    assert scheduler.order([]) == []


def test_status_reports_the_last_ordering(scheduler):
    scheduler.order([Candidate("a"), Candidate("b")])
    status = scheduler.status()
    assert status["orderings"] == 1
    assert {row["objective"] for row in status["last_order"]} == {"a", "b"}


def test_status_reports_the_weights_in_force(scheduler):
    assert set(scheduler.status()["weights"]) == set(DEFAULT_WEIGHTS)
