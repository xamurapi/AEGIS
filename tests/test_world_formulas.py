"""The arithmetic, pinned to exact values (spec M1.3, M1.5).

Comparative assertions — "the better action wins", "more evidence narrows the
interval" — pass just as happily when an operator is wrong, because both sides
of the comparison move together. Every formula the predictive contour rests on
is therefore checked here against a number worked out independently.

This matters more than it looks: these estimates are what the planner ranks on,
what the behaviour policy measures against, and what the discovery engine fits
laws to. A quietly wrong probability here is a wrong decision everywhere
downstream, and nothing else would report it.
"""
import math

import pytest

from aegis.layers.world.outcome import OutcomeModel
from aegis.layers.world.prediction import Prediction, PredictionScorer
from aegis.layers.world.simulate import COST_WEIGHT, RISK_WEIGHT, Simulator
from aegis.layers.world.state import StateEncoder, StateKey, bucket
from aegis.layers.world.transition import TransitionModel
from aegis.util.stats import (
    Z_95, Welford, brier_score, calibration_curve, expected_calibration_error,
    exponential_smooth, laplace_rate, wilson_interval, wilson_lower,
)


def state(name: str) -> StateKey:
    return StateKey(energy=name)


# ── Welford ──────────────────────────────────────────────────────────

def test_welford_accumulates_the_exact_sum_of_squared_deviations():
    # [1, 3, 2] -> mean 2, deviations -1/+1/0, m2 = 2.
    tracker = Welford()
    for value in (1.0, 3.0, 2.0):
        tracker.update(value)
    assert tracker.mean == pytest.approx(2.0)
    assert tracker.m2 == pytest.approx(2.0)
    assert tracker.variance() == pytest.approx(1.0)
    assert tracker.sd() == pytest.approx(1.0)


def test_welford_matches_a_two_pass_computation():
    values = [0.5, 2.5, 4.0, 1.0, 3.0, 0.25]
    tracker = Welford()
    for value in values:
        tracker.update(value)
    expected_mean = sum(values) / len(values)
    expected_m2 = sum((v - expected_mean) ** 2 for v in values)
    assert tracker.mean == pytest.approx(expected_mean)
    assert tracker.m2 == pytest.approx(expected_m2)


def test_welford_ageing_scales_both_weight_and_spread():
    tracker = Welford(n=10, mean=2.0, m2=8.0)
    tracker.scale(0.5)
    assert tracker.n == 5
    assert tracker.m2 == pytest.approx(4.0)


# ── Wilson ───────────────────────────────────────────────────────────

def test_the_wilson_interval_matches_the_closed_form():
    successes, trials, z = 7, 10, Z_95
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = (z * math.sqrt(phat * (1 - phat) / trials
                            + z * z / (4 * trials * trials))) / denominator

    low, high = wilson_interval(successes, trials)
    assert low == pytest.approx(centre - margin)
    assert high == pytest.approx(centre + margin)


def test_the_wilson_lower_bound_for_one_from_one():
    # The case the pessimism exists for: a single success is not a certainty.
    assert wilson_lower(1, 1) == pytest.approx(0.2065, abs=1e-4)


def test_the_wilson_lower_bound_for_a_long_clean_record():
    assert wilson_lower(100, 100) == pytest.approx(0.96301, abs=1e-5)


def test_a_wider_z_gives_a_wider_interval():
    narrow = wilson_interval(5, 10, z=1.0)
    wide = wilson_interval(5, 10, z=3.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ── smoothing ────────────────────────────────────────────────────────

def test_the_smoothed_rate_is_add_alpha():
    assert laplace_rate(3, 10, alpha=1.0) == pytest.approx(4 / 12)
    assert laplace_rate(3, 10, alpha=2.0) == pytest.approx(5 / 14)


def test_the_smoothed_rate_of_no_evidence_is_a_half():
    assert laplace_rate(0, 0, alpha=1.0) == pytest.approx(0.5)


def test_exponential_smoothing_is_the_declared_convex_combination():
    assert exponential_smooth(0.0, 1.0, alpha=0.25) == pytest.approx(0.25)
    assert exponential_smooth(1.0, 0.0, alpha=0.25) == pytest.approx(0.75)
    assert exponential_smooth(0.4, 0.8, alpha=0.5) == pytest.approx(0.6)


def test_brier_is_the_squared_error():
    assert brier_score(0.3, True) == pytest.approx(0.49)
    assert brier_score(0.3, False) == pytest.approx(0.09)


# ── calibration ──────────────────────────────────────────────────────

def test_calibration_error_is_the_size_weighted_gap():
    # Ten forecasts of 0.9 that all came true (gap 0.1) and ten of 0.1 that all
    # failed (gap 0.1), equally weighted -> 0.1.
    pairs = [(0.9, True)] * 10 + [(0.1, False)] * 10
    assert expected_calibration_error(pairs, bins=10) == pytest.approx(0.1)


def test_calibration_error_weights_the_bigger_bin_more():
    # Thirty forecasts with a 0.1 gap, ten with a 0.5 gap.
    pairs = [(0.9, True)] * 30 + [(0.5, False)] * 10
    expected = (30 / 40) * 0.1 + (10 / 40) * 0.5
    assert expected_calibration_error(pairs, bins=10) == pytest.approx(expected)


def test_the_calibration_curve_reports_the_declared_bin_edges():
    curve = calibration_curve([(0.1, True), (0.6, False)], bins=4)
    assert curve[0]["from"] == pytest.approx(0.0)
    assert curve[0]["to"] == pytest.approx(0.25)
    assert curve[2]["from"] == pytest.approx(0.5)
    assert curve[2]["to"] == pytest.approx(0.75)


def test_the_calibration_curve_reports_predicted_and_observed_means():
    # All three land in the 0.8-0.9 bin; two of them came true.
    curve = calibration_curve([(0.80, True), (0.88, False), (0.85, True)], bins=10)
    bucket_row = curve[8]
    assert bucket_row["n"] == 3
    # The curve rounds to four places for display, so the tolerance matches.
    assert bucket_row["predicted"] == pytest.approx((0.80 + 0.88 + 0.85) / 3, abs=5e-5)
    assert bucket_row["observed"] == pytest.approx(2 / 3, abs=5e-5)


def test_a_value_of_one_lands_in_the_last_bin():
    curve = calibration_curve([(1.0, True)], bins=10)
    assert curve[-1]["n"] == 1


# ── state encoding ───────────────────────────────────────────────────

def test_the_bucket_index_is_the_count_of_cuts_passed():
    labels = ("a", "b", "c", "d")
    assert bucket(0.0, [1, 2, 3], labels) == "a"
    assert bucket(1.0, [1, 2, 3], labels) == "b"
    assert bucket(2.0, [1, 2, 3], labels) == "c"
    assert bucket(3.0, [1, 2, 3], labels) == "d"


def test_the_state_space_estimate_is_the_declared_product():
    encoder = StateEncoder()
    # 3 energy x 3 error x 3 load x 3 perf x moods x modes x focus kinds.
    assert encoder.space_size(moods=6, modes=4, focus_kinds=5) == \
        3 * 3 * 3 * 3 * 6 * 4 * 5


def test_the_error_rate_reading_is_errors_over_ticks(isolated_state):
    from aegis.layers.world.state import collect_state_inputs
    from aegis.layers.substrate import Substrate
    substrate = Substrate()
    substrate.health.successful_ticks = 8
    substrate.health.failed_ticks = 2
    substrate.health.error_count = 3
    assert collect_state_inputs(substrate)["error_rate"] == pytest.approx(0.3)


def test_the_latency_reading_is_the_mean_of_the_samples(isolated_state):
    from aegis.layers.world.state import collect_state_inputs
    from aegis.layers.substrate import Substrate
    substrate = Substrate()
    substrate.health.tick_durations.clear()
    for value in (10.0, 20.0, 30.0):
        substrate.health.tick_durations.append(value)
    assert collect_state_inputs(substrate)["avg_tick_ms"] == pytest.approx(20.0)


# ── transition probabilities ─────────────────────────────────────────

def test_the_transition_estimate_is_the_declared_smoothed_form():
    """P(s'|s,a) = (c + α·P_backoff(s'|a)) / (n + α)."""
    model = TransitionModel(smoothing=1.0, min_n=3, half_life=0)
    for _ in range(2):
        model.observe(state("s"), "go", state("b"))
    model.observe(state("s"), "go", state("c"))

    # The action marginal is the only back-off level in play: b twice, c once.
    assert model.probability(state("s"), "go", state("b")) == \
        pytest.approx((2 + 1.0 * (2 / 3)) / (3 + 1.0))
    assert model.probability(state("s"), "go", state("c")) == \
        pytest.approx((1 + 1.0 * (1 / 3)) / (3 + 1.0))


def test_the_estimates_over_the_observed_successors_sum_to_one():
    model = TransitionModel(smoothing=1.0, half_life=0)
    for _ in range(2):
        model.observe(state("s"), "go", state("b"))
    model.observe(state("s"), "go", state("c"))
    total = sum(p for _, p in model.top_next(state("s"), "go", 10))
    assert total == pytest.approx(1.0)


def test_more_smoothing_pulls_harder_toward_the_back_off():
    strong = TransitionModel(smoothing=10.0, half_life=0)
    weak = TransitionModel(smoothing=0.1, half_life=0)
    for model in (strong, weak):
        for name in "defgh":
            model.observe(state(name), "go", state("common"))
        model.observe(state("z"), "go", state("rare"))
    assert strong.probability(state("z"), "go", state("common")) > \
        weak.probability(state("z"), "go", state("common"))


def test_the_action_marginal_is_a_plain_frequency():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(3):
        model.observe(state("a"), "go", state("x"))
    model.observe(state("b"), "go", state("y"))
    # From an unseen state, with no smoothing, the answer is the marginal.
    assert model.probability(state("unseen"), "go", state("x")) == pytest.approx(0.75)


def test_the_global_prior_is_used_when_the_action_is_new():
    model = TransitionModel(smoothing=0.0, half_life=0)
    for _ in range(3):
        model.observe(state("a"), "go", state("x"))
    model.observe(state("a"), "go", state("y"))
    assert model.probability(state("a"), "never_tried", state("x")) == pytest.approx(0.75)


def test_the_decay_factor_is_a_half_per_half_life():
    """One observation, then a half-life's worth of activity elsewhere."""
    model = TransitionModel(half_life=10, smoothing=1.0)
    model.observe(state("a"), "go", state("b"))          # weight 1
    for i in range(9):
        model.observe(state(f"other{i}"), "go", state("b"))
    model.observe(state("a"), "go", state("b"))
    # The old unit of evidence halved, then the new one was added.
    assert model.support(state("a"), "go") == pytest.approx(1.5)


def test_two_half_lives_quarter_the_weight():
    model = TransitionModel(half_life=5, smoothing=1.0)
    model.observe(state("a"), "go", state("b"))
    for i in range(9):
        model.observe(state(f"other{i}"), "go", state("b"))
    model.observe(state("a"), "go", state("b"))
    assert model.support(state("a"), "go") == pytest.approx(1.25)


def test_knowledge_is_the_fraction_of_the_minimum_sample():
    model = TransitionModel(min_n=4, half_life=0)
    model.observe(state("a"), "go", state("b"))
    assert model.knows(state("a"), "go") == pytest.approx(0.25)
    model.observe(state("a"), "go", state("b"))
    assert model.knows(state("a"), "go") == pytest.approx(0.5)


def test_surprise_is_the_negative_log_of_the_estimate():
    model = TransitionModel(smoothing=1.0, half_life=0)
    for _ in range(2):
        model.observe(state("s"), "go", state("b"))
    model.observe(state("s"), "go", state("c"))
    probability = model.probability(state("s"), "go", state("b"))
    assert model.surprise(state("s"), "go", state("b")) == \
        pytest.approx(-math.log(probability))


# ── outcome estimates ────────────────────────────────────────────────

def test_the_success_rate_is_the_smoothed_frequency():
    model = OutcomeModel(min_n=3, half_life=0, smoothing=1.0)
    for i in range(10):
        model.observe(state("s"), "go", success=(i < 7), reward=0.5)
    assert model.p_success(state("s"), "go") == pytest.approx(laplace_rate(7, 10, 1.0))


def test_the_pessimistic_rate_is_the_wilson_lower_bound():
    model = OutcomeModel(min_n=3, half_life=0)
    for i in range(10):
        model.observe(state("s"), "go", success=(i < 7), reward=0.5)
    assert model.p_success(state("s"), "go", pessimistic=True) == \
        pytest.approx(wilson_lower(7, 10))


def test_the_back_off_threshold_is_the_declared_minimum():
    model = OutcomeModel(min_n=5, half_life=0)
    for name in "abcdefgh":
        for _ in range(5):
            model.observe(state(name), "go", success=True, reward=0.9)
    for _ in range(4):
        model.observe(state("z"), "go", success=False, reward=0.0)
    # Four observations is under the threshold, so the marginal answers.
    assert model.predict(state("z"), "go").backed_off is True
    model.observe(state("z"), "go", success=False, reward=0.0)
    assert model.predict(state("z"), "go").backed_off is False


def test_the_reward_mean_is_exact():
    model = OutcomeModel(half_life=0)
    for value in (0.1, 0.2, 0.9):
        model.observe(state("s"), "go", success=True, reward=value)
    assert model.expected_reward(state("s"), "go") == pytest.approx(0.4)


def test_the_reward_spread_is_the_sample_standard_deviation():
    model = OutcomeModel(half_life=0)
    values = [0.0, 1.0, 0.0, 1.0]
    for value in values:
        model.observe(state("s"), "go", success=True, reward=value)
    expected_mean = sum(values) / len(values)
    expected_sd = math.sqrt(sum((v - expected_mean) ** 2 for v in values) / (len(values) - 1))
    assert model.reward_sd(state("s"), "go") == pytest.approx(expected_sd)


def test_the_outcome_decay_halves_the_weight_too():
    model = OutcomeModel(half_life=10, min_n=3)
    model.observe(state("a"), "go", success=True, reward=1.0)
    for i in range(9):
        model.observe(state(f"other{i}"), "go", success=True, reward=1.0)
    model.observe(state("a"), "go", success=True, reward=1.0)
    assert model.support(state("a"), "go") == pytest.approx(1.5)


# ── the value of one step ────────────────────────────────────────────

def build(**kwargs):
    transitions = TransitionModel(half_life=0, min_n=3)
    outcomes = OutcomeModel(half_life=0, min_n=3)
    return transitions, outcomes, Simulator(transitions, outcomes, **kwargs)


def test_a_step_is_worth_its_reward_times_its_pessimistic_chance():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    for _ in range(10):
        outcomes.observe(state("s"), "go", success=True, reward=0.5, cost=0.0)
    expected = 0.5 * outcomes.p_success(state("s"), "go", pessimistic=True)
    assert simulator.immediate_value(state("s").key(), "go") == pytest.approx(expected)


def test_cost_is_subtracted_at_the_declared_weight():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    for _ in range(10):
        outcomes.observe(state("s"), "go", success=True, reward=0.5, cost=2.0)
    expected = 0.5 * outcomes.p_success(state("s"), "go", pessimistic=True) \
        - COST_WEIGHT * 2.0
    assert simulator.immediate_value(state("s").key(), "go") == pytest.approx(expected)


def test_risk_is_subtracted_at_the_declared_weight():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    for i in range(10):
        outcomes.observe(state("s"), "go", success=True,
                         reward=float(i % 2), cost=0.0)
    pessimistic = outcomes.p_success(state("s"), "go", pessimistic=True)
    spread = outcomes.reward_sd(state("s"), "go")
    expected = (outcomes.expected_reward(state("s"), "go") * pessimistic
                - RISK_WEIGHT * spread * (1.0 - pessimistic))
    assert simulator.immediate_value(state("s").key(), "go") == pytest.approx(expected)


def test_the_exploration_bonus_is_added_in_full_for_the_unknown():
    _, _, simulator = build(explore_bonus=0.15)
    # Nothing observed: neutral reward 0.5, undecided chance 0.5, nothing known.
    assert simulator.immediate_value(state("s").key(), "untried") == \
        pytest.approx(0.5 * 0.5 + 0.15)


def test_the_exploration_bonus_vanishes_once_the_pair_is_known():
    transitions, outcomes, simulator = build(explore_bonus=0.15)
    for _ in range(10):
        outcomes.observe(state("s"), "go", success=True, reward=0.5)
    expected = 0.5 * outcomes.p_success(state("s"), "go", pessimistic=True)
    assert simulator.immediate_value(state("s").key(), "go") == pytest.approx(expected)


# ── the value of a plan ──────────────────────────────────────────────

def test_a_two_step_value_is_the_step_plus_the_discounted_next():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    simulator.discount = 0.5
    for _ in range(10):
        transitions.observe(state("s"), "go", state("t"))
        outcomes.observe(state("s"), "go", success=True, reward=0.5)
        transitions.observe(state("t"), "go", state("t"))
        outcomes.observe(state("t"), "go", success=True, reward=0.5)

    here = simulator.immediate_value(state("s").key(), "go")
    there = simulator.immediate_value(state("t").key(), "go")
    assert simulator.rollout(state("s"), ["go"], depth=2).value == \
        pytest.approx(here + 0.5 * there)


def test_pricing_a_sequence_discounts_each_further_step():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    simulator.discount = 0.5
    for _ in range(10):
        transitions.observe(state("s"), "go", state("s"))
        outcomes.observe(state("s"), "go", success=True, reward=0.5)

    step = simulator.immediate_value(state("s").key(), "go")
    assert simulator.evaluate(state("s"), ["go", "go", "go"]) == \
        pytest.approx(step + 0.5 * step + 0.25 * step)


def test_the_expected_future_is_probability_weighted():
    transitions, outcomes, simulator = build(explore_bonus=0.0)
    simulator.discount = 1.0
    # Three quarters of the time "go" leads somewhere rich, otherwise nowhere.
    for _ in range(30):
        transitions.observe(state("s"), "go", state("rich"))
    for _ in range(10):
        transitions.observe(state("s"), "go", state("poor"))
    for _ in range(40):
        outcomes.observe(state("s"), "go", success=True, reward=0.0)
        outcomes.observe(state("rich"), "go", success=True, reward=1.0)
        outcomes.observe(state("poor"), "go", success=True, reward=0.0)

    rich = simulator.immediate_value(state("rich").key(), "go")
    poor = simulator.immediate_value(state("poor").key(), "go")
    p_rich = transitions.probability(state("s"), "go", state("rich"))
    p_poor = transitions.probability(state("s"), "go", state("poor"))
    mass = p_rich + p_poor
    here = simulator.immediate_value(state("s").key(), "go")

    assert simulator.rollout(state("s"), ["go"], depth=2).value == \
        pytest.approx(here + (p_rich / mass) * rich + (p_poor / mass) * poor)


# ── the forecast's own arithmetic ────────────────────────────────────

def test_the_leftover_mass_is_split_evenly():
    prediction = Prediction(id="p", tick=1, state="s", action="a", p_success=0.5,
                            expected_reward=0.5, reward_sd=0.0,
                            predicted_next=[("a", 0.5), ("b", 0.3)],
                            other_mass=0.2, other_states=4)
    assert prediction.probability_of("elsewhere") == pytest.approx(0.05)


def test_the_baselines_are_scored_on_the_history_before_the_event():
    scorer = PredictionScorer(store_path=None)
    for i in range(4):
        prediction = scorer.open(Prediction(
            id=f"p{i}", tick=i, state="s", action="a", p_success=0.5,
            expected_reward=0.5, reward_sd=0.0, predicted_next=[("n", 1.0)]))
        scorer.score(prediction.id, True, 0.5, "n")
    # Four successes so far, so the running mean is 1.0 and the next
    # "predict the average" forecast is a perfect 1.0 — computed BEFORE the
    # fifth event is folded in, or the baseline would be seeing the future.
    prediction = scorer.open(Prediction(
        id="p4", tick=4, state="s", action="a", p_success=0.5,
        expected_reward=0.5, reward_sd=0.0, predicted_next=[("n", 1.0)]))
    score = scorer.score("p4", True, 0.5, "n")
    assert score.baseline_brier_mean == pytest.approx(0.0)
    assert score.baseline_brier_half == pytest.approx(0.25)


def test_the_first_event_has_no_history_to_average():
    scorer = PredictionScorer(store_path=None)
    prediction = scorer.open(Prediction(
        id="p", tick=0, state="s", action="a", p_success=0.9,
        expected_reward=0.5, reward_sd=0.0, predicted_next=[("n", 1.0)]))
    score = scorer.score("p", True, 0.5, "n")
    assert score.baseline_brier_mean == pytest.approx(0.25)      # falls back to 0.5


def test_surprise_is_the_mean_over_the_window():
    scorer = PredictionScorer(store_path=None, window=3)
    for i in range(5):
        prediction = scorer.open(Prediction(
            id=f"p{i}", tick=i, state="s", action="a", p_success=0.5,
            expected_reward=0.5, reward_sd=0.0,
            predicted_next=[("n", 0.5 if i < 3 else 0.25)]))
        scorer.score(prediction.id, True, 0.5, "n")
    # Only the last three are in the window: 0.5, 0.25, 0.25.
    expected = (-math.log(0.5) - math.log(0.25) - math.log(0.25)) / 3
    assert scorer.surprise() == pytest.approx(expected)


def test_confidence_combines_evidence_and_spread(tmp_path):
    from aegis.layers.world_model import PredictiveWorldModel
    model = PredictiveWorldModel(store_path=tmp_path / "model.json")
    for i in range(10):
        model.observe_transition(state("s"), "go", state("t"))
        model.observe_outcome(state("s"), "go", success=True, reward=float(i % 2))
    prediction = model.make_prediction(state("s"), "go", tick=1)
    known = model.knows(state("s"), "go")
    spread = model.outcomes.reward_sd(state("s"), "go")
    assert prediction.confidence == pytest.approx(round(known * (1 - min(1.0, spread)), 4))


def test_coverage_is_the_fraction_of_decisions_with_an_estimate(tmp_path):
    from aegis.layers.world_model import PredictiveWorldModel
    model = PredictiveWorldModel(store_path=tmp_path / "model.json")
    for _ in range(10):
        model.observe_transition(state("s"), "known", state("t"))
        model.observe_outcome(state("s"), "known", success=True, reward=0.5)
    for _ in range(3):
        model.make_prediction(state("s"), "known", tick=1)
    model.make_prediction(state("s"), "untried", tick=2)
    assert model.coverage() == pytest.approx(0.75)
