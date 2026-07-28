"""Degenerate inputs across the predictive contour (spec M1).

These are the branches that only fire when something is missing, corrupt or
empty. They are the ones that go wrong quietly — a rollout that silently
returns zero, a statistic that quietly reports a plausible number from no data
— so they get named tests rather than incidental coverage.
"""
import math

import pytest

from aegis.layers.world.outcome import OutcomeModel
from aegis.layers.world.prediction import Prediction, PredictionScorer
from aegis.layers.world.simulate import Simulator
from aegis.layers.world.state import StateEncoder, StateKey
from aegis.layers.world.transition import TransitionModel
from aegis.layers.world_model import PredictiveWorldModel
from aegis.util.stats import (
    Welford, calibration_curve, clamp, exponential_smooth, laplace_rate, mean,
    safe_log, trend,
)


def state(name: str) -> StateKey:
    return StateKey(energy=name)


# ── statistics ───────────────────────────────────────────────────────

def test_welford_restores_from_a_malformed_record():
    assert Welford.from_dict({"n": "many"}).n == 0
    assert Welford.from_dict(None).n == 0
    assert Welford.from_dict("not a mapping").n == 0


def test_welford_ageing_keeps_the_mean_and_drops_the_weight():
    # A model following a drifting world should become less certain of an old
    # estimate, not start believing a different one.
    tracker = Welford()
    for value in (1.0, 2.0, 3.0):
        tracker.update(value)
    before = tracker.mean
    tracker.scale(0.5)
    assert tracker.mean == before
    assert tracker.n < 3


def test_a_small_weight_survives_a_gentle_ageing():
    """Truncating the weight to an integer would send 1 straight to 0.

    Every observation after that would restart from nothing, so the mean would
    become whatever was seen last and the variance would collapse to zero —
    statistics that look healthy while describing a single sample.
    """
    tracker = Welford()
    tracker.update(5.0)
    tracker.scale(0.999)
    assert tracker.n > 0.9
    tracker.update(1.0)
    assert tracker.mean == pytest.approx(3.0, abs=0.01)


def test_repeated_gentle_ageing_does_not_erase_the_record():
    tracker = Welford()
    for value in (0.0, 1.0, 0.0, 1.0, 0.0, 1.0):
        tracker.scale(0.998)
        tracker.update(value)
    assert tracker.n > 5.0
    assert tracker.mean == pytest.approx(0.5, abs=0.05)
    assert tracker.sd() > 0.4


def test_welford_ageing_is_clamped():
    tracker = Welford(n=10, mean=1.0, m2=4.0)
    tracker.scale(5.0)          # nonsense factor
    assert tracker.n <= 10
    tracker.scale(-1.0)
    assert tracker.n >= 0


def test_a_smoothed_rate_with_no_smoothing_is_undecided():
    assert laplace_rate(0, 0, alpha=0.0) == 0.5
    assert laplace_rate(5, -1) == 0.5


def test_an_unobserved_rate_is_undecided_not_zero():
    assert laplace_rate(0, 0) == 0.5


def test_a_calibration_curve_with_no_bins_is_empty():
    assert calibration_curve([(0.5, True)], bins=0) == []


def test_a_calibration_curve_of_nothing_still_has_its_bins():
    assert len(calibration_curve([], bins=4)) == 4


def test_smoothing_is_seeded_by_the_first_sample():
    # Seeding at zero would make every freshly-created metric look excellent
    # for its first few hundred observations — exactly when someone is looking.
    assert exponential_smooth(None, 0.8) == 0.8


def test_smoothing_moves_toward_the_sample():
    assert 0.1 < exponential_smooth(0.0, 1.0, alpha=0.5) < 0.9


def test_a_nonsense_smoothing_weight_is_clamped():
    assert exponential_smooth(0.0, 1.0, alpha=5.0) == 1.0
    assert exponential_smooth(0.0, 1.0, alpha=-1.0) == 0.0


def test_the_log_floor_keeps_surprise_finite():
    assert safe_log(0.0) > -100
    assert math.isfinite(safe_log(0.0))


def test_clamp_bounds_on_both_sides():
    assert clamp(5, 0, 1) == 1
    assert clamp(-5, 0, 1) == 0
    assert clamp(0.5, 0, 1) == 0.5


def test_the_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
    assert mean([1, 2, 3]) == 2.0


def test_a_trend_needs_two_points():
    assert trend([]) == "flat"
    assert trend([0.5]) == "flat"


def test_a_trend_inside_the_band_is_flat():
    assert trend([0.5, 0.501], flat_band=0.01) == "flat"
    assert trend([0.5, 0.9], flat_band=0.01) == "up"
    assert trend([0.9, 0.5], flat_band=0.01) == "down"


# ── the state key ────────────────────────────────────────────────────

def test_a_state_reports_itself_as_a_mapping():
    fields = state("hi").as_dict()
    assert fields["energy"] == "hi"
    assert len(fields) == 7


def test_a_non_numeric_window_setting_falls_back():
    encoder = StateEncoder({"perf": {"window": "five", "flat_band": "wide"}})
    assert encoder.perf_label([0.1, 0.9]) == "up"


# ── the rollout's degenerate cases ───────────────────────────────────

def test_a_search_that_hits_its_ceiling_reports_zero_for_the_rest():
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    for i in range(30):
        for j in range(10):
            # Each action leads somewhere different, so memoisation cannot
            # collapse the search and the ceiling is genuinely reached.
            transitions.observe(state(f"s{i}"), f"a{j}", state(f"s{(i + j + 1) % 30}"))
            outcomes.observe(state(f"s{i}"), f"a{j}", success=True, reward=0.5)
    simulator = Simulator(transitions, outcomes, max_nodes=3)
    result = simulator.rollout(state("s0"), [f"a{j}" for j in range(10)], depth=4)
    assert result.truncated is True
    assert result.sequence          # still produces a usable first move


def test_a_dead_end_ends_the_plan_rather_than_scoring_zero():
    # An action with no recorded successor is worth its immediate value, not
    # nothing — treating the unknown future as zero would make every plan that
    # reaches new ground look worse than standing still.
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    outcomes.observe(state("s"), "leap", success=True, reward=0.9)
    simulator = Simulator(transitions, outcomes, explore_bonus=0.0)
    assert simulator.rollout(state("s"), ["leap"], depth=3).value > 0


def test_a_successor_set_with_no_mass_is_treated_as_a_dead_end():
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    outcomes.observe(state("s"), "act", success=True, reward=0.5)
    simulator = Simulator(transitions, outcomes)
    assert simulator.rollout(state("s"), ["act"], depth=2).sequence == ["act"]


def test_pricing_stops_at_the_first_unknown_step():
    transitions = TransitionModel(half_life=0)
    outcomes = OutcomeModel(half_life=0)
    outcomes.observe(state("s"), "a", success=True, reward=0.9)
    simulator = Simulator(transitions, outcomes)
    # The second step has nowhere to go, so it contributes nothing rather than
    # being priced against a state the model never saw.
    assert simulator.evaluate(state("s"), ["a", "a"]) == \
        pytest.approx(simulator.evaluate(state("s"), ["a"]))


# ── the prediction log ───────────────────────────────────────────────

def test_an_unwritable_log_does_not_break_scoring(tmp_path, monkeypatch):
    scorer = PredictionScorer(store_path=tmp_path / "sub" / "predictions.jsonl")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.open", explode)
    prediction = scorer.open(Prediction(
        id="p", tick=1, state="s", action="a", p_success=0.5,
        expected_reward=0.5, reward_sd=0.0, predicted_next=[("n", 1.0)]))
    assert scorer.score("p", True, 0.5, "n") is not None      # scoring survives


def test_an_unreadable_log_reads_as_empty(tmp_path, monkeypatch):
    path = tmp_path / "predictions.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    scorer = PredictionScorer(store_path=path)

    def explode(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr("pathlib.Path.open", explode)
    assert scorer.recent() == []


def test_truncation_leaves_a_short_log_alone(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text("{}\n{}\n", encoding="utf-8")
    scorer = PredictionScorer(store_path=path, max_predictions=100)
    scorer.rows_written = 10_000        # pretend it looked oversized
    scorer._truncate()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert scorer.rows_written == 2


def test_truncation_survives_an_unreadable_log(tmp_path, monkeypatch):
    path = tmp_path / "predictions.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    scorer = PredictionScorer(store_path=path)

    def explode(*args, **kwargs):
        raise OSError("gone")

    monkeypatch.setattr("pathlib.Path.open", explode)
    scorer._truncate()          # must not raise


def test_a_blank_line_in_the_log_is_skipped(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert len(PredictionScorer(store_path=path).recent(10)) == 2


def test_a_malformed_smoothed_metric_reads_as_absent():
    scorer = PredictionScorer(store_path=None)
    scorer.load_state({"brier": "lots", "scored": "many"})
    assert scorer.brier is None
    assert scorer.scored == 0


# ── the facade's setters and delegates ───────────────────────────────

@pytest.fixture
def wm(tmp_path):
    return PredictiveWorldModel(store_path=tmp_path / "model.json")


def test_the_chain_list_can_be_replaced(wm):
    wm.chains = [{"objective": "o"}]
    assert wm.causal.chains == [{"objective": "o"}]


def test_the_observation_counter_can_be_set(wm):
    wm.total_observations = 42
    assert wm.causal.total_observations == 42


def test_the_link_table_can_be_replaced_and_stays_searchable(wm):
    wm.links = {"planted_cause": {"effect": {"observations": 3, "successes": 0,
                                             "updated": 0.0}}}
    # The index has to be rebuilt with it, or the risk lookup would answer from
    # a table it can no longer see.
    assert wm.risks_for(["planted"])[0]["cause"] == "planted_cause"


def test_successors_can_be_asked_for_directly(wm):
    for _ in range(5):
        wm.observe_transition(state("a"), "go", state("b"))
    assert wm.predict_next(state("a"), "go", 1)[0][0] == state("b").key()


def test_a_rollout_is_reachable_through_the_facade(wm):
    for _ in range(5):
        wm.observe_transition(state("a"), "go", state("a"))
        wm.observe_outcome(state("a"), "go", success=True, reward=0.7)
    assert wm.rollout(state("a"), ["go"], depth=2).sequence == ["go", "go"]


def test_the_state_can_be_read_off_a_substrate(wm, isolated_state):
    from aegis.layers.substrate import Substrate
    assert isinstance(wm.encode_substrate(Substrate()), StateKey)


def test_a_corrupt_coverage_counter_resets(tmp_path):
    import json
    directory = tmp_path
    (directory / "calibration.json").write_text(
        json.dumps({"schema_version": 2, "covered": "lots", "decisions": 5}),
        encoding="utf-8")
    model = PredictiveWorldModel(store_path=directory / "model.json")
    assert model.coverage() == 0.0


def test_publishing_metrics_without_telemetry_is_a_no_op(wm):
    wm.publish_metrics(tick=1)      # must not raise


def test_a_broken_telemetry_store_cannot_break_publication(wm):
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("disk on fire")

    wm.telemetry = Exploding()
    wm.publish_metrics(tick=1)      # must not raise
