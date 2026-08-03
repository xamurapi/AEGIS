"""Scoring forecasts (spec M1.5, M1.9).

The ordering is the whole point: recorded before the action, scored after it.
Checked here on synthetic data where the right answer is known analytically, so
"the model is calibrated" is a fact rather than an impression — including the
acceptance criterion that it must beat both trivial baselines.
"""
import math

import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.layers.world.prediction import Prediction, PredictionScorer
from aegis.util.stats import (
    brier_score, calibration_curve, expected_calibration_error, wilson_interval,
    wilson_lower,
)


@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


@pytest.fixture
def scorer(frozen):
    return PredictionScorer(store_path=None)


def forecast(scorer, p_success=0.7, reward=0.5, tick=1, successors=None,
             other_mass=0.0, other_states=0):
    return scorer.open(Prediction(
        id=scorer.next_id(tick), tick=tick, state="s", action="a",
        p_success=p_success, expected_reward=reward, reward_sd=0.0,
        predicted_next=successors if successors is not None else [("next", 1.0)],
        other_mass=other_mass, other_states=other_states))


# ── the primitives ───────────────────────────────────────────────────

def test_a_perfect_forecast_scores_zero():
    assert brier_score(1.0, True) == 0.0
    assert brier_score(0.0, False) == 0.0


def test_a_confident_mistake_scores_worst():
    assert brier_score(1.0, False) == 1.0


def test_hedging_scores_a_quarter():
    assert brier_score(0.5, True) == 0.25


def test_a_perfectly_calibrated_model_has_no_calibration_error():
    # 70% predicted, right 70% of the time — Brier is far from zero here, and
    # that is exactly the difference ECE exists to isolate.
    pairs = [(0.7, i < 70) for i in range(100)]
    assert expected_calibration_error(pairs, bins=10) < 0.02


def test_an_overconfident_model_has_calibration_error():
    pairs = [(0.95, i < 50) for i in range(100)]
    assert expected_calibration_error(pairs, bins=10) > 0.4


def test_calibration_error_of_nothing_is_zero():
    assert expected_calibration_error([], bins=10) == 0.0
    assert expected_calibration_error([(0.5, True)], bins=0) == 0.0


def test_the_calibration_curve_has_one_row_per_bin():
    curve = calibration_curve([(0.15, True), (0.85, False)], bins=10)
    assert len(curve) == 10
    assert sum(row["n"] for row in curve) == 2


def test_an_empty_bin_reports_no_observation():
    curve = calibration_curve([(0.05, True)], bins=10)
    assert curve[9]["observed"] is None


def test_wilson_narrows_as_evidence_grows():
    thin = wilson_interval(1, 1)
    thick = wilson_interval(100, 100)
    assert (thin[1] - thin[0]) > (thick[1] - thick[0])


def test_wilson_stays_inside_the_unit_interval():
    for successes, trials in ((0, 1), (1, 1), (0, 100), (100, 100)):
        low, high = wilson_interval(successes, trials)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_with_no_evidence_admits_total_ignorance():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_clamps_impossible_counts():
    assert wilson_lower(5, 3) == wilson_lower(3, 3)


# ── open and close ───────────────────────────────────────────────────

def test_a_forecast_is_pending_until_it_is_scored(scorer):
    prediction = forecast(scorer)
    assert scorer.pending() == 1
    scorer.score(prediction.id, True, 0.5, "next")
    assert scorer.pending() == 0


def test_ids_are_unique(scorer):
    assert len({scorer.next_id(1) for _ in range(20)}) == 20


def test_scoring_an_unknown_forecast_returns_none(scorer):
    # A tick that failed legitimately leaves a forecast unclosed; that is not
    # the scorer's problem to raise about.
    assert scorer.score("no_such_prediction", True, 0.5, "next") is None


def test_a_correct_confident_forecast_scores_well(scorer):
    prediction = forecast(scorer, p_success=0.95)
    score = scorer.score(prediction.id, True, 0.5, "next")
    assert score.brier < 0.01


def test_a_wrong_confident_forecast_scores_badly(scorer):
    prediction = forecast(scorer, p_success=0.95)
    score = scorer.score(prediction.id, False, 0.5, "next")
    assert score.brier > 0.9


def test_reward_error_is_the_absolute_miss(scorer):
    prediction = forecast(scorer, reward=0.3)
    score = scorer.score(prediction.id, True, 0.8, "next")
    assert score.reward_error == pytest.approx(0.5)


def test_an_expected_successor_is_unsurprising(scorer):
    prediction = forecast(scorer, successors=[("next", 0.99)])
    assert scorer.score(prediction.id, True, 0.5, "next").nll_next < 0.05


def test_an_unlisted_successor_uses_the_leftover_mass(scorer):
    # Without a bucket for "everything else", landing outside the top-k scores
    # as an impossible event and the surprise metric measures the log floor
    # rather than the model.
    prediction = forecast(scorer, successors=[("a", 0.6)],
                          other_mass=0.4, other_states=4)
    score = scorer.score(prediction.id, True, 0.5, "elsewhere")
    assert score.nll_next == pytest.approx(-math.log(0.1))


def test_an_unlisted_successor_with_no_leftover_hits_the_floor(scorer):
    prediction = forecast(scorer, successors=[("a", 1.0)])
    score = scorer.score(prediction.id, True, 0.5, "elsewhere")
    assert score.nll_next > 10          # surprising, but finite


def test_a_forecast_reports_what_it_assigned():
    prediction = Prediction(id="p", tick=1, state="s", action="a",
                            p_success=0.5, expected_reward=0.5, reward_sd=0.0,
                            predicted_next=[("a", 0.7)],
                            other_mass=0.3, other_states=3)
    assert prediction.probability_of("a") == 0.7
    assert prediction.probability_of("b") == pytest.approx(0.1)


def test_a_forecast_with_nothing_left_over_assigns_zero():
    prediction = Prediction(id="p", tick=1, state="s", action="a",
                            p_success=0.5, expected_reward=0.5, reward_sd=0.0,
                            predicted_next=[("a", 1.0)])
    assert prediction.probability_of("b") == 0.0


def test_outstanding_forecasts_are_bounded(scorer):
    for tick in range(400):
        forecast(scorer, tick=tick)
    assert scorer.pending() <= 256


# ── the acceptance criterion (§M1.9) ─────────────────────────────────

def test_a_model_that_learned_beats_both_baselines(scorer):
    """The question §M1.9 actually asks, on data with a known answer.

    Two situations, one that succeeds 90% of the time and one that succeeds
    10%. A model that has learned the difference beats both "always predict the
    average" and "always predict 0.5"; one that has not, cannot.
    """
    for i in range(400):
        good = i % 2 == 0
        succeeded = (i % 10 != 0) if good else (i % 10 == 0)
        prediction = scorer.open(Prediction(
            id=scorer.next_id(i), tick=i, state="good" if good else "bad",
            action="a", p_success=0.9 if good else 0.1,
            expected_reward=0.5, reward_sd=0.0,
            predicted_next=[("next", 1.0)]))
        scorer.score(prediction.id, succeeded, 0.5, "next")

    report = scorer.calibration()
    assert report["beats_baselines"] is True
    assert report["brier"] < report["baseline_brier_mean"]
    assert report["brier"] < report["baseline_brier_half"]


def test_a_model_that_learned_nothing_does_not_beat_the_baselines(scorer):
    for i in range(200):
        prediction = forecast(scorer, p_success=0.5, tick=i)
        scorer.score(prediction.id, i % 2 == 0, 0.5, "next")
    assert scorer.calibration()["beats_baselines"] is False


def test_an_untrained_model_makes_no_claim(scorer):
    assert scorer.beats_baselines() is False


def test_calibration_meets_the_acceptance_thresholds(scorer):
    # Honestly calibrated data: the good state succeeds 8 times in 10 and is
    # predicted at 0.8, the bad one twice in 10 and is predicted at 0.2.
    for i in range(400):
        good = i % 2 == 0
        within_decade = (i // 2) % 10
        succeeded = within_decade < (8 if good else 2)
        prediction = scorer.open(Prediction(
            id=scorer.next_id(i), tick=i, state="good" if good else "bad",
            action="a", p_success=0.8 if good else 0.2,
            expected_reward=0.5, reward_sd=0.0,
            predicted_next=[("next", 1.0)]))
        scorer.score(prediction.id, succeeded, 0.5, "next")
    report = scorer.calibration()
    assert report["brier"] <= 0.18          # §M1.9
    assert report["ece"] <= 0.08
    assert report["reward_mae"] <= 0.12


def test_a_miscalibrated_model_is_caught_even_when_its_brier_is_good(scorer):
    # Predicting 0.99 for something that happens 80% of the time is accurate
    # more often than not — Brier stays low while the stated confidence is
    # plainly wrong. This is the failure ECE exists to name.
    for i in range(400):
        prediction = forecast(scorer, p_success=0.99, tick=i)
        scorer.score(prediction.id, i % 10 < 8, 0.5, "next")
    report = scorer.calibration()
    assert report["brier"] < 0.25
    assert report["ece"] > 0.08


# ── surprise ─────────────────────────────────────────────────────────

def test_surprise_is_zero_before_anything_happens(scorer):
    assert scorer.surprise() == 0.0


def test_a_well_modelled_world_is_unsurprising(scorer):
    for i in range(50):
        prediction = forecast(scorer, tick=i, successors=[("next", 0.99)])
        scorer.score(prediction.id, True, 0.5, "next")
    assert scorer.surprise() < 0.1


def test_an_unpredictable_world_is_surprising(scorer):
    for i in range(50):
        prediction = forecast(scorer, tick=i, successors=[("expected", 0.9)],
                              other_mass=0.1, other_states=10)
        scorer.score(prediction.id, True, 0.5, f"unexpected_{i}")
    assert scorer.surprise() > 2.0


# ── persistence ──────────────────────────────────────────────────────

def test_calibration_survives_a_restart(scorer, frozen):
    for i in range(30):
        prediction = forecast(scorer, p_success=0.8, tick=i)
        scorer.score(prediction.id, True, 0.5, "next")

    revived = PredictionScorer(store_path=None)
    revived.load_state(scorer.state_dict())
    assert revived.brier == pytest.approx(scorer.brier)
    assert revived.scored == scorer.scored


def test_loading_junk_state_is_survivable(scorer):
    scorer.load_state("not a dict")
    scorer.load_state({"brier": "lots", "calibration": "not a list"})
    assert scorer.brier is None


def test_a_malformed_calibration_row_is_skipped(scorer):
    scorer.load_state({"calibration": [[0.5, True], "junk", [0.7]]})
    assert len(scorer._calibration) == 1


def test_closed_forecasts_are_written_to_the_log(tmp_path, frozen):
    path = tmp_path / "predictions.jsonl"
    logged = PredictionScorer(store_path=path)
    prediction = forecast(logged)
    logged.score(prediction.id, True, 0.5, "next")
    assert path.exists()
    assert logged.recent(5)[0]["score"]["success"] is True


def test_the_log_is_bounded(tmp_path, frozen):
    path = tmp_path / "predictions.jsonl"
    logged = PredictionScorer(store_path=path, max_predictions=20)
    for i in range(120):
        prediction = forecast(logged, tick=i)
        logged.score(prediction.id, True, 0.5, "next")
    assert len(path.read_text(encoding="utf-8").splitlines()) <= 40


def test_a_torn_log_line_does_not_hide_the_rest(tmp_path, frozen):
    path = tmp_path / "predictions.jsonl"
    logged = PredictionScorer(store_path=path)
    prediction = forecast(logged)
    logged.score(prediction.id, True, 0.5, "next")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{ torn\n")
    assert len(logged.recent(10)) == 1


def test_reading_a_missing_log_is_empty(tmp_path, frozen):
    assert PredictionScorer(store_path=tmp_path / "absent.jsonl").recent() == []


def test_recent_serves_from_memory_not_from_the_log(tmp_path, frozen):
    """The in-memory buffer is the source; the log is only the fallback.

    `recent` was defined twice, and the later (disk-only) definition shadowed
    the in-memory one — so every dashboard poll re-read the JSONL tail, the
    exact IO the buffer exists to avoid. Deleting the log after a closure
    proves which copy answers: the process already has the data.
    """
    path = tmp_path / "predictions.jsonl"
    logged = PredictionScorer(store_path=path)
    prediction = forecast(logged)
    logged.score(prediction.id, True, 0.5, "next")
    path.unlink()                       # the disk copy is gone
    rows = logged.recent(5)
    assert len(rows) == 1
    assert rows[0]["id"] == prediction.id


def test_recent_works_without_a_store_path(frozen):
    """With store_path=None the closures still happened; the shadowed disk-only
    `recent` returned [] for them, which made the panel of a diskless scorer
    permanently empty."""
    scorer = PredictionScorer(store_path=None)
    prediction = forecast(scorer)
    scorer.score(prediction.id, True, 0.5, "next")
    rows = scorer.recent()
    assert len(rows) == 1 and rows[0]["score"]["success"] is True


def test_a_forecast_round_trips_through_its_dict_form():
    original = Prediction(id="p", tick=3, state="s", action="a", p_success=0.6,
                          expected_reward=0.4, reward_sd=0.1,
                          predicted_next=[("x", 0.5)], other_mass=0.5,
                          other_states=2, confidence=0.3, horizon=2)
    restored = Prediction.from_dict(original.to_dict())
    assert restored.id == "p" and restored.horizon == 2
    assert restored.predicted_next == [("x", 0.5)]
    assert restored.other_states == 2


def test_a_malformed_forecast_row_is_rejected():
    assert Prediction.from_dict({"no_id": True}) is None
