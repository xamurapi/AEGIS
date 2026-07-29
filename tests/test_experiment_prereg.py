"""Committing to a test before running it (spec M7.6, M7.11).

Preregistration is the difference between an experiment and a story told about
data. The plan — variable, levels, sample size, analysis — is hashed *before*
any observation exists, and any later mismatch makes the result ``invalid``
rather than merely weaker.

The failure this prevents is not exotic. Without it the engine may fit a model,
look at how the comparison came out, choose the analysis that reached
significance, and register that. A system doing only that, on nothing but
noise, still accumulates laws indefinitely.
"""
import json

import pytest

from aegis.layers.discovery.datapool import Frame
from aegis.layers.discovery.experiment import (
    ANALYSES, DESIGNS, MIN_HOLDOUT_R2, Preregistration, append_prereg,
    levels_for, preregister,
    run_observational,
)
from aegis.layers.discovery.symbolic import fit
from aegis.util.quasirandom import hash_unit


class _Hypothesis:
    id = "hyp_test"


def _model_frame(count=400, start_tick=0):
    rows = []
    for index in range(count):
        x = hash_unit("x", index)
        rows.append({"tick": start_tick + index, "x": x,
                     "y": 3.0 * x + 0.02 * (hash_unit("e", index) - 0.5)})
    return Frame.from_rows(rows)


@pytest.fixture
def model():
    return fit(_model_frame(), "y", ["x"])


# ── the plan is frozen ───────────────────────────────────────────────

def test_a_plan_is_hashed_when_it_is_frozen(model):
    record = preregister(_Hypothesis(), model, tick=10)
    assert record.frozen_hash
    assert record.intact()


def test_changing_the_plan_after_freezing_breaks_the_hash(model):
    """The whole mechanism. Every field of the plan is covered."""
    record = preregister(_Hypothesis(), model, tick=10)
    record.n_required = record.n_required + 1
    assert not record.intact()


@pytest.mark.parametrize("field,value", [
    ("analysis", "mann_whitney"), ("design", "interventional_abab"),
    ("predicted_effect", 99.0), ("direction", "decrease"),
    ("model_expr", "something else"), ("created_tick", 999),
    ("block_ticks", 7), ("hypothesis_id", "other"),
])
def test_every_field_of_the_plan_is_covered_by_the_hash(model, field, value):
    record = preregister(_Hypothesis(), model, tick=10)
    setattr(record, field, value)
    assert not record.intact(), f"{field} is not covered by the frozen hash"


def test_an_unfrozen_plan_is_not_intact(model):
    record = Preregistration(hypothesis_id="h", model_expr="", predicted_effect=0.0,
                             direction="increase", design="observational_holdout")
    assert not record.intact()


def test_a_plan_round_trips_through_a_dict(model):
    record = preregister(_Hypothesis(), model, tick=10)
    restored = Preregistration.from_dict(record.as_dict())
    assert restored.frozen_hash == record.frozen_hash
    assert restored.intact()


@pytest.mark.parametrize("bad", [None, {}, "text", {"no_id": 1}])
def test_a_malformed_record_is_not_a_plan(bad):
    assert Preregistration.from_dict(bad) is None


def test_a_record_with_unusable_numbers_is_not_a_plan():
    assert Preregistration.from_dict({"hypothesis_id": "h",
                                      "predicted_effect": "large"}) is None


# ── what may be preregistered ────────────────────────────────────────

def test_an_unknown_design_is_refused(model):
    assert preregister(_Hypothesis(), model, design="whatever") is None


def test_an_unknown_analysis_is_refused(model):
    assert preregister(_Hypothesis(), model, analysis="vibes") is None


@pytest.mark.parametrize("design", DESIGNS)
def test_every_declared_design_is_reachable(design, model):
    record = preregister(_Hypothesis(), model, design=design,
                         variable="explore_bonus", levels=(0.1, 0.2))
    assert record is not None and record.design == design


@pytest.mark.parametrize("analysis", ANALYSES)
def test_every_declared_analysis_is_reachable(analysis, model):
    record = preregister(_Hypothesis(), model, analysis=analysis)
    assert record is not None and record.analysis == analysis


def test_an_intervention_on_an_uncontrolled_variable_is_never_planned(model):
    """Appendix F is a whitelist, not a blacklist: a variable that is merely
    not forbidden may not be experimented on."""
    assert preregister(_Hypothesis(), model, design="interventional_abab",
                       variable="ETHICAL_THRESHOLD_AUTO",
                       levels=(0.1, 0.2)) is None
    assert preregister(_Hypothesis(), model, design="interventional_abab",
                       variable="something_new", levels=(0.1, 0.2)) is None


def test_an_intervention_needs_exactly_two_levels(model):
    for levels in ((), (0.1,), (0.1, 0.2, 0.3)):
        assert preregister(_Hypothesis(), model, design="interventional_abab",
                           variable="explore_bonus", levels=levels) is None


def test_the_required_sample_size_is_computed_not_guessed(model):
    """Committing to an n before seeing the data is what stops an experiment
    from being stopped the moment it happens to look significant."""
    small = preregister(_Hypothesis(), model, effect_size=0.2)
    large = preregister(_Hypothesis(), model, effect_size=1.0)
    assert small.n_required > large.n_required > 0


# ── the observational design ─────────────────────────────────────────

def test_only_data_recorded_after_the_plan_is_scored(model):
    """"After the registration" is a claim about when the observation happened.
    Slicing the last N rows would include pre-registration data whenever the
    series had gaps."""
    record = preregister(_Hypothesis(), model, tick=200)
    result = run_observational(record, model, _model_frame(count=400), ["x"], "y")
    assert result["n"] < 400


def test_a_model_that_holds_up_out_of_sample_is_supported(model):
    record = preregister(_Hypothesis(), model, tick=100)
    result = run_observational(record, model, _model_frame(count=400), ["x"], "y")
    assert result["status"] == "supported"
    assert result["r2_holdout"] >= 0.5


def test_a_model_that_does_not_hold_up_is_refuted(model):
    """Fresh data drawn from a different law. The formula was real on its own
    data and is wrong here, which is exactly what an out-of-sample test is for.
    """
    rows = [{"tick": 500 + index, "x": hash_unit("x", index),
             "y": -8.0 * hash_unit("x", index) + 5.0} for index in range(300)]
    record = preregister(_Hypothesis(), model, tick=400)
    result = run_observational(record, model, Frame.from_rows(rows), ["x"], "y")
    assert result["status"] == "refuted"


def test_an_altered_plan_makes_the_result_invalid(model):
    record = preregister(_Hypothesis(), model, tick=10)
    record.analysis = "mann_whitney"          # after freezing
    result = run_observational(record, model, _model_frame(), ["x"], "y")
    assert result["status"] == "invalid"


def test_not_enough_fresh_data_is_pending_rather_than_a_verdict(model):
    """A verdict on eight rows is not a weak verdict, it is not a verdict."""
    record = preregister(_Hypothesis(), model, tick=100_000)
    result = run_observational(record, model, _model_frame(), ["x"], "y")
    assert result["status"] == "pending"


def test_a_formula_that_cannot_be_applied_is_pending_not_refuted(model):
    """Refuted means the data disagreed. A formula that never ran did not."""
    rows = [{"tick": 500 + index, "unrelated": 1.0, "y": 1.0}
            for index in range(200)]
    record = preregister(_Hypothesis(), model, tick=400)
    result = run_observational(record, model, Frame.from_rows(rows), ["x"], "y")
    assert result["status"] == "pending"


def test_the_result_carries_the_hash_it_was_judged_under(model):
    record = preregister(_Hypothesis(), model, tick=100)
    result = run_observational(record, model, _model_frame(count=400), ["x"], "y")
    assert result["frozen_hash"] == record.frozen_hash


# ── amplitude ────────────────────────────────────────────────────────

def test_levels_are_a_fraction_of_the_range_not_of_the_value():
    """A fraction of the value would let a parameter sitting near zero be moved
    by nothing and one sitting high be moved a long way, for no reason
    connected to what is safe."""
    low, high = levels_for(0.5, 0.0, 1.0, max_delta=0.2)
    assert low == pytest.approx(0.3) and high == pytest.approx(0.7)

    low, high = levels_for(0.01, 0.0, 1.0, max_delta=0.2)
    assert high - low == pytest.approx(0.21, abs=1e-9)


def test_levels_never_leave_the_declared_range():
    low, high = levels_for(0.95, 0.0, 1.0, max_delta=0.5)
    assert low >= 0.0 and high <= 1.0


def test_a_zero_amplitude_gives_no_movement():
    assert levels_for(0.5, 0.0, 1.0, max_delta=0.0) == (0.5, 0.5)


# ── the preregistration log ──────────────────────────────────────────

def test_the_plan_is_written_to_the_log_before_the_experiment(tmp_path, model):
    path = tmp_path / "prereg.jsonl"
    record = preregister(_Hypothesis(), model, tick=5)
    assert append_prereg(path, record) is True

    written = json.loads(path.read_text(encoding="utf-8").strip())
    assert written["frozen_hash"] == record.frozen_hash


def test_the_log_appends_rather_than_replaces(tmp_path, model):
    path = tmp_path / "prereg.jsonl"
    append_prereg(path, preregister(_Hypothesis(), model, tick=1))
    append_prereg(path, preregister(_Hypothesis(), model, tick=2))
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_a_log_that_cannot_be_written_does_not_raise(tmp_path, model):
    """The experiment is more important than the audit trail is convenient —
    but the caller finds out."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    assert append_prereg(blocker / "prereg.jsonl",
                         preregister(_Hypothesis(), model)) is False


# ── the bar an observational result is judged against ────────────────

def test_the_bar_is_half_the_effect_the_plan_committed_to(model):
    """Not a flat fraction of variance.

    A relationship that explains a quarter of the variance in sample and still
    explains a quarter out of it has held up — that is a weak law, not a false
    one. A fixed "R² ≥ 0.5" would refute every genuine secondary effect the
    system has while passing anything dominant, which is a statement about
    effect size dressed up as a statement about replication.
    """
    record = preregister(_Hypothesis(), model, tick=100)
    result = run_observational(record, model, _model_frame(count=400), ["x"], "y")
    assert result["threshold"] == pytest.approx(
        max(MIN_HOLDOUT_R2, 0.5 * record.predicted_effect), abs=1e-4)
    assert result["status"] == "supported"


def test_a_plan_that_predicted_nothing_cannot_clear_its_own_bar():
    """Otherwise a model explaining nothing passes by continuing to explain
    nothing, which is the floor's whole job."""
    weak = fit(_model_frame(), "y", ["x"])
    record = preregister(_Hypothesis(), weak, tick=100)
    record.predicted_effect = 0.0
    record.freeze()
    assert max(MIN_HOLDOUT_R2, 0.5 * record.predicted_effect) == MIN_HOLDOUT_R2


# ── the plan's own fields ────────────────────────────────────────────

def test_the_default_analysis_follows_the_design():
    """An observational study checks a formula out of sample, so its analysis
    is an out-of-sample R². An intervention compares two arms, so its analysis
    is a two-sample test. Getting this backwards would run a comparison of two
    arms that do not exist, or score a formula against itself."""
    model = fit(_model_frame(), "y", ["x"])
    observational = preregister(_Hypothesis(), model,
                                design="observational_holdout")
    interventional = preregister(_Hypothesis(), model,
                                 design="interventional_abab",
                                 variable="explore_bonus", levels=(0.1, 0.2))
    assert observational.analysis == "r2_holdout"
    assert interventional.analysis == "welch_t"


def test_the_levels_survive_a_round_trip(model):
    """The levels *are* the intervention. A plan that came back from disk
    without them would describe an experiment nobody could reproduce."""
    record = preregister(_Hypothesis(), model, design="interventional_abab",
                         variable="explore_bonus", levels=(0.125, 0.275))
    restored = Preregistration.from_dict(record.as_dict())
    assert restored.levels == (0.125, 0.275)
    assert restored.variable == "explore_bonus"
    assert restored.intact()


def test_a_plan_with_no_levels_round_trips_as_none(model):
    record = preregister(_Hypothesis(), model)
    restored = Preregistration.from_dict(record.as_dict())
    assert restored.levels == () and restored.variable is None


# ── the observational result reports what it measured ────────────────

def test_the_reported_residual_is_the_error_of_the_prediction(model):
    """Actual minus predicted, in that order. A residual computed the other way
    round has the right magnitude and the wrong sign, and the sign is what says
    whether the formula runs high or low."""
    from aegis.layers.discovery import symbolic

    record = preregister(_Hypothesis(), model, tick=100)
    frame = _model_frame(count=400)
    result = run_observational(record, model, frame, ["x"], "y")

    fresh = frame.filter(lambda row: int(row.get("tick", -1)) > 100).numeric("y", "x")
    actual, predicted = [], []
    for row in fresh.rows():
        value = symbolic.predict(model, row, ["x"])
        if value is None:
            continue
        actual.append(float(row["y"]))
        predicted.append(value)
    expected = sum(a - p for a, p in zip(actual, predicted)) / len(actual)
    assert result["residual_mean"] == pytest.approx(round(expected, 6), abs=1e-6)


def test_a_bar_of_half_the_effect_is_not_the_effect_doubled(model):
    """``0.5 × effect`` and ``effect ÷ 0.5`` are both "a half" in the wrong
    direction, and the second would demand twice the in-sample fit out of
    sample — a bar no honest model clears."""
    record = preregister(_Hypothesis(), model, tick=100)
    record.predicted_effect = 0.8
    record.freeze()
    result = run_observational(record, model, _model_frame(count=400), ["x"], "y")
    assert result["threshold"] == pytest.approx(0.4, abs=1e-6)


# ── amplitude, over a range that straddles zero ──────────────────────

def test_the_span_of_a_range_that_straddles_zero_is_its_width():
    """``|high − low|``, not ``|high + low|``. For a parameter allowed to run
    from −1 to +1 the width is 2 and the sum is 0, so the second would freeze
    every symmetric parameter at its current value and quietly run an
    experiment with no intervention in it."""
    low, high = levels_for(0.0, -1.0, 1.0, max_delta=0.5)
    assert (low, high) == pytest.approx((-1.0, 1.0))


def test_the_span_is_the_width_for_an_ordinary_range():
    low, high = levels_for(5.0, 0.0, 10.0, max_delta=0.2)
    assert (low, high) == pytest.approx((3.0, 7.0))


# ── the log ──────────────────────────────────────────────────────────

def test_the_log_creates_the_directories_it_needs(tmp_path, model):
    """The preregistration is written before the experiment runs, on a path
    that may not exist yet — the first discovery of a fresh install writes the
    first line of a file in a directory nobody has made."""
    path = tmp_path / "deep" / "nested" / "prereg.jsonl"
    assert append_prereg(path, preregister(_Hypothesis(), model)) is True
    assert path.exists()


# ── whose experiment this is ─────────────────────────────────────────

def test_a_plan_takes_its_identity_from_whatever_shape_it_was_given():
    """Three shapes reach this: a hypothesis object, a bare ``{"id": ...}`` for
    an intervention started before its hypothesis was queued, and a plain id.

    The identity is what the ledger and the preregistration log are keyed on, so
    a shape that fell through to ``str(the whole object)`` would produce a plan
    the ledger could never match — and a shape that raised would take the tick
    with it.
    """
    from aegis.layers.discovery.experiment import _identity_of

    class _Object:
        id = "hyp_object"

    assert _identity_of(_Object()) == "hyp_object"
    assert _identity_of({"id": "hyp_mapping"}) == "hyp_mapping"
    assert _identity_of("hyp_bare_string") == "hyp_bare_string"
    assert _identity_of(1234) == "1234"


def test_an_intervention_planned_from_a_mapping_is_keyed_on_its_id(model):
    """The case that produced a plan named after the text of a dict."""
    record = preregister({"id": "hyp_from_mapping"}, model,
                         design="interventional_abab",
                         variable="explore_bonus", levels=(0.1, 0.2))
    assert record.hypothesis_id == "hyp_from_mapping"
