"""Tests for the IntrospectionEngine."""
import pytest

from aegis.layers.introspection import (
    IntrospectionEngine, MAX_CALIBRATION_SAMPLES, MIN_CALIBRATION_SAMPLES,
)


def test_trace_decision_records():
    e = IntrospectionEngine()
    trace = e.trace_decision("go", ["stay", "wait"], "because", confidence=0.8)
    assert trace["decision"] == "go"
    assert trace["alternatives"] == ["stay", "wait"]
    assert trace["calibrated_confidence"] <= trace["confidence"]
    assert len(e.decision_trace) == 1


def test_decision_trace_truncates_at_200():
    e = IntrospectionEngine()
    for i in range(250):
        e.trace_decision(f"d{i}", [], "r", confidence=0.5)
    assert len(e.decision_trace) <= 200


def test_inspect_activations_defaults():
    e = IntrospectionEngine()
    acts = e.inspect_activations("layer")
    assert set(acts.keys()) == {
        "attention", "reasoning", "memory_access", "goal_eval",
        "ethics_check", "creativity", "pattern_match", "planning",
    }
    assert all(0.0 <= v <= 1.0 for v in acts.values())
    assert e.activation_map == acts


def test_inspect_activations_with_metrics():
    e = IntrospectionEngine()
    metrics = {
        "memory_load": 1.0,
        "goal_pressure": 1.0,
        "ethics_load": 1.0,
        "energy": 1.0,
        "information_gain": 1.0,
        "error_rate": 1.0,
        "llm_active": True,
    }
    acts = e.inspect_activations("layer", metrics)
    # memory_access = min(1.0, 0.2 + 0.8*1.0) = 1.0
    assert acts["memory_access"] == 1.0
    # reasoning includes llm_active bump, clamped at 1.0
    assert acts["reasoning"] == 1.0


def test_inspect_activations_llm_inactive():
    e = IntrospectionEngine()
    acts = e.inspect_activations("layer", {"llm_active": False, "energy": 0.0})
    # reasoning = min(1.0, 0.2 + 0.5*0 + 0.3*0) = 0.2
    assert abs(acts["reasoning"] - 0.2) < 1e-9


def test_detect_bias_insufficient_data():
    e = IntrospectionEngine()
    report = e.detect_bias([{"confidence": 0.5}] * 3)
    assert report["status"] == "insufficient_data"
    assert report["biases"] == []


def test_detect_bias_overconfidence():
    e = IntrospectionEngine()
    decisions = [{"decision": f"d{i}", "confidence": 0.95} for i in range(10)]
    report = e.detect_bias(decisions)
    types = [b["type"] for b in report["biases"]]
    assert "overconfidence" in types
    assert len(e.bias_reports) == 1


def test_detect_bias_underconfidence():
    e = IntrospectionEngine()
    decisions = [{"decision": f"d{i}", "confidence": 0.2} for i in range(10)]
    report = e.detect_bias(decisions)
    types = [b["type"] for b in report["biases"]]
    assert "underconfidence" in types


def test_detect_bias_repetition():
    e = IntrospectionEngine()
    # All the same decision -> unique ratio very low
    decisions = [{"decision": "same", "confidence": 0.5} for _ in range(10)]
    report = e.detect_bias(decisions)
    types = [b["type"] for b in report["biases"]]
    assert "repetition_bias" in types


def test_detect_bias_no_bias():
    e = IntrospectionEngine()
    decisions = [{"decision": f"d{i}", "confidence": 0.5} for i in range(10)]
    report = e.detect_bias(decisions)
    assert report["biases_found"] == 0


def test_calibrate_clamps():
    e = IntrospectionEngine()
    assert e._calibrate(2.0) == 1.0
    assert e._calibrate(-1.0) == 0.0


# ── calibration is measured, not declared (audit R5) ─────────────────
#
# These three tests replace ones that asserted a hardcoded 0.05 and set
# `_calibration_error` by hand. They passed against an engine whose ECE could
# not move, which is precisely the defect: `confidence_history` was never
# appended to by any code path, so the published "ECE" was a literal for the
# lifetime of the process.


def _feed(engine, pairs):
    """Record (confidence, outcome) pairs through the real API."""
    for confidence, success in pairs:
        engine.trace_decision(f"d{len(engine.decision_trace)}", [], "r", confidence)
        engine.record_outcome(success)


def test_calibration_is_unknown_until_there_is_evidence():
    """A number from four samples is not a measurement. Below the floor the
    engine has to say so rather than emit a placeholder."""
    e = IntrospectionEngine()
    cal = e.get_confidence_calibration()
    assert cal["ece"] is None
    assert cal["status"] == "insufficient_data"
    assert cal["samples"] == 0 and cal["needed"] == MIN_CALIBRATION_SAMPLES

    _feed(e, [(0.8, True)] * (MIN_CALIBRATION_SAMPLES - 1))
    assert e.get_confidence_calibration()["ece"] is None


def test_a_well_calibrated_stream_scores_near_zero():
    """Says 80%, is right 80% of the time — the definition of calibrated."""
    e = IntrospectionEngine()
    _feed(e, [(0.8, i % 5 != 0) for i in range(40)])   # 32/40 = 0.8 observed
    cal = e.get_confidence_calibration()
    assert cal["ece"] < 0.05, cal
    assert cal["status"] == "calibrated"
    assert abs(cal["bias"]) < 0.05


def test_an_overconfident_stream_is_caught_and_signed():
    """Claims 90%, succeeds 20% of the time. ECE must be large, and the bias
    must be POSITIVE — the sign is what tells a correction which way to go."""
    e = IntrospectionEngine()
    _feed(e, [(0.9, i % 5 == 0) for i in range(40)])   # 8/40 = 0.2 observed
    cal = e.get_confidence_calibration()
    assert cal["ece"] > 0.5, cal
    assert cal["status"] == "needs_recalibration"
    assert cal["bias"] > 0.6


def test_calibration_corrects_in_both_directions():
    """The old version multiplied by 0.95 no matter what, so a system that
    already understated itself was pushed further down."""
    over = IntrospectionEngine()
    _feed(over, [(0.9, i % 5 == 0) for i in range(40)])
    assert over._calibrate(0.9) < 0.9

    under = IntrospectionEngine()
    _feed(under, [(0.2, i % 5 != 0) for i in range(40)])   # 0.8 observed
    assert under._calibrate(0.2) > 0.2


def test_confidence_passes_through_before_anything_is_measured():
    e = IntrospectionEngine()
    assert e._calibrate(0.73) == 0.73


def test_an_outcome_is_recorded_once_and_needs_a_decision():
    """Double-reporting one tick would let a single outcome outvote the rest,
    and an outcome with no decision behind it has no confidence to pair with."""
    e = IntrospectionEngine()
    assert e.record_outcome(True) is False          # nothing traced yet
    e.trace_decision("d", [], "r", 0.6)
    assert e.record_outcome(True) is True
    assert e.record_outcome(False) is False         # same decision, twice
    assert len(e.confidence_history) == 1


def test_the_sample_window_is_bounded():
    """Calibration must describe how the system judges now; an unbounded log
    would let a retired regime hold the estimate down forever."""
    e = IntrospectionEngine()
    _feed(e, [(0.5, True)] * (MAX_CALIBRATION_SAMPLES + 25))
    assert len(e.confidence_history) == MAX_CALIBRATION_SAMPLES


def test_status_shape():
    e = IntrospectionEngine()
    e.trace_decision("go", [], "r", confidence=0.9)
    e.inspect_activations("layer")
    e.detect_bias([{"decision": f"d{i}", "confidence": 0.95} for i in range(10)])
    s = e.status()
    assert s["total_decisions_traced"] == 1
    assert s["activation_map"]
    assert s["bias_reports_count"] == 1
    assert s["latest_bias_report"] is not None
    assert "calibration" in s
    assert len(s["recent_decisions"]) == 1


def test_status_no_data():
    e = IntrospectionEngine()
    s = e.status()
    assert s["latest_bias_report"] is None
    assert s["recent_decisions"] == []


# ── the arithmetic itself, not just its shape ────────────────────────
#
# Mutation testing found eighteen survivors here: every test asserted that an
# activation was produced and inside [0,1], none that it was the right number.
# A mutant turning `0.3 + 0.4 * goal_pressure` into `0.3 - 0.4 * goal_pressure`
# still returned a plausible float in range, so the suite stayed green while
# the map meant something else. These pin the coefficients.


def test_activation_formulas_are_pinned_to_their_coefficients():
    e = IntrospectionEngine()
    a = e.inspect_activations("main", {
        "memory_load": 0.5, "goal_pressure": 0.5, "energy": 0.5,
        "ethics_load": 0.5, "information_gain": 0.5, "error_rate": 0.5,
        "llm_active": True,
    })
    assert a["attention"] == pytest.approx(0.3 + 0.4 * 0.5 + 0.3 * 0.5)      # 0.65
    assert a["reasoning"] == pytest.approx(0.2 + 0.5 * 0.5 + 0.3 * 1.0)      # 0.75
    assert a["memory_access"] == pytest.approx(0.2 + 0.8 * 0.5)              # 0.60
    assert a["goal_eval"] == pytest.approx(0.3 + 0.7 * 0.5)                  # 0.65
    assert a["ethics_check"] == pytest.approx(0.4 + 0.6 * 0.5)               # 0.70
    assert a["creativity"] == pytest.approx(0.1 + 0.5 * 0.5 + 0.4 * 0.5)     # 0.55
    assert a["pattern_match"] == pytest.approx(0.3 + 0.4 * 0.5 + 0.3 * 0.5)  # 0.65
    assert a["planning"] == pytest.approx(0.2 + 0.4 * 0.5 + 0.4 * 0.5)       # 0.60


def test_activations_move_the_right_way_with_their_inputs():
    """Sign, not just magnitude: `attention` rises with goal pressure and with
    LOW energy, and `reasoning` rises with energy. A flipped sign keeps every
    value in range, so only a directional check catches it."""
    e = IntrospectionEngine()
    calm = e.inspect_activations("m", {"goal_pressure": 0.0, "energy": 1.0,
                                       "error_rate": 0.0, "memory_load": 0.0})
    pressed = e.inspect_activations("m", {"goal_pressure": 1.0, "energy": 1.0,
                                          "error_rate": 0.0, "memory_load": 0.0})
    drained = e.inspect_activations("m", {"goal_pressure": 0.0, "energy": 0.0,
                                          "error_rate": 0.0, "memory_load": 0.0})
    assert pressed["attention"] > calm["attention"]
    assert drained["attention"] > calm["attention"]
    assert calm["reasoning"] > drained["reasoning"]
    noisy = e.inspect_activations("m", {"goal_pressure": 0.0, "energy": 0.0,
                                        "error_rate": 1.0, "memory_load": 0.0})
    assert drained["pattern_match"] > noisy["pattern_match"]


def test_the_repetition_ratio_is_the_ratio_it_reports():
    """`unique_ratio` is the number an operator reads to judge how stuck the
    system is; a multiply where a divide belongs still produces a float."""
    e = IntrospectionEngine()
    decisions = [{"decision": "same", "confidence": 0.5} for _ in range(10)]
    decisions[0]["decision"] = "other"
    report = e.detect_bias(decisions)
    repetition = [b for b in report["biases"] if b["type"] == "repetition_bias"]
    assert repetition, report
    assert repetition[0]["unique_ratio"] == pytest.approx(0.2)   # 2 unique / 10


def test_a_non_numeric_confidence_is_refused_not_recorded():
    """The trace comes from a phase that can be handed anything by a model.
    A confidence that will not parse must leave the sample untouched rather
    than enter it as a zero."""
    e = IntrospectionEngine()
    e.trace_decision("d", [], "r", 0.5)
    e.decision_trace[-1]["confidence"] = "not a number"
    e.decision_trace[-1].pop("outcome_recorded", None)
    assert e.record_outcome(True) is False
    assert e.confidence_history == []
