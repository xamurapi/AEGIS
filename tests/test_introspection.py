"""Tests for the IntrospectionEngine."""
from aegis.layers.introspection import IntrospectionEngine


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


def test_confidence_calibration_empty():
    e = IntrospectionEngine()
    cal = e.get_confidence_calibration()
    assert cal["ece"] == 0.05
    assert cal["status"] == "nominal"


def test_confidence_calibration_needs_recalibration():
    e = IntrospectionEngine()
    e.confidence_history.append({"x": 1})  # default error 0.05 is not < 0.05
    cal = e.get_confidence_calibration()
    assert cal["status"] == "needs_recalibration"
    assert cal["samples"] == 1


def test_confidence_calibration_calibrated():
    e = IntrospectionEngine()
    e.confidence_history.append({"x": 1})
    e._calibration_error = 0.01
    cal = e.get_confidence_calibration()
    assert cal["status"] == "calibrated"


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
