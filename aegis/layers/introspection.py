"""Layer 2: Introspection Engine — self-analysis (I-001..I-005).

All metrics are computed from real system state — no random activations.
"""
import math
from aegis.clock import CLOCK


class IntrospectionEngine:
    def __init__(self):
        self.decision_trace: list[dict] = []
        self.bias_reports: list[dict] = []
        self.confidence_history: list[dict] = []
        self.activation_map: dict[str, float] = {}
        self._calibration_error = 0.05

    def trace_decision(self, decision: str, alternatives: list[str],
                       reasoning: str, confidence: float) -> dict:
        trace = {
            "timestamp": CLOCK.now(),
            "decision": decision,
            "alternatives": alternatives,
            "reasoning": reasoning,
            "confidence": confidence,
            "calibrated_confidence": self._calibrate(confidence),
        }
        self.decision_trace.append(trace)
        if len(self.decision_trace) > 200:
            self.decision_trace = self.decision_trace[-200:]
        return trace

    def inspect_activations(self, layer_name: str, system_metrics: dict | None = None) -> dict[str, float]:
        """Compute module activations from real system metrics.

        system_metrics keys (all optional, default 0.5):
            memory_load    — working memory fullness (0..1)
            goal_pressure  — ratio of active goals to capacity
            ethics_load    — recent ethics checks / total checks
            energy         — emotional energy level
            information_gain — cumulative info gain
            error_rate     — recent error rate
            llm_active     — whether LLM is currently thinking
            tick           — current tick number
        """
        m = system_metrics or {}

        memory_load = m.get("memory_load", 0.5)
        goal_pressure = m.get("goal_pressure", 0.5)
        ethics_load = m.get("ethics_load", 0.5)
        energy = m.get("energy", 0.5)
        info_gain = m.get("information_gain", 0.0)
        error_rate = m.get("error_rate", 0.0)
        llm_active = 1.0 if m.get("llm_active") else 0.0

        activations = {
            "attention": min(1.0, 0.3 + 0.4 * goal_pressure + 0.3 * (1.0 - energy)),
            "reasoning": min(1.0, 0.2 + 0.5 * energy + 0.3 * llm_active),
            "memory_access": min(1.0, 0.2 + 0.8 * memory_load),
            "goal_eval": min(1.0, 0.3 + 0.7 * goal_pressure),
            "ethics_check": min(1.0, 0.4 + 0.6 * ethics_load),
            "creativity": min(1.0, 0.1 + 0.5 * energy + 0.4 * min(info_gain, 1.0)),
            "pattern_match": min(1.0, 0.3 + 0.4 * memory_load + 0.3 * (1.0 - error_rate)),
            "planning": min(1.0, 0.2 + 0.4 * goal_pressure + 0.4 * energy),
        }
        self.activation_map = activations
        return activations

    def detect_bias(self, decisions: list[dict]) -> dict:
        if len(decisions) < 5:
            return {"status": "insufficient_data", "biases": []}

        biases = []
        confidences = [d.get("confidence", 0.5) for d in decisions]
        mean_conf = sum(confidences) / len(confidences)
        if mean_conf > 0.85:
            biases.append({
                "type": "overconfidence",
                "severity": "medium",
                "mean_confidence": round(mean_conf, 3),
            })
        elif mean_conf < 0.3:
            biases.append({
                "type": "underconfidence",
                "severity": "low",
                "mean_confidence": round(mean_conf, 3),
            })

        unique_decisions = set(d.get("decision", "") for d in decisions)
        if len(unique_decisions) < len(decisions) * 0.3:
            biases.append({
                "type": "repetition_bias",
                "severity": "high",
                "unique_ratio": round(len(unique_decisions) / len(decisions), 3),
            })

        report = {
            "timestamp": CLOCK.now(),
            "samples_analyzed": len(decisions),
            "biases_found": len(biases),
            "biases": biases,
        }
        self.bias_reports.append(report)
        return report

    def _calibrate(self, raw_confidence: float) -> float:
        calibrated = raw_confidence * (1 - self._calibration_error)
        return max(0.0, min(1.0, calibrated))

    def get_confidence_calibration(self) -> dict:
        if not self.confidence_history:
            return {"ece": self._calibration_error, "status": "nominal"}
        return {
            "ece": self._calibration_error,
            "samples": len(self.confidence_history),
            "status": "calibrated" if self._calibration_error < 0.05 else "needs_recalibration",
        }

    def status(self) -> dict:
        return {
            "total_decisions_traced": len(self.decision_trace),
            "activation_map": self.activation_map,
            "bias_reports_count": len(self.bias_reports),
            "latest_bias_report": self.bias_reports[-1] if self.bias_reports else None,
            "calibration": self.get_confidence_calibration(),
            "recent_decisions": [
                {"decision": d["decision"], "confidence": d["confidence"],
                 "time": d["timestamp"]}
                for d in self.decision_trace[-5:]
            ],
        }
