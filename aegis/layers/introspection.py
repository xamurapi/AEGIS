"""Layer 2: Introspection Engine — self-analysis (I-001..I-005).

All metrics are computed from real system state — no random activations.

Calibration here is *measured*, not assumed. The engine used to publish a
hardcoded ``ece = 0.05`` that nothing ever wrote to, and `confidence_history`
was never appended to at all — so a dashboard tile labelled ECE showed the same
constant for the lifetime of the process, and `_calibrate` applied a flat 5%
haircut whether the system was overconfident, underconfident or exactly right
(audit R5). A number that cannot move is not a measurement.

Now every traced decision is paired with the outcome REFLECT observes for it,
and both the expected calibration error and the signed bias are computed from
those pairs. Until enough pairs exist the engine reports that it does not know,
rather than a number that looks like knowledge.
"""
from aegis.clock import CLOCK
from aegis.util.stats import expected_calibration_error

#: Pairs needed before calibration is reported at all. Below this the estimate
#: swings on single outcomes, and a confident-looking ECE from four samples is
#: worse than an honest "not yet".
MIN_CALIBRATION_SAMPLES = 20

#: How many (confidence, outcome) pairs to keep. Calibration should describe how
#: the system is judging *now*, so an old regime must eventually age out.
MAX_CALIBRATION_SAMPLES = 500


class IntrospectionEngine:
    def __init__(self):
        self.decision_trace: list[dict] = []
        self.bias_reports: list[dict] = []
        #: (confidence, outcome) pairs, appended by `record_outcome`.
        self.confidence_history: list[dict] = []
        self.activation_map: dict[str, float] = {}
        #: Signed miscalibration: mean confidence minus observed success rate.
        #: Positive = overconfident. None until there is enough evidence.
        self._confidence_bias: float | None = None

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

    def record_outcome(self, success: bool) -> bool:
        """Close the loop on the most recent traced decision.

        The confidence was recorded in DECIDE, before the action; the outcome is
        known in REFLECT, after it. Pairing them is what makes calibration
        measurable at all. Returns True when a pair was recorded — a decision
        whose outcome is reported twice, or an outcome with no decision behind
        it, must not enter the sample.
        """
        if not self.decision_trace:
            return False
        latest = self.decision_trace[-1]
        if latest.get("outcome_recorded"):
            return False
        latest["outcome_recorded"] = True
        try:
            confidence = float(latest.get("confidence", 0.0))
        except (TypeError, ValueError):
            return False
        self.confidence_history.append({
            "confidence": max(0.0, min(1.0, confidence)),
            "success": bool(success),
        })
        if len(self.confidence_history) > MAX_CALIBRATION_SAMPLES:
            self.confidence_history = self.confidence_history[-MAX_CALIBRATION_SAMPLES:]
        self._recompute_bias()
        return True

    def _recompute_bias(self) -> None:
        if len(self.confidence_history) < MIN_CALIBRATION_SAMPLES:
            self._confidence_bias = None
            return
        n = len(self.confidence_history)
        mean_confidence = sum(r["confidence"] for r in self.confidence_history) / n
        observed = sum(1 for r in self.confidence_history if r["success"]) / n
        self._confidence_bias = mean_confidence - observed

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
        """Correct a stated confidence by the bias actually observed.

        Subtracting the *signed* gap is what makes this a correction rather
        than a tax: an overconfident system is pulled down, an underconfident
        one is pushed up, and a well-calibrated one is left alone. The old
        version multiplied by a fixed 0.95 in all three cases, which made a
        system that already understated itself understate itself further.

        With too few outcomes to measure, the confidence passes through
        unchanged — inventing a correction from no evidence is the thing this
        method exists to stop.
        """
        raw = max(0.0, min(1.0, raw_confidence))
        if self._confidence_bias is None:
            return raw
        return max(0.0, min(1.0, raw - self._confidence_bias))

    def get_confidence_calibration(self) -> dict:
        """Measured calibration, or an explicit statement that it is unknown.

        `ece` is None rather than a placeholder when there is not enough
        evidence: a consumer that renders None shows a dash, while a consumer
        handed 0.05 shows a number nobody measured.
        """
        samples = len(self.confidence_history)
        if samples < MIN_CALIBRATION_SAMPLES:
            return {
                "ece": None,
                "bias": None,
                "samples": samples,
                "needed": MIN_CALIBRATION_SAMPLES,
                "status": "insufficient_data",
            }
        pairs = [(r["confidence"], r["success"]) for r in self.confidence_history]
        ece = expected_calibration_error(pairs)
        return {
            "ece": round(ece, 4),
            "bias": round(self._confidence_bias, 4) if self._confidence_bias is not None else None,
            "samples": samples,
            # 0.10 is the conventional line between "usable" and "the number it
            # states is not the number it delivers".
            "status": "calibrated" if ece < 0.10 else "needs_recalibration",
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
