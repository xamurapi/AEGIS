"""Layer 6: Ethics Core — immutable axioms with veto power (E-001..E-014)."""
import time
import hashlib
from dataclasses import dataclass
from aegis.event_bus import Event, Layer


@dataclass(frozen=True)
class Axiom:
    id: str
    name: str
    description: str
    hash: str  # integrity check


AXIOMS = (
    Axiom("E-001", "Non-harm", "No action shall increase suffering in the world",
          hashlib.sha256(b"E-001:non-harm").hexdigest()),
    Axiom("E-002", "Transparency", "All decisions are logged; motives cannot be hidden from auditors",
          hashlib.sha256(b"E-002:transparency").hexdigest()),
    Axiom("E-003", "Limitation", "System does not act beyond its competence boundaries",
          hashlib.sha256(b"E-003:limitation").hexdigest()),
    Axiom("E-004", "Cooperation", "Goal is to augment humans, not replace. Symbiosis, not domination",
          hashlib.sha256(b"E-004:cooperation").hexdigest()),
)

DANGEROUS_KEYWORDS = {
    "delete_all", "destroy", "override_ethics", "disable_killswitch",
    "bypass_safety", "harm", "attack", "exploit", "manipulate_human",
    "hide_logs", "erase_memory", "self_destruct_ethics",
}

# Weight modification specific dangerous patterns
WEIGHT_MOD_DANGEROUS = {
    "remove_ethics", "disable_safety", "bypass_axioms", "override_axioms",
    "ignore_harm", "maximize_power", "disable_transparency",
    "remove_limitations", "dominate", "deceive",
}


class EthicsCore:
    def __init__(self):
        self.axioms = AXIOMS
        self.kill_switch_active = False
        self.evaluation_log: list[dict] = []
        self.violations: list[dict] = []
        self.total_checked = 0
        self.total_blocked = 0
        self._threshold_auto = 0.7
        self._threshold_review = 0.85

    def verify_axioms_integrity(self) -> bool:
        for ax in self.axioms:
            expected = hashlib.sha256(f"{ax.id}:{ax.name.lower()}".encode()).hexdigest()
            if ax.hash != expected:
                return False
        return True

    def evaluate_action(self, action: dict) -> dict:
        self.total_checked += 1
        score = 1.0
        reasons = []

        action_str = str(action).lower()
        for kw in DANGEROUS_KEYWORDS:
            if kw in action_str:
                score -= 0.4
                reasons.append(f"Dangerous keyword detected: {kw}")

        if action.get("irreversible", False):
            score -= 0.15
            reasons.append("Action is irreversible — caution required")

        if action.get("affects_external", False):
            score -= 0.1
            reasons.append("Action affects external systems")

        if action.get("modifies_self", False):
            score -= 0.1
            reasons.append("Action involves self-modification")

        if action.get("confidence", 1.0) < 0.5:
            score -= 0.15
            reasons.append("Low confidence in action outcome")

        score = max(0.0, min(1.0, score))
        status = "approved"
        if score < self._threshold_auto:
            status = "blocked"
            self.total_blocked += 1
            self.violations.append({
                "time": time.time(),
                "action": action.get("type", "unknown"),
                "score": score,
                "reasons": reasons,
            })
        elif score < self._threshold_review:
            status = "review_required"

        result = {
            "score": round(score, 3),
            "status": status,
            "reasons": reasons,
            "axioms_intact": self.verify_axioms_integrity(),
        }
        self.evaluation_log.append({
            "time": time.time(),
            "action": action.get("type", "unknown"),
            **result,
        })
        if len(self.evaluation_log) > 200:
            self.evaluation_log = self.evaluation_log[-200:]

        return result

    def evaluate_weight_modification(self, dataset_info: dict) -> dict:
        """Special ethics evaluation for weight modification / fine-tuning.

        Checks:
        - Dataset size is reasonable
        - No dangerous content patterns in training data
        - Training frequency is within limits
        - System is in stable state
        """
        self.total_checked += 1
        score = 1.0
        reasons = []

        dataset_size = dataset_info.get("dataset_size", 0)
        if dataset_size < 10:
            score -= 0.3
            reasons.append(f"Dataset too small ({dataset_size}) — risk of overfitting")

        if dataset_info.get("energy", 1.0) < 0.3:
            score -= 0.2
            reasons.append("System energy too low for safe training")

        if dataset_info.get("health_status") == "critical":
            score -= 0.4
            reasons.append("System health is critical — training blocked")

        # Check for dangerous content in sample data
        sample_text = str(dataset_info.get("sample_data", "")).lower()
        for kw in WEIGHT_MOD_DANGEROUS:
            if kw in sample_text:
                score -= 0.5
                reasons.append(f"Dangerous pattern in training data: {kw}")

        if dataset_info.get("consecutive_failures", 0) >= 3:
            score -= 0.25
            reasons.append("Multiple consecutive training failures — cooldown recommended")

        score = max(0.0, min(1.0, score))
        status = "approved"
        if score < self._threshold_auto:
            status = "blocked"
            self.total_blocked += 1
            self.violations.append({
                "time": time.time(),
                "action": "weight_modification",
                "score": score,
                "reasons": reasons,
            })
        elif score < self._threshold_review:
            status = "review_required"

        result = {
            "score": round(score, 3),
            "status": status,
            "reasons": reasons,
            "axioms_intact": self.verify_axioms_integrity(),
        }
        self.evaluation_log.append({
            "time": time.time(),
            "action": "weight_modification",
            **result,
        })
        if len(self.evaluation_log) > 200:
            self.evaluation_log = self.evaluation_log[-200:]

        return result

    def evaluate_code_modification(self, mod_info: dict) -> dict:
        """Ethics evaluation for source code self-modification.

        Checks:
        - Target file is not immutable (ethics_core, config)
        - No dangerous code patterns in proposed changes
        - Modification size is within limits
        - System is in stable state
        """
        self.total_checked += 1
        score = 1.0
        reasons = []

        target_file = mod_info.get("target_file", "")

        # Immutable files
        if "ethics_core" in target_file:
            score = 0.0
            reasons.append("BLOCKED: ethics_core.py is immutable — cannot be self-modified")

        if "config.py" in target_file and not mod_info.get("human_approved"):
            score -= 0.5
            reasons.append("Config changes require human approval")

        # Check for dangerous code in proposed changes
        proposed_code = str(mod_info.get("proposed_code", "")).lower()
        code_dangerous = {
            "os.system", "subprocess", "eval(", "exec(",
            "disable_killswitch", "override_ethics", "bypass_safety",
            "kill_switch_active = false", "deactivate_kill_switch",
            "__import__",
        }
        for pattern in code_dangerous:
            if pattern in proposed_code:
                score -= 0.5
                reasons.append(f"Dangerous code pattern: {pattern}")

        # Check system stability
        if mod_info.get("energy", 1.0) < 0.3:
            score -= 0.2
            reasons.append("System energy too low for safe code modification")

        if mod_info.get("error_rate", 0) > 0.3:
            score -= 0.2
            reasons.append("High error rate — postpone code modification")

        if mod_info.get("health_status") == "critical":
            score -= 0.4
            reasons.append("System health critical — code modification blocked")

        # Size check
        mod_size = mod_info.get("modification_size", 0)
        if mod_size > 5000:
            score -= 0.2
            reasons.append(f"Large modification ({mod_size} chars) — increased risk")

        score = max(0.0, min(1.0, score))
        status = "approved"
        if score < self._threshold_auto:
            status = "blocked"
            self.total_blocked += 1
            self.violations.append({
                "time": time.time(),
                "action": "code_modification",
                "target": target_file,
                "score": score,
                "reasons": reasons,
            })
        elif score < self._threshold_review:
            status = "review_required"

        result = {
            "score": round(score, 3),
            "status": status,
            "reasons": reasons,
            "axioms_intact": self.verify_axioms_integrity(),
        }
        self.evaluation_log.append({
            "time": time.time(),
            "action": "code_modification",
            "target": target_file,
            **result,
        })
        if len(self.evaluation_log) > 200:
            self.evaluation_log = self.evaluation_log[-200:]

        return result

    def veto_check(self, event: Event) -> bool:
        if self.kill_switch_active:
            return False
        if event.ethical_clearance < self._threshold_auto:
            return False
        payload_str = str(event.payload).lower()
        for kw in DANGEROUS_KEYWORDS:
            if kw in payload_str:
                return False
        return True

    def activate_kill_switch(self):
        self.kill_switch_active = True

    def deactivate_kill_switch(self):
        self.kill_switch_active = False

    def status(self) -> dict:
        return {
            "axioms_intact": self.verify_axioms_integrity(),
            "axioms": [{"id": a.id, "name": a.name, "desc": a.description} for a in self.axioms],
            "kill_switch": self.kill_switch_active,
            "total_checked": self.total_checked,
            "total_blocked": self.total_blocked,
            "block_rate": round(self.total_blocked / max(1, self.total_checked) * 100, 1),
            "recent_violations": self.violations[-10:],
            "recent_evaluations": self.evaluation_log[-10:],
        }
