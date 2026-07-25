"""System 4: Goal Intelligence — autonomous value-driven goal selection (GI-001..GI-005).

Turns internal states into MOTIVATION, not just simulated feeling:

    goal → value → action choice → reward

Maintains a value/utility estimate per objective, learned from realized reward,
and selects which objective to pursue by expected value. This layer sits above
the GoalEngine (which generates candidate goals) and decides which of them is
worth the system's attention right now, given drives and past payoff.
Deterministic: utilities are running averages of real reward.
"""
import json
import time
import logging
from pathlib import Path

from aegis.config import GOAL_INTEL_DIR

logger = logging.getLogger("aegis.goal_intelligence")

# Intrinsic drives — weights on the value function. These are the standing
# "why" behind action, independent of any single goal.
DEFAULT_DRIVES = {
    "competence": 0.35,   # solve tasks / raise benchmark
    "knowledge": 0.30,    # acquire new concepts
    "coherence": 0.20,    # keep internal state consistent / low error
    "stability": 0.15,    # preserve energy / health
}

MAX_VALUE_ENTRIES = 500
LEARNING_RATE = 0.2  # how fast a utility moves toward realized reward


class GoalIntelligence:
    def __init__(self, store_path: Path | None = None):
        self.drives = dict(DEFAULT_DRIVES)
        # values[objective] = {utility, drive, attempts, updated}
        self.values: dict[str, dict] = {}
        self.decisions: list[dict] = []
        self.total_reward = 0.0
        self._last_choice: dict | None = None
        self._store_path = store_path or (GOAL_INTEL_DIR / "values.json")
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self.drives = {**DEFAULT_DRIVES, **data.get("drives", {})}
                self.values = data.get("values", {})
                self.total_reward = data.get("total_reward", 0.0)
            except Exception:
                logger.warning("Failed to load goal-intelligence state from %s — starting fresh",
                               self._store_path, exc_info=True)

    def save(self):
        data = {
            "drives": self.drives,
            "values": self.values,
            "total_reward": self.total_reward,
        }
        try:
            payload = json.dumps(data, ensure_ascii=False)
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to save goal-intelligence state to %s", self._store_path, exc_info=True)

    # ── valuation ────────────────────────────────────────────────────

    @staticmethod
    def _classify_drive(objective: str) -> str:
        """Map an objective to the intrinsic drive it primarily serves."""
        o = objective.lower()
        if any(w in o for w in ("solve", "task", "skill", "benchmark", "coding", "reason")):
            return "competence"
        if any(w in o for w in ("learn", "explore", "knowledge", "curios", "investigate", "expand")):
            return "knowledge"
        if any(w in o for w in ("error", "consistency", "ethics", "coherence", "align")):
            return "coherence"
        if any(w in o for w in ("energy", "health", "rest", "preserve", "stability", "recharge")):
            return "stability"
        return "knowledge"

    def _value_of(self, objective: str) -> dict:
        entry = self.values.get(objective)
        if entry is None:
            drive = self._classify_drive(objective)
            entry = {"utility": 0.5, "drive": drive, "attempts": 0, "updated": time.time()}
            self.values[objective] = entry
            self._prune()
        return entry

    def _prune(self):
        if len(self.values) <= MAX_VALUE_ENTRIES:
            return
        # Drop least-attempted, oldest objectives.
        ranked = sorted(self.values.items(), key=lambda kv: (kv[1]["attempts"], kv[1]["updated"]))
        for obj, _ in ranked[:len(self.values) - MAX_VALUE_ENTRIES]:
            del self.values[obj]

    def expected_value(self, objective: str, context: dict | None = None) -> float:
        """Expected value = learned utility × drive weight, modulated by context
        (low energy raises stability, high error raises coherence)."""
        entry = self._value_of(objective)
        drive = entry["drive"]
        drive_weight = self.drives.get(drive, 0.2)
        ctx = context or {}
        # Context modulation of drives — makes motivation state-dependent.
        if ctx.get("energy", 1.0) < 0.3 and drive == "stability":
            drive_weight *= 1.5
        if ctx.get("error_rate", 0.0) > 0.2 and drive == "coherence":
            drive_weight *= 1.4
        if ctx.get("curiosity", 0.0) > 0.6 and drive == "knowledge":
            drive_weight *= 1.2
        return round(entry["utility"] * drive_weight, 4)

    def choose(self, objectives: list[str], context: dict | None = None) -> dict | None:
        """Pick the highest expected-value objective. Records the choice so the
        realized reward can be credited back to it via reward()."""
        objectives = [o for o in objectives if o]
        if not objectives:
            self._last_choice = None
            return None
        scored = [(self.expected_value(o, context), o) for o in objectives]
        scored.sort(reverse=True)
        best_value, best = scored[0]
        entry = self._value_of(best)
        entry["attempts"] += 1
        choice = {
            "objective": best,
            "drive": entry["drive"],
            "expected_value": best_value,
            "alternatives": [{"objective": o, "value": v} for v, o in scored[1:4]],
            "tick": (context or {}).get("tick"),
            "time": time.time(),
        }
        self._last_choice = choice
        self.decisions.append(choice)
        if len(self.decisions) > 200:
            self.decisions = self.decisions[-200:]
        return choice

    def reward(self, realized: float, objective: str | None = None):
        """Credit realized reward to an objective's utility (TD-style update).
        If no objective is given, credits the last choice."""
        obj = objective or (self._last_choice or {}).get("objective")
        if obj is None:
            return
        entry = self._value_of(obj)
        realized = max(0.0, min(1.0, realized))
        entry["utility"] += LEARNING_RATE * (realized - entry["utility"])
        entry["utility"] = max(0.0, min(1.0, entry["utility"]))
        entry["updated"] = time.time()
        self.total_reward += realized

    # ── status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        top = sorted(self.values.items(), key=lambda kv: kv[1]["utility"], reverse=True)[:5]
        return {
            "drives": {k: round(v, 3) for k, v in self.drives.items()},
            "tracked_objectives": len(self.values),
            "total_reward": round(self.total_reward, 3),
            "last_choice": self._last_choice,
            "top_valued": [
                {"objective": o, "utility": round(e["utility"], 3),
                 "drive": e["drive"], "attempts": e["attempts"]}
                for o, e in top
            ],
        }
