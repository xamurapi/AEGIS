"""System 5: Real-world Feedback Loop — learning from actual outcomes (FB-001..FB-005).

Closes the experience loop the spec calls for:

    situation → AEGIS decision → real result → evaluation →
    cause of success/failure → new experience

Each recorded experience is a structured row (not raw text), so the dataset
that feeds LoRA/skill-synthesis contains WHY an outcome happened, not just what
was said. Also emits reward signals back to GoalIntelligence and cause→effect
observations to the WorldModel, turning every action into training signal.
Deterministic; LLM only used (optionally, by the Substrate) to phrase the cause.
"""
import json
import time
import logging
from pathlib import Path

from aegis.config import FEEDBACK_DIR

logger = logging.getLogger("aegis.feedback_loop")

MAX_OPEN = 200          # cap on outstanding (unresolved) experiences
MAX_JSONL_ROWS = 5000   # cap on the persisted experience log


class FeedbackLoop:
    def __init__(self, store_path: Path | None = None):
        # Situations awaiting their real result, keyed by experience id.
        self._open: dict[str, dict] = {}
        self._seq = 0
        self.resolved = 0
        self.successes = 0
        self.failures = 0
        self.recent: list[dict] = []
        self._store_path = store_path or (FEEDBACK_DIR / "experiences.jsonl")
        self._load()

    def _load(self):
        """Rebuild counters from the persisted JSONL log so status()/success_rate
        are correct after a restart, and continue the id sequence past the last
        stored experience (otherwise new exp_ids would collide with old rows).
        """
        if not self._store_path.exists():
            return
        try:
            with self._store_path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(ln) for ln in fh if ln.strip()]
        except Exception:
            logger.warning("Failed to load experiences from %s", self._store_path, exc_info=True)
            return
        self.resolved = len(rows)
        self.successes = sum(1 for r in rows if r.get("success"))
        self.failures = self.resolved - self.successes
        max_seq = 0
        for r in rows:
            eid = str(r.get("id", ""))
            if eid.startswith("exp_"):
                try:
                    max_seq = max(max_seq, int(eid.split("_")[1]))
                except (ValueError, IndexError):
                    pass
        self._seq = max_seq
        self.recent = rows[-50:]

    # ── recording ────────────────────────────────────────────────────

    def record_situation(self, situation: str, decision: str, context: dict | None = None) -> str:
        """Open a new experience: a decision taken in a situation, awaiting its
        real-world result. Returns an id used later to close the loop."""
        self._seq += 1
        exp_id = f"exp_{self._seq:08d}"
        self._open[exp_id] = {
            "id": exp_id,
            "situation": situation[:300],
            "decision": decision[:200],
            "context": {k: context[k] for k in list(context or {})[:10]},
            "opened": time.time(),
        }
        # Bound outstanding experiences — drop the oldest unresolved ones.
        if len(self._open) > MAX_OPEN:
            for stale in sorted(self._open, key=lambda k: self._open[k]["opened"])[:len(self._open) - MAX_OPEN]:
                del self._open[stale]
        return exp_id

    def record_result(self, exp_id: str, success: bool, metric: float,
                      cause: str = "", expected: str = "") -> dict | None:
        """Close the loop: attach the real result to a prior situation, classify
        the cause of success/failure, persist it as a training row, and return
        the completed experience (for reward propagation by the caller)."""
        situation = self._open.pop(exp_id, None)
        if situation is None:
            return None
        experience = {
            **situation,
            "success": bool(success),
            "metric": round(float(metric), 4),
            "cause": (cause or self._infer_cause(success, metric))[:300],
            "expected": expected[:200],
            "resolved": time.time(),
            "latency_s": round(time.time() - situation["opened"], 2),
        }
        self.resolved += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self._append(experience)
        self.recent.append(experience)
        if len(self.recent) > 50:
            self.recent = self.recent[-50:]
        return experience

    @staticmethod
    def _infer_cause(success: bool, metric: float) -> str:
        if success and metric >= 0.7:
            return "decision matched situation; high verified metric"
        if success:
            return "decision succeeded but with a weak margin"
        if metric <= 0.2:
            return "decision failed hard; likely wrong action for the situation"
        return "decision failed; partial progress, action under-fit the situation"

    def _append(self, experience: dict):
        """Append one experience row to the JSONL log, keeping it bounded."""
        try:
            with self._store_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(experience, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("Failed to append experience to %s", self._store_path, exc_info=True)
            return
        self._truncate_if_needed()

    def _truncate_if_needed(self):
        """Bound the on-disk log so it cannot grow without limit."""
        try:
            if not self._store_path.exists():
                return
            # Count cheaply; only rewrite when clearly over budget (2× cap) to
            # avoid rewriting the whole file every append.
            with self._store_path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
            if len(lines) <= MAX_JSONL_ROWS * 2:
                return
            keep = lines[-MAX_JSONL_ROWS:]
            tmp = self._store_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(keep), encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to truncate experience log %s", self._store_path, exc_info=True)

    def save(self):
        """Experiences are persisted incrementally on record_result (append to
        JSONL), so there is nothing to flush here. Provided for a uniform
        save() interface with the other higher-order systems."""
        return

    # ── training-set export ──────────────────────────────────────────

    def export_examples(self, limit: int = 200) -> list[dict]:
        """Return recent experiences as supervised rows for dataset building:
        the situation+decision as prompt, the outcome+cause as the lesson."""
        rows = []
        try:
            if self._store_path.exists():
                with self._store_path.open("r", encoding="utf-8") as fh:
                    for line in fh.readlines()[-limit:]:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
        except Exception:
            logger.warning("Failed to read experiences for export", exc_info=True)
        return [{
            "prompt": f"Situation: {r['situation']}\nDecision: {r['decision']}",
            "completion": f"Result: {'success' if r['success'] else 'failure'} "
                          f"(metric={r['metric']}). Cause: {r['cause']}",
            "success": r["success"],
            "metric": r["metric"],
        } for r in rows]

    # ── status ───────────────────────────────────────────────────────

    def success_rate(self) -> float:
        return round(self.successes / self.resolved, 3) if self.resolved else 0.0

    def status(self) -> dict:
        return {
            "open_experiences": len(self._open),
            "resolved": self.resolved,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate(),
            "recent": [
                {"situation": r["situation"][:60], "success": r["success"],
                 "metric": r["metric"], "cause": r["cause"][:80]}
                for r in self.recent[-5:]
            ],
        }
