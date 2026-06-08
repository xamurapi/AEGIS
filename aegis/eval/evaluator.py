"""Point 2 — the eval harness: the fitness graph that replaces self-report.

Runs the benchmark through the multi-agent solver and produces a score in
[0, 1] = fraction of tasks whose answer passed verification. Tracks a score
history over time and persists it. This score is the external ground truth the
whole system optimizes toward; ``Substrate._compute_reward`` reads it.
"""
import json
import time
import logging
from collections import deque
from pathlib import Path

from aegis.eval.benchmark import Task, DEFAULT_BENCHMARK, all_kinds
from aegis.eval.solver import MultiAgentSolver

logger = logging.getLogger("aegis.evaluator")


class Evaluator:
    def __init__(self, solver: MultiAgentSolver, tasks: list[Task] | None = None,
                 store_path: Path | None = None):
        self.solver = solver
        self.tasks = tasks or list(DEFAULT_BENCHMARK)
        self._store_path = store_path
        self.history: deque = deque(maxlen=200)
        self.last_score: float | None = None
        self.last_report: dict = {}
        self.total_runs = 0
        self._load()

    def _load(self):
        if not self._store_path or not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self.last_score = data.get("last_score")
            self.total_runs = data.get("total_runs", 0)
            for h in data.get("history", [])[-200:]:
                self.history.append(h)
        except Exception:
            logger.warning("Failed to load eval history", exc_info=True)

    def _save(self):
        if not self._store_path:
            return
        try:
            self._store_path.write_text(json.dumps({
                "last_score": self.last_score,
                "total_runs": self.total_runs,
                "history": list(self.history),
            }), encoding="utf-8")
        except Exception:
            logger.warning("Failed to save eval history", exc_info=True)

    def run(self, only_kinds: list[str] | None = None, record: bool = True) -> dict:
        """Run the benchmark (optionally a subset of kinds). Returns a report."""
        tasks = [t for t in self.tasks if (only_kinds is None or t.kind in only_kinds)]
        per_kind: dict[str, list[int]] = {}
        passed = 0
        for t in tasks:
            res = self.solver.solve(t)
            per_kind.setdefault(t.kind, [0, 0])
            per_kind[t.kind][1] += 1
            if res.solved:
                passed += 1
                per_kind[t.kind][0] += 1
        total = len(tasks)
        score = passed / total if total else 0.0
        report = {
            "timestamp": time.time(),
            "score": round(score, 4),
            "passed": passed,
            "total": total,
            "per_kind": {k: {"passed": v[0], "total": v[1]} for k, v in per_kind.items()},
        }
        if record and only_kinds is None:
            self.last_score = report["score"]
            self.last_report = report
            self.total_runs += 1
            self.history.append({"t": report["timestamp"], "score": report["score"]})
            self._save()
        return report

    def kind_pass_rate(self, kind: str) -> float:
        """Pass-rate for a single kind — used by the skill-acceptance gate."""
        report = self.run(only_kinds=[kind], record=False)
        return report["score"]

    def pass_rate_on(self, tasks: list[Task]) -> float:
        """Pass-rate over an explicit task list (e.g. a held-out split).

        Used by the synthesis gate to measure GENERALIZATION: the candidate
        skill is scored on tasks it was not shown during proposal."""
        if not tasks:
            return 0.0
        passed = sum(1 for t in tasks if self.solver.solve(t).solved)
        return passed / len(tasks)

    def failing_kinds(self) -> list[str]:
        """Kinds not at 100% pass — synthesis targets."""
        failing = []
        for k in all_kinds(self.tasks):
            if self.kind_pass_rate(k) < 1.0:
                failing.append(k)
        return failing

    def status(self) -> dict:
        return {
            "last_score": self.last_score,
            "total_runs": self.total_runs,
            "last_report": self.last_report,
            "history_tail": list(self.history)[-20:],
            "benchmark_size": len(self.tasks),
        }
