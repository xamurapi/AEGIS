"""Point 2 — the eval harness: the fitness graph that replaces self-report.

Runs the benchmark through the multi-agent solver and produces a score in
[0, 1] = fraction of tasks whose answer passed verification. Tracks a score
history over time and persists it. This score is the external ground truth the
whole system optimizes toward; ``Substrate._compute_reward`` reads it.
"""
import json
import logging
from collections import deque
from pathlib import Path

from aegis._atomic import atomic_write_text
from aegis.eval.benchmark import Task, DEFAULT_BENCHMARK, all_kinds
from aegis.eval.coding import CodingTask, CODING_BENCHMARK, verify_solution
from aegis.eval.composite import CompositeTask, COMPOSITE_BENCHMARK
from aegis.eval.autocompose import AutoComposeTask, AUTOCOMPOSE_BENCHMARK
from aegis.eval.solver import MultiAgentSolver
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.evaluator")


class Evaluator:
    def __init__(self, solver: MultiAgentSolver, tasks: list[Task] | None = None,
                 coding_tasks: list[CodingTask] | None = None,
                 composite_tasks: list[CompositeTask] | None = None,
                 autocompose_tasks: list[AutoComposeTask] | None = None,
                 store_path: Path | None = None):
        self.solver = solver
        self.tasks = tasks or list(DEFAULT_BENCHMARK)
        self.coding_tasks = coding_tasks if coding_tasks is not None else list(CODING_BENCHMARK)
        self.composite_tasks = composite_tasks if composite_tasks is not None else list(COMPOSITE_BENCHMARK)
        self.autocompose_tasks = (autocompose_tasks if autocompose_tasks is not None
                                  else list(AUTOCOMPOSE_BENCHMARK))
        self._store_path = store_path
        self.history: deque = deque(maxlen=200)
        self.last_score: float | None = None
        self.last_report: dict = {}
        self.total_runs = 0
        # What the last full benchmark found. Read by the action registry's
        # preconditions: deciding what to do must not stop to re-measure, and
        # `failing_kinds()` re-runs the whole sandboxed benchmark — seconds of
        # subprocess work that turned a 30 ms decision into a 3 s one.
        self._failing_kinds: list[str] = []
        self._unsolved_coding: list[CodingTask] = []
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
            atomic_write_text(self._store_path, json.dumps({
                "last_score": self.last_score,
                "total_runs": self.total_runs,
                "history": list(self.history),
            }))
        except Exception:
            logger.warning("Failed to save eval history", exc_info=True)

    def coding_solved(self, task: CodingTask) -> bool:
        """A coding task is solved if any stored solution passes ALL hidden tests."""
        for skill in self.solver.library.for_kind(task.kind_key()):
            if verify_solution(skill.code, task, timeout=self.solver.timeout)["solved"]:
                self.solver.library.record(skill.name, True)
                return True
        return False

    def run(self, only_kinds: list[str] | None = None, record: bool = True) -> dict:
        """Run the full benchmark (skills + coding + composite). Returns a report.

        ``only_kinds`` restricts to skill-task kinds (used by the per-kind gate)
        and skips coding/composite families."""
        subset = only_kinds is not None
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

        coding_passed = composite_passed = auto_passed = 0
        coding_total = composite_total = auto_total = 0
        if not subset:
            coding_total = len(self.coding_tasks)
            unsolved = [t for t in self.coding_tasks if not self.coding_solved(t)]
            coding_passed = coding_total - len(unsolved)
            self._unsolved_coding = unsolved
            composite_total = len(self.composite_tasks)
            composite_passed = sum(1 for t in self.composite_tasks if self.solver.solve_composite(t).solved)
            auto_total = len(self.autocompose_tasks)
            auto_passed = sum(
                1 for t in self.autocompose_tasks
                if self.solver.auto_compose(t.start, t.target, t.kinds, t.max_depth) is not None
            )

        grand_passed = passed + coding_passed + composite_passed + auto_passed
        grand_total = total + coding_total + composite_total + auto_total
        score = grand_passed / grand_total if grand_total else 0.0
        report = {
            "timestamp": CLOCK.now(),
            "score": round(score, 4),
            "passed": grand_passed,
            "total": grand_total,
            "skills": {"passed": passed, "total": total},
            "coding": {"passed": coding_passed, "total": coding_total},
            "composite": {"passed": composite_passed, "total": composite_total},
            "autocompose": {"passed": auto_passed, "total": auto_total},
            "per_kind": {k: {"passed": v[0], "total": v[1]} for k, v in per_kind.items()},
        }
        if not subset:
            # Derived from the run that just happened, so a later decision can
            # read it without paying for the measurement again.
            self._failing_kinds = sorted(
                kind for kind, counts in per_kind.items()
                if counts[0] < counts[1])
        if record and not subset:
            self.last_score = report["score"]
            self.last_report = report
            self.total_runs += 1
            self.history.append({"t": report["timestamp"], "score": report["score"]})
            self._save()
        return report

    # ── what the last measurement found ──────────────────────────────
    # Cheap reads for the action registry. They answer "what did the most
    # recent benchmark say", which is the right question for a decision;
    # `failing_kinds()` and `unsolved_coding()` answer "what is true right
    # now" and cost a full sandboxed run to do it.

    def failing_kinds_cached(self) -> list[str]:
        return list(self._failing_kinds)

    def unsolved_coding_cached(self) -> list[CodingTask]:
        return list(self._unsolved_coding)

    def has_measured(self) -> bool:
        """Whether a full benchmark has run at all this session."""
        return self.total_runs > 0 or bool(self.last_report)

    def unsolved_coding(self) -> list[CodingTask]:
        return [t for t in self.coding_tasks if not self.coding_solved(t)]

    def kind_pass_rate(self, kind: str) -> float:
        """Pass-rate for a single kind — used by the skill-acceptance gate."""
        report = self.run(only_kinds=[kind], record=False)
        return report["score"]

    # ── the three-way split (spec M9.3) ──────────────────────────────

    def splits(self) -> dict[str, list[Task]]:
        """``train`` / ``valid`` / ``test`` over this evaluator's task set.

        Recomputed rather than cached: the task set grows when generators are
        added, and a cached split would keep describing the set it was built
        from. The assignment is a pure function of the task ids, so recomputing
        is cheap and always agrees with itself.
        """
        from aegis.eval.benchmark import three_way_split

        return three_way_split(self.tasks)

    def split_sizes(self) -> dict[str, int]:
        return {name: len(tasks) for name, tasks in self.splits().items()}

    def score_on_split(self, split: str) -> float:
        """Pass-rate on one split.

        Selection reads ``valid``; ``test`` confirms a champion once and is
        never selected on (§M5.5). Keeping them separate methods rather than one
        flag is deliberate — the distinction is the whole protection against
        overfitting, and it should be visible at the call site.
        """
        return self.pass_rate_on(self.splits().get(str(split), []))

    def valid_test_gap(self) -> float:
        """How much better the selected-on split looks than the untouched one.

        The overfitting indicator of §M5.8. A gap that grows while `valid`
        improves is evolution learning the validation set rather than the task.
        """
        return round(self.score_on_split("valid") - self.score_on_split("test"), 6)

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

    def history_csv(self) -> str:
        """Export the full fitness history as CSV text (run,timestamp,score)."""
        lines = ["run,timestamp,score"]
        for i, h in enumerate(self.history, 1):
            lines.append(f"{i},{h.get('t', '')},{h.get('score', '')}")
        return "\n".join(lines) + "\n"

    def status(self) -> dict:
        return {
            "last_score": self.last_score,
            "total_runs": self.total_runs,
            "last_report": self.last_report,
            "history_tail": list(self.history)[-20:],
            "benchmark_size": len(self.tasks),
            "splits": self.split_sizes(),
        }
