"""Point 6 — multi-agent solving with a VERIFIER.

For a task, every skill that claims the task's kind is an independent "agent":
each runs in its own sandbox and proposes an answer. The task's own verifier
(ground truth) acts as the critic and selects the first answer that passes. This
is genuine division of labor plus verification — no single skill is trusted, and
the selection is objective, not a vote of confidence.
"""
import time
from dataclasses import dataclass

from aegis.eval.benchmark import Task
from aegis.eval.skill_library import SkillLibrary
from aegis.eval.sandbox import run_skill


@dataclass
class SolveResult:
    task_id: str
    kind: str
    solved: bool
    answer: object = None
    winning_skill: str | None = None
    candidates: int = 0
    elapsed_ms: float = 0.0


class MultiAgentSolver:
    def __init__(self, library: SkillLibrary, timeout: float = 3.0):
        self.library = library
        self.timeout = timeout

    def solve(self, task: Task) -> SolveResult:
        t0 = time.time()
        skills = self.library.for_kind(task.kind)
        candidates = 0
        for skill in skills:
            candidates += 1
            out = run_skill(skill.code, skill.func, task.payload, timeout=self.timeout)
            ok = bool(out.get("ok")) and task.verify(out.get("result"))
            self.library.record(skill.name, ok)
            if ok:
                return SolveResult(
                    task.id, task.kind, True, out.get("result"),
                    skill.name, candidates, round((time.time() - t0) * 1000, 1),
                )
        return SolveResult(
            task.id, task.kind, False, None, None, candidates,
            round((time.time() - t0) * 1000, 1),
        )
