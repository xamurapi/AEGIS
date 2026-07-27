"""Point 6 — multi-agent solving with a VERIFIER.

For a task, every skill that claims the task's kind is an independent "agent":
each runs in its own sandbox and proposes an answer. The task's own verifier
(ground truth) acts as the critic and selects the first answer that passes. This
is genuine division of labor plus verification — no single skill is trusted, and
the selection is objective, not a vote of confidence.
"""
from collections import deque
from dataclasses import dataclass

from aegis.eval.benchmark import Task
from aegis.eval.skill_library import SkillLibrary
from aegis.eval.sandbox import run_skill
from aegis.clock import CLOCK


@dataclass
class SolveResult:
    task_id: str
    kind: str
    solved: bool
    answer: object = None
    winning_skill: str | None = None
    candidates: int = 0
    elapsed_ms: float = 0.0
    # The last answer any skill produced, even when none verified. `answer` is
    # deliberately None on failure (nothing was accepted), but a repair loop
    # needs to know WHAT was answered wrongly, not just that something was.
    last_answer: object = None
    last_skill: str | None = None


class MultiAgentSolver:
    def __init__(self, library: SkillLibrary, timeout: float = 3.0):
        self.library = library
        self.timeout = timeout

    def solve(self, task: Task) -> SolveResult:
        t0 = CLOCK.now()
        skills = self.library.for_kind(task.kind)
        candidates = 0
        last_answer = None
        last_skill = None
        for skill in skills:
            candidates += 1
            out = run_skill(skill.code, skill.func, task.payload, timeout=self.timeout)
            ok = bool(out.get("ok")) and task.verify(out.get("result"))
            self.library.record(skill.name, ok)
            if out.get("ok"):
                last_answer, last_skill = out.get("result"), skill.name
            if ok:
                return SolveResult(
                    task.id, task.kind, True, out.get("result"),
                    skill.name, candidates, round((CLOCK.now() - t0) * 1000, 1),
                    last_answer=out.get("result"), last_skill=skill.name,
                )
        return SolveResult(
            task.id, task.kind, False, None, None, candidates,
            round((CLOCK.now() - t0) * 1000, 1),
            last_answer=last_answer, last_skill=last_skill,
        )

    def solve_composite(self, task) -> SolveResult:
        """Solve a composite task by threading a string through its pipeline of
        primitive skills (each kind takes {"s": str} and returns a string)."""
        t0 = CLOCK.now()
        current = task.payload.get("s")
        steps = 0
        for kind in task.pipeline:
            skills = self.library.for_kind(kind)
            if not skills:
                return SolveResult(task.id, "compose", False, None, None, steps,
                                   round((CLOCK.now() - t0) * 1000, 1))
            produced = None
            for skill in skills:
                out = run_skill(skill.code, skill.func, {"s": current}, timeout=self.timeout)
                if out.get("ok"):
                    produced = out["result"]
                    break
            if produced is None:
                return SolveResult(task.id, "compose", False, None, None, steps,
                                   round((CLOCK.now() - t0) * 1000, 1))
            current = produced
            steps += 1
        solved = task.verify(current)
        return SolveResult(task.id, "compose", solved, current if solved else None,
                           "+".join(task.pipeline), steps, round((CLOCK.now() - t0) * 1000, 1))

    def auto_compose(self, start: str, target: str, kinds, max_depth: int = 3):
        """BFS search for a pipeline of transform skills mapping start -> target.

        Returns the list of kinds applied (the discovered pipeline) or None.
        Shortest pipeline wins (BFS); cycles are pruned via a visited set."""
        if start == target:
            return []
        seen = {start}
        queue: deque = deque([(start, [])])
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for kind in kinds:
                produced = None
                for skill in self.library.for_kind(kind):
                    out = run_skill(skill.code, skill.func, {"s": current}, timeout=self.timeout)
                    if out.get("ok"):
                        produced = out["result"]
                        break
                if not isinstance(produced, str):
                    continue
                new_path = path + [kind]
                if produced == target:
                    return new_path
                if produced not in seen:
                    seen.add(produced)
                    queue.append((produced, new_path))
        return None
