"""Point 5 — grounding: a task ENVIRONMENT the agent acts in, with consequences.

Where the evaluator is a periodic batch score, the environment is live, episodic
interaction: each ``step`` presents the next task, the agent acts (solver runs
skills), and the environment returns a REAL reward (1.0 solved / 0.0 not). That
reward flows into emotions and goals, so the agent's internal state is now
driven by verifiable outcomes instead of self-report. Task order is round-robin
(deterministic), consistent with the deterministic core cycle.
"""
import time
from collections import deque

from aegis.eval.benchmark import Task, DEFAULT_BENCHMARK
from aegis.eval.solver import MultiAgentSolver


class TaskEnvironment:
    def __init__(self, solver: MultiAgentSolver, tasks: list[Task] | None = None,
                 window: int = 20):
        self.solver = solver
        # `tasks or ...` would silently replace a deliberately-empty [] with the
        # default set; distinguish "not provided" (None) from "empty".
        self.tasks = tasks if tasks is not None else list(DEFAULT_BENCHMARK)
        self._idx = 0
        self.rewards: deque = deque(maxlen=window)
        self.total_steps = 0
        self.total_solved = 0
        self.last_step: dict = {}

    def step(self) -> dict:
        """Present the next task, act, return the outcome with a real reward."""
        if not self.tasks:
            return {"reward": 0.0, "solved": False, "task": None}
        task = self.tasks[self._idx % len(self.tasks)]
        self._idx += 1

        res = self.solver.solve(task)
        reward = 1.0 if res.solved else 0.0
        self.rewards.append(reward)
        self.total_steps += 1
        self.total_solved += int(res.solved)
        self.last_step = {
            "time": time.time(),
            "task": task.id,
            "kind": task.kind,
            "solved": res.solved,
            "reward": reward,
            "winning_skill": res.winning_skill,
            "candidates": res.candidates,
        }
        return self.last_step

    def rolling_reward(self) -> float:
        """Mean reward over the recent window — the live grounding signal."""
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def status(self) -> dict:
        return {
            "total_steps": self.total_steps,
            "total_solved": self.total_solved,
            "rolling_reward": round(self.rolling_reward(), 3),
            "lifetime_solve_rate": round(self.total_solved / max(1, self.total_steps), 3),
            "last_step": self.last_step,
        }
