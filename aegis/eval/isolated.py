"""Evaluating a configuration without touching the live one (spec M5.5, M9.1).

Evolution has to score ten variants, and an arena has to score a candidate
strategy, and neither may disturb the system doing the scoring. The obvious
implementation — set the parameter, run the benchmark, set it back — is wrong in
a way that is hard to see: the benchmark *records* which skills succeeded, so
the act of measuring a variant changes the success counters the live solver
ranks by. Ten variants later the library has been trained by an experiment
nobody meant to be training.

So an evaluation is a *request*: a plain dictionary carrying a snapshot of the
skills, the tasks, and the settings to evaluate under. It is picklable, it is
run by a module-level function, and it can therefore go to a pool worker. What
comes back is a report. Nothing in the live system is reachable from inside.

The request is also the isolation boundary for the pool: if it cannot be
expressed as data, it does not belong in another process.
"""
from __future__ import annotations

import logging

from aegis.eval.benchmark import Task
from aegis.eval.skill_library import Skill, SkillLibrary
from aegis.eval.solver import MultiAgentSolver

logger = logging.getLogger("aegis.eval.isolated")


def task_to_dict(task: Task) -> dict:
    return {"id": task.id, "kind": task.kind, "prompt": task.prompt,
            "payload": dict(task.payload), "expected": task.expected}


def task_from_dict(data: dict) -> Task:
    return Task(id=str(data["id"]), kind=str(data["kind"]),
                prompt=str(data.get("prompt", "")),
                payload=dict(data.get("payload") or {}),
                expected=data.get("expected"))


def export_skills(library: SkillLibrary) -> list[dict]:
    """Everything needed to rebuild the library elsewhere, code included.

    Distinct from :meth:`SkillLibrary.snapshot`, which hashes the code because
    it is for digests and comparisons. A worker needs the code itself.
    """
    with library._lock:                      # noqa: SLF001 — same package
        return sorted((skill.to_dict() for skill in library.skills.values()),
                      key=lambda row: row["name"])


def library_from_export(rows) -> SkillLibrary:
    """Rebuild a library from exported rows — no disk, no seeding.

    ``seed=False`` matters: seeding would silently add the built-in skills to a
    variant that was supposed to be evaluated without them, and every ablation
    would measure the same library.
    """
    library = SkillLibrary(store_path=None, seed=False)
    for row in rows or []:
        # A row that is not a mapping at all is the shape a torn file or an
        # older schema produces. It costs itself, not the whole library — an
        # evaluation that refused to start because one row was odd would take
        # the generation down with it.
        if not isinstance(row, dict):
            logger.debug("Skipping a non-mapping exported skill row")
            continue
        row = dict(row)
        row.pop("success_rate", None)
        try:
            library.skills[str(row["name"])] = Skill(**row)
        except (KeyError, TypeError):
            logger.debug("Skipping an unusable exported skill row", exc_info=True)
    return library


def make_request(library: SkillLibrary, tasks, *, timeout: float = 3.0,
                 solver_order: str = "by_success", label: str = "") -> dict:
    """Package an evaluation so it can cross a process boundary."""
    return {
        "label": str(label),
        "skills": export_skills(library),
        "tasks": [task_to_dict(task) for task in tasks],
        "timeout": float(timeout),
        "solver_order": str(solver_order),
    }


def run_request(request: dict) -> dict:
    """Score one request. Module-level and picklable, so a pool can run it.

    Returns counts rather than objects: what crosses back is data, and a report
    the caller has to reconstruct objects from is a report that can disagree
    with itself.
    """
    request = dict(request or {})
    tasks = [task_from_dict(row) for row in request.get("tasks") or []]
    library = library_from_export(request.get("skills"))
    solver = MultiAgentSolver(library, timeout=float(request.get("timeout", 3.0)))
    solver.order = str(request.get("solver_order", "by_success"))

    per_kind: dict[str, list[int]] = {}
    solved_ids: list[str] = []
    for task in tasks:
        result = solver.solve(task)
        counts = per_kind.setdefault(task.kind, [0, 0])
        counts[1] += 1
        if result.solved:
            counts[0] += 1
            solved_ids.append(task.id)

    passed, total = len(solved_ids), len(tasks)
    return {
        "label": request.get("label", ""),
        "passed": passed,
        "total": total,
        "score": round(passed / total, 6) if total else 0.0,
        "per_kind": {kind: {"passed": counts[0], "total": counts[1]}
                     for kind, counts in sorted(per_kind.items())},
        "solved": sorted(solved_ids),
        # The counters the run produced, so a caller that WANTS the learning can
        # fold it back deliberately instead of getting it by accident.
        "skills": export_skills(library),
    }
