"""Measure mean phase durations in a clean process, and print them as JSON.

Run as a subprocess by ``tests/test_phase_budget_measured.py``. It lives in a
file of its own rather than inside the test because a performance measurement
has to happen in a process that nothing else has been running in: worker pools,
detached tasks and warmed caches left by an earlier test all compete for the
same CPU, and a budget measured against that is a measurement of the suite.
"""
import asyncio
import importlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from conftest import _STATE_DIRS                       # noqa: E402
from aegis.layers.substrate import Substrate           # noqa: E402


def _cheap(substrate):
    """Remove exactly what §3.4 excludes, and nothing else."""
    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.llm.enabled = False
    substrate.sensors.read_all = lambda: {"pinned": True}
    substrate.evaluator.run = lambda *a, **k: {"score": 0.5, "per_kind": {}}
    substrate.environment.step = lambda *a, **k: {
        "reward": 0.25, "solved": True, "task": "canned", "kind": "calc"}
    substrate.evolution.run_generation = lambda *a, **k: {"generation": 1}
    return substrate


def main(ticks: int) -> None:
    root = Path(tempfile.mkdtemp())
    for module_name, constant, subdir in _STATE_DIRS:
        module = importlib.import_module(module_name)
        if not hasattr(module, constant):
            continue
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        setattr(module, constant, target)

    import aegis.layers.substrate as substrate_mod
    from aegis.layers.state_backup import StateBackup

    substrate_mod.StateBackup = lambda *a, **k: StateBackup(
        backup_dir=root / "backups")

    substrate = _cheap(Substrate())
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    async def _drive():
        for _ in range(ticks):
            await substrate.tick()
            for phase, duration in substrate._ctx.durations_ms.items():
                totals[phase] = totals.get(phase, 0.0) + duration
                counts[phase] = counts.get(phase, 0) + 1
        await substrate.cancel_background_tasks()

    asyncio.run(_drive())
    print(json.dumps({phase: totals[phase] / counts[phase] for phase in totals}))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
