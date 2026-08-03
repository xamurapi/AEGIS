#!/usr/bin/env python3
"""The 24-hour soak of §VII.5.

Four claims, and the run either supports all of them or it does not:

* **No memory leak.** RSS must grow no more than ``--max-growth`` (5% by
  default) across the *last half* of the run. The first half is excluded on
  purpose: every one of these contours fills a bounded structure on the way up —
  the world model learns states, the graph gains nodes, telemetry accumulates
  series — and measuring from a cold start would call that growth a leak. What
  a leak looks like is growth that has not stopped once the structures are full.
* **Nothing is left running.** No child processes and no pending asyncio tasks
  after shutdown. A pool worker that outlives its substrate is a leak of a kind
  no memory graph shows.
* **Every state file is valid.** Read back with the real loaders at the end,
  including after an interrupt (``--interrupt-at``), which kills the run at a
  chosen fraction and checks that what is on disk still loads.
* **It keeps ticking.** Ticks continue at a sane rate for the whole run and the
  phase budgets of §3.4 hold at the end as well as at the start — a system that
  survives by slowing to a crawl has not survived.

Usage::

    python scripts/soak.py                        # the full 24 hours
    python scripts/soak.py --hours 0.5            # a shakedown
    python scripts/soak.py --interrupt-at 0.5     # kill halfway, then verify
    python scripts/soak.py --hours 2 --report soak.json

The run is deliberately survivable: it writes a JSON report as it goes, so a
machine that reboots at hour twenty does not cost the whole measurement.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:                                   # pragma: no cover
    HAS_PSUTIL = False

#: How often RSS and tick progress are sampled, in seconds.
SAMPLE_SECONDS = 60.0

#: Fraction of the run used as the baseline for the leak test. The spec asks
#: about "the last 12 hours" of 24, which is the second half.
LEAK_WINDOW = 0.5

#: The spec's bar.
DEFAULT_MAX_GROWTH = 0.05

#: Below this many hours the leak and tick-rate verdicts are withheld.
#:
#: Not timidity — the thresholds are meaningless before the bounded structures
#: are full. The world model is still meeting states, the graph is still
#: gaining nodes, the telemetry series are still short: on a six-minute run the
#: "second half" is still the warm-up, and a shakedown reported +5.5% growth
#: that says nothing about a day. The spec asks about the last twelve hours of
#: twenty-four, by which time the growth that is going to stop has stopped.
MIN_JUDGED_HOURS = 2.0


def _rss_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def _children() -> list[int]:
    if not HAS_PSUTIL:
        return []
    try:
        return [child.pid for child in
                psutil.Process(os.getpid()).children(recursive=True)]
    except Exception:
        return []


def _pending_tasks() -> int:
    try:
        return len([task for task in asyncio.all_tasks() if not task.done()])
    except RuntimeError:
        return 0


def _state_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix in (".json", ".jsonl"))


def verify_state(root: Path) -> dict:
    """Read every state file back with the loaders that own it.

    A file that parses is not the same as a file the system can use, so the
    versioned stores go through ``read_store`` — which is what applies the
    migration and what would report an unreadable payload as empty state.
    """
    from aegis.store.migrations import read_store

    report = {"checked": 0, "unreadable": [], "empty_after_load": []}
    for path in _state_files(root):
        report["checked"] += 1
        if path.suffix == ".jsonl":
            # Append-only logs: a torn *last* line is expected after a kill and
            # is not a fault. Anything torn earlier is.
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines[:-1]):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    report["unreadable"].append(f"{path}:{index + 1}")
            continue
        try:
            loaded = read_store(path)
        except Exception as error:                    # pragma: no cover
            report["unreadable"].append(f"{path}: {error}")
            continue
        if set(loaded) <= {"schema_version"} and path.stat().st_size > 40:
            # It loaded as empty although the file has content: the store was
            # unreadable and the loader degraded, which is exactly the silent
            # failure §VII.5 is asking about.
            report["empty_after_load"].append(str(path))
    return report


def _build(root: Path):
    """A substrate whose stores live under ``root`` and which does no network."""
    import importlib

    import aegis.config as cfg

    for attribute in ("MEMORY_DIR", "TELEMETRY_DIR", "WORLD_MODEL_DIR",
                      "COGNITIVE_GRAPH_DIR", "EVOLUTION_DIR", "GOAL_INTEL_DIR",
                      "FEEDBACK_DIR", "CHECKPOINTS_DIR", "EVAL_DIR",
                      "POLICY_DIR", "MOTIVATION_DIR", "REASONING_DIR",
                      "DISCOVERY_DIR", "META_DIR", "CORTEX_DIR",
                      "CODE_BACKUPS_DIR", "WEIGHT_DATASETS_DIR"):
        if hasattr(cfg, attribute):
            target = root / attribute.lower().replace("_dir", "")
            target.mkdir(parents=True, exist_ok=True)
            setattr(cfg, attribute, target)

    for module_name, constant in (
            ("aegis.layers.memory", "MEMORY_DIR"),
            ("aegis.telemetry.store", "TELEMETRY_DIR"),
            ("aegis.layers.world_model", "WORLD_MODEL_DIR"),
            ("aegis.layers.cognitive_graph", "COGNITIVE_GRAPH_DIR"),
            ("aegis.layers.evolution_engine", "EVOLUTION_DIR"),
            ("aegis.layers.goal_intelligence", "GOAL_INTEL_DIR"),
            ("aegis.layers.feedback_loop", "FEEDBACK_DIR"),
            ("aegis.layers.substrate", "CHECKPOINTS_DIR"),
            ("aegis.layers.substrate", "EVAL_DIR"),
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, constant):
            setattr(module, constant, getattr(
                importlib.import_module("aegis.config"), constant))

    from aegis.layers.state_backup import StateBackup
    import aegis.layers.substrate as substrate_mod
    substrate_mod.StateBackup = lambda *a, **k: StateBackup(
        backup_dir=root / "backups")

    from aegis.layers.substrate import Substrate

    substrate = Substrate()

    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    # The network is the one thing a soak must not depend on: a run that failed
    # at hour nineteen because a host was unreachable would say nothing about
    # memory.
    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning

    # Weight training is off, and this is the one exclusion that needs
    # defending. It downloads a model and holds it resident: a shakedown run
    # measured RSS going from 94 MB to 4.4 GB in two minutes the first time the
    # contour fired. That is not a leak — it is a model being loaded — but a
    # soak cannot tell the two apart, and worse, four gigabytes of legitimate
    # growth would hide a real leak of any size underneath it.
    #
    # The LoRA contour has its own tests and its own acceptance; what this run
    # is measuring is whether the *cognitive cycle* leaks over a day, and the
    # cycle is what stays.
    async def _no_training():
        return {"success": False, "reason": "disabled for the soak"}

    substrate._weight_training_cycle = _no_training
    return substrate


async def _soak(substrate, seconds: float, interrupt_at: float | None,
                report_path: Path, root: Path, tick_interval: float) -> dict:
    import aegis.config as cfg

    started = time.time()
    samples: list[dict] = []
    last_sample = 0.0
    interrupted = False

    while True:
        elapsed = time.time() - started
        if elapsed >= seconds:
            break
        if interrupt_at is not None and elapsed >= seconds * interrupt_at:
            interrupted = True
            break

        await substrate.tick()
        if tick_interval:
            await asyncio.sleep(tick_interval)

        if elapsed - last_sample >= SAMPLE_SECONDS:
            last_sample = elapsed
            samples.append({
                "elapsed": round(elapsed, 1),
                "tick": substrate.tick_count,
                "rss_mb": round(_rss_mb(), 2),
                "children": len(_children()),
                "pending_tasks": _pending_tasks(),
                "avg_tick_ms": round(
                    sum(substrate.cycle_times) * 1000
                    / max(1, len(substrate.cycle_times)), 2),
            })
            _write(report_path, {"status": "running", "samples": samples})
            print(f"  {elapsed / 3600:5.2f}h  tick {substrate.tick_count:7d}  "
                  f"rss {samples[-1]['rss_mb']:8.1f} MB  "
                  f"tick {samples[-1]['avg_tick_ms']:6.2f} ms  "
                  f"children {samples[-1]['children']}", flush=True)

    if interrupted:
        # Killed at a chosen point, exactly as a SIGTERM would. Nothing is
        # cleaned up on purpose: what is on disk now is what a real kill leaves.
        print(f"\ninterrupted at {(time.time() - started) / 3600:.2f}h "
              "— leaving state exactly as a kill would")
        return {"samples": samples, "interrupted": True,
                "children_after": _children(), "pending_after": _pending_tasks()}

    await substrate.cancel_background_tasks()
    substrate._save_checkpoint()
    substrate.telemetry.flush()
    # Give the operating system a moment to reap what was just shut down.
    await asyncio.sleep(2.0)
    return {"samples": samples, "interrupted": False,
            "children_after": _children(), "pending_after": _pending_tasks()}


def _write(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    except Exception:
        pass


def judge(result: dict, state: dict, max_growth: float,
          hours: float) -> tuple[bool, list[str], list[str]]:
    """Every claim of §VII.5, checked against what the run recorded.

    Returns ``(passed, problems, withheld)``. A verdict that was not reached
    because the run was too short is *withheld*, not passed: reporting a green
    soak from six minutes would be worse than reporting nothing.
    """
    problems: list[str] = []
    withheld: list[str] = []
    samples = result["samples"]
    judged = hours >= MIN_JUDGED_HOURS

    if len(samples) < 4:
        withheld.append(f"only {len(samples)} samples — nothing to judge")
    elif not judged:
        window = samples[int(len(samples) * (1 - LEAK_WINDOW)):]
        baseline, peak = window[0]["rss_mb"], max(s["rss_mb"] for s in window)
        if baseline > 0:
            result["rss_growth"] = round((peak - baseline) / baseline, 4)
            result["rss_baseline_mb"] = baseline
            result["rss_peak_mb"] = peak
        withheld.append(
            f"a {hours:g}h run is below the {MIN_JUDGED_HOURS:g}h needed for the "
            "leak and tick-rate verdicts — the bounded structures are still "
            "filling, so growth here is not a leak")
    else:
        window = samples[int(len(samples) * (1 - LEAK_WINDOW)):]
        baseline = window[0]["rss_mb"]
        peak = max(sample["rss_mb"] for sample in window)
        if baseline > 0:
            growth = (peak - baseline) / baseline
            result["rss_growth"] = round(growth, 4)
            result["rss_baseline_mb"] = baseline
            result["rss_peak_mb"] = peak
            if growth > max_growth:
                problems.append(
                    f"RSS grew {growth:.1%} over the second half "
                    f"({baseline:.0f} -> {peak:.0f} MB), past {max_growth:.0%}")

        first_rate = samples[1]["tick"] - samples[0]["tick"]
        last_rate = samples[-1]["tick"] - samples[-2]["tick"]
        result["ticks_per_sample_first"] = first_rate
        result["ticks_per_sample_last"] = last_rate
        if first_rate and last_rate < first_rate * 0.5:
            problems.append(
                f"tick rate halved: {first_rate} -> {last_rate} per sample")

    if not result["interrupted"]:
        if result["children_after"]:
            problems.append(f"{len(result['children_after'])} child processes "
                            "survived shutdown")
        if result["pending_after"] > 1:      # the driver's own task is expected
            problems.append(f"{result['pending_after']} asyncio tasks still pending")

    if state["unreadable"]:
        problems.append(f"{len(state['unreadable'])} unreadable state rows: "
                        f"{state['unreadable'][:3]}")
    if state["empty_after_load"]:
        problems.append(f"{len(state['empty_after_load'])} stores loaded empty "
                        f"despite having content: {state['empty_after_load'][:3]}")
    return (not problems), problems, withheld


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--max-growth", type=float, default=DEFAULT_MAX_GROWTH,
                        help="allowed RSS growth over the second half")
    parser.add_argument("--interrupt-at", type=float, default=None,
                        help="kill the run at this fraction (0..1) and verify "
                             "the state files afterwards")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "data" / "logs" / "soak_report.json")
    parser.add_argument("--state", type=Path, default=None,
                        help="where the soak's own state lives "
                             "(default: a temporary directory)")
    parser.add_argument("--tick-interval", type=float, default=0.0,
                        help="seconds to sleep between ticks; 0 runs flat out")
    args = parser.parse_args()

    if not HAS_PSUTIL:
        print("psutil is not installed — RSS cannot be measured, and the leak "
              "check is the point of this run.\n"
              "    pip install psutil")
        return 2

    import tempfile
    root = Path(args.state) if args.state else Path(tempfile.mkdtemp(prefix="soak_"))
    root.mkdir(parents=True, exist_ok=True)

    print(f"soak — {args.hours}h, state in {root}")
    print(f"       RSS may grow at most {args.max_growth:.0%} over the second half")
    if args.interrupt_at:
        print(f"       killing at {args.interrupt_at:.0%} of the run")
    print("-" * 72)

    substrate = _build(root)
    result = asyncio.run(_soak(substrate, args.hours * 3600, args.interrupt_at,
                               args.report, root, args.tick_interval))

    print("-" * 72)
    print("reading every state file back...")
    state = verify_state(root)
    passed, problems, withheld = judge(result, state, args.max_growth,
                                       args.hours)

    status = "failed" if problems else ("passed" if not withheld else "inconclusive")
    payload = {"status": status, "hours": args.hours, "problems": problems,
               "withheld": withheld, "state": state, **result}
    _write(args.report, payload)

    print(f"  ticks run            {result['samples'][-1]['tick'] if result['samples'] else 0}")
    if "rss_growth" in result:
        print(f"  RSS second half      {result['rss_baseline_mb']:.0f} -> "
              f"{result['rss_peak_mb']:.0f} MB  ({result['rss_growth']:+.1%}, "
              f"allowed {args.max_growth:.0%})")
    print(f"  child processes      {len(result['children_after'])}")
    print(f"  pending tasks        {result['pending_after']}")
    print(f"  state files checked  {state['checked']}")
    print(f"  report               {args.report}")
    print()
    for note in withheld:
        print(f"WITHHELD — {note}")
    if problems:
        for problem in problems:
            print(f"FAIL — {problem}")
        return 1
    if withheld:
        print()
        print("INCONCLUSIVE — the run was clean but too short to be a "
              f"soak. Use --hours {MIN_JUDGED_HOURS:g} or more.")
        return 2
    print("OK — the soak meets §VII.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
