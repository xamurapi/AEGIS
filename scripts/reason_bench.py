"""Acceptance harness for the reasoning contour (spec §M6.10).

Held-out accuracy over a fixed number of cycles, plus the three things that
decide whether a number on that curve means anything:

1. **Is it measured on problems the system has never seen?** The working queue
   walks forward from index zero; the held-out set walks back from ten million.
   They cannot meet within any run length this system will reach, so the
   separation is structural rather than bookkeeping.
2. **Did abstention do its job?** The rate of *confident errors* — answered,
   wrong, and not an abstention — must fall. A system whose accuracy rises
   while its confident errors also rise has learned to guess more.
3. **Is the ceiling reachable at all?** The harness reports the best single
   built-in and the best hand-written combination, so a gain can be read
   against what is actually available rather than against 100%.

Deterministic: no RNG, a frozen clock, no network, no model. Two runs give the
same numbers.

Usage:
    python scripts/reason_bench.py                # 30 cycles
    python scripts/reason_bench.py --cycles 5     # a quicker read
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aegis.clock import frozen  # noqa: E402
from aegis.eval import reasoning_bench as bench  # noqa: E402
from aegis.layers.reasoning import ReasoningEngine  # noqa: E402

#: Problems worked per cycle. Enough that every strategy gets attempts in every
#: family within the first few cycles, so selection has something to select on.
TASKS_PER_CYCLE = 64

#: Held-out size. Large enough that a one-point move is not one lucky task.
HOLDOUT = 200

#: The best strategy the DSL can express by hand, used as the ceiling. It is
#: *not* installed in the library — it is what a synthesiser is supposed to
#: find, and a library that already contained it would make M6 unfalsifiable.
REACHABLE = [
    {"op": "DECOMPOSE", "max_parts": 8},
    {"op": "SOLVE"},
    {"op": "VERIFY", "checker": "confidence"},
    {"op": "BRANCH", "cond": "insufficient", "then": [{"op": "ABSTAIN"}]},
]


def _holdout_tasks(count: int) -> list:
    return [bench.build(10_000_000 - offset) for offset in range(count)]


def _score(engine: ReasoningEngine, tasks: list, strategy=None) -> dict:
    """Accuracy and confident-error rate over a fixed set."""
    solved = confident_errors = abstentions = 0
    for task in tasks:
        chosen = strategy or engine.select(str(task.family), task.id)
        trace = engine.interpreter.run(chosen, task, budget=engine._budget())
        if trace.solved:
            solved += 1
        elif not trace.abstained and trace.answer is not None:
            confident_errors += 1
        if trace.abstained:
            abstentions += 1
    total = max(1, len(tasks))
    return {"accuracy": solved / total,
            "confident_error_rate": confident_errors / total,
            "abstain_rate": abstentions / total}


def run(cycles: int, root: Path, holdout: int) -> dict:
    engine = ReasoningEngine(store_path=root / "strategies.json")
    tasks = _holdout_tasks(holdout)

    curve = [_score(engine, tasks)]
    for _ in range(cycles):
        engine.solve(TASKS_PER_CYCLE)
        curve.append(_score(engine, tasks))

    per_builtin = {
        strategy.name: _score(engine, tasks, strategy)["accuracy"]
        for strategy in engine.library.builtins()}
    reachable = _score(engine, tasks, REACHABLE)["accuracy"]

    return {"curve": curve, "per_builtin": per_builtin, "reachable": reachable,
            "weaknesses": engine.weaknesses(), "status": engine.status()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=30)
    parser.add_argument("--holdout", type=int, default=HOLDOUT)
    parser.add_argument("--gain", type=float, default=0.15,
                        help="required rise in held-out accuracy over the start")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with frozen():
            result = run(args.cycles, root, args.holdout)

    curve = result["curve"]
    print(f"\nreason bench — {args.cycles} cycles, {args.holdout} held-out "
          f"problems\n" + "=" * 64)
    for index, point in enumerate(curve):
        print(f"  cycle {index:3d}  held-out {point['accuracy']:.4f}   "
              f"confident errors {point['confident_error_rate']:.4f}   "
              f"abstentions {point['abstain_rate']:.4f}")
    print("-" * 64)

    print("  best single built-in:")
    for name, score in sorted(result["per_builtin"].items(),
                              key=lambda row: -row[1])[:3]:
        print(f"    {name:28s} {score:.4f}")
    print(f"  reachable by combining them  {result['reachable']:.4f}")
    if result["weaknesses"]:
        worst = result["weaknesses"][0]
        print(f"  weakest class                {worst['family']} "
              f"({worst['solved']}/{worst['used']})")

    first, last = curve[0], curve[-1]
    gain = last["accuracy"] - first["accuracy"]
    error_change = last["confident_error_rate"] - first["confident_error_rate"]

    print("-" * 64)
    print(f"  held-out accuracy     {first['accuracy']:.4f} -> "
          f"{last['accuracy']:.4f}  ({gain * 100:+.1f} pp, required "
          f"{args.gain * 100:+.0f})")
    print(f"  confident errors      {first['confident_error_rate']:.4f} -> "
          f"{last['confident_error_rate']:.4f}  ({error_change * 100:+.1f} pp, "
          "must not rise)")

    failures = []
    if gain < args.gain:
        failures.append(f"held-out gain {gain * 100:.1f} pp below the required "
                        f"{args.gain * 100:.0f} pp")
    if error_change > 1e-9:
        failures.append(f"confident errors rose by {error_change * 100:.1f} pp — "
                        "the system has learned to guess more, not to reason more")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK — the reasoning contour meets §M6.10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
