"""Acceptance harness for population evolution (spec §M5.9).

Three questions, and the second is the one that makes the first mean anything:

1. **Does the champion improve?** Fitness must rise by at least 15% over twenty
   generations, and must never fall — an evolution whose champion can regress
   has no elitism worth the name.
2. **Is the improvement real?** ``valid_test_gap ≤ 0.05``. Selection reads
   ``valid``; ``test`` is untouched until a champion is confirmed. A gap that
   grows while ``valid`` improves is evolution learning the validation set.
3. **Does it stay affordable?** The cost term is part of fitness, so a champion
   that bought its score with a ten-second sandbox timeout and a beam of
   sixteen fails on its own terms rather than needing a separate rule.

Deterministic: no RNG anywhere, a frozen clock, no network, no model. Two runs
give the same numbers.

Usage:
    python scripts/evo_bench.py                  # 20 generations
    python scripts/evo_bench.py --generations 5  # a quicker read
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aegis.config as cfg  # noqa: E402
from aegis.clock import frozen  # noqa: E402
from aegis.eval.pool import EvaluationPool  # noqa: E402
from aegis.layers.evolution.genome import Genome  # noqa: E402
from aegis.layers.evolution_engine import EvolutionEngine  # noqa: E402


def run(generations: int, root: Path, per_kind: int, workers: int) -> dict:
    """Run the search and report the curve."""
    pool = EvaluationPool(workers=workers, task_timeout=300.0)
    engine = EvolutionEngine(store_path=root / "lineage.json", pool=pool)
    engine.evaluator.per_kind = per_kind

    # Start from a deliberately poor configuration, so there is something to
    # find: a champion seeded at the optimum would make any curve flat and the
    # harness would be measuring nothing.
    start = Genome({"solver_timeout": 0.5, "solver_order": "by_length"})
    # Measure the starting point rather than declaring it zero. §M5.9 asks for
    # a gain "over the starting fitness", and a champion seeded at 0.0 makes
    # the first generation look like an infinite improvement — which says
    # nothing about whether the search works.
    baseline = engine.evaluator.evaluate([start])[0]
    engine.register_champion(start.to_dict(), fitness=baseline.fitness)

    curve, gaps = [baseline.fitness], []
    try:
        for _ in range(generations):
            summary = engine.run_generation(tick=engine.generation * 100)
            curve.append(engine.champion["fitness"])
            if summary["valid_test_gap"] is not None:
                gaps.append(abs(summary["valid_test_gap"]))
    finally:
        pool.shutdown()

    return {
        "curve": curve,
        "gaps": gaps,
        "champion": engine.champion,
        "promotions": engine.promotions,
        "novelty_skips": engine.archive.skips,
        "generations": engine.generation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--per-kind", type=int, default=3,
                        help="generated tasks per kind in the evaluation set")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gain", type=float, default=0.15,
                        help="required improvement over the starting fitness")
    parser.add_argument("--max-gap", type=float, default=0.05,
                        help="largest acceptable valid/test gap")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("evolution", "eval", "motivation"):
            (root / name).mkdir(parents=True, exist_ok=True)
        cfg.EVOLUTION_DIR = root / "evolution"
        with frozen():
            result = run(args.generations, root / "evolution",
                         args.per_kind, args.workers)

    curve = result["curve"]
    print(f"\nevo bench — {args.generations} generations\n" + "=" * 52)
    for index, fitness in enumerate(curve, 1):
        print(f"  gen {index:3d}  champion fitness {fitness:.4f}")
    print("-" * 52)

    if not curve:
        print("no generations ran")
        return 1

    first, last = curve[0], curve[-1]
    gain = (last - first) / abs(first) if first else last
    worst_gap = max(result["gaps"]) if result["gaps"] else 0.0
    regressions = sum(1 for a, b in zip(curve, curve[1:]) if b < a - 1e-9)

    print(f"  start / end           {first:.4f} -> {last:.4f}")
    print(f"  gain                  {gain * 100:+.1f}%  (required "
          f"{args.gain * 100:.0f}%)")
    print(f"  worst valid/test gap  {worst_gap:.4f}  (limit {args.max_gap})")
    print(f"  promotions            {result['promotions']}")
    print(f"  novelty skips         {result['novelty_skips']}")
    print(f"  regressions           {regressions}  (must be 0)")

    failures = []
    if gain < args.gain:
        failures.append(f"fitness gain {gain * 100:.1f}% below the required "
                        f"{args.gain * 100:.0f}%")
    if worst_gap > args.max_gap:
        failures.append(f"valid/test gap {worst_gap:.4f} above {args.max_gap} — "
                        "the population is learning the validation set")
    if regressions:
        failures.append(f"the champion regressed {regressions} time(s); "
                        "elitism should make that impossible")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK — evolution meets §M5.9")
    return 0


if __name__ == "__main__":
    sys.exit(main())
