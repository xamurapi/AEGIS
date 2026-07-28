"""The acceptance harnesses have to be reproducible (spec §3.1).

An A/B run that gives a different answer each time is not an experiment, and
the failure is quiet: both arms still finish, both still print a number, and the
number is simply wrong by an unknown amount. This is how a planner was recorded
at +113% and then at +84% on the same code.

The cause was not randomness — there is none in the package — but *timing*. The
held-out benchmark runs as a detached task, and how long its sandboxed
subprocesses take decides which tick its result lands on. Two runs diverge from
the first completion onward: same actions, same rewards, different accumulated
state. Measured at tick 56 of 60.

Both harnesses therefore pin the benchmark. These tests make sure they stay
pinned, by the only check that would have caught it: run the same arm twice and
compare the state digest tick by tick.
"""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TICKS = 70   # past the first benchmark completion (measured at 56)


@pytest.fixture(params=["scripts.ab_planner", "scripts.ab_policy"])
def harness(request):
    return importlib.import_module(request.param)


def _digests(harness, root: Path, ticks: int = TICKS) -> list[str]:
    """Per-tick state digests of one arm."""
    import aegis.config as cfg
    from aegis.clock import frozen

    with frozen() as clock:
        substrate = harness.build(True, root)
        digests = []

        async def go():
            for _ in range(ticks):
                if hasattr(harness, "energy_at"):
                    substrate.emotions.energy = harness.energy_at(
                        substrate.tick_count + 1)
                await substrate.tick()
                clock.advance(cfg.TICK_INTERVAL)
                digests.append(substrate.state_digest())

        harness._run(go())
        harness._run(substrate.cancel_background_tasks())
    return digests


def test_two_arms_of_the_same_configuration_agree_tick_by_tick(harness, tmp_path):
    first = _digests(harness, tmp_path / "first")
    second = _digests(harness, tmp_path / "second")
    assert len(first) == TICKS
    diverged = next((index for index, (a, b) in enumerate(zip(first, second))
                     if a != b), None)
    assert diverged is None, (
        f"{harness.__name__} diverged at tick {diverged}: the harness is not "
        "reproducible, so any number it reports is unrepeatable")


def test_the_benchmark_is_pinned_rather_than_timed(harness, tmp_path):
    """The specific thing that made it unreproducible.

    A detached benchmark task whose duration depends on subprocess scheduling
    lands its result on a different tick each run. Pinning it is what makes the
    comparison a comparison.
    """
    from aegis.clock import frozen

    with frozen():
        substrate = harness.build(True, tmp_path / "pinned")
        assert substrate._last_benchmark_score == 0.5
        harness._run(substrate._run_benchmark(0))
        # Still 0.5: the pinned stand-in does not go and measure anything.
        assert substrate._last_benchmark_score == 0.5
        harness._run(substrate.cancel_background_tasks())


def test_the_arm_writes_nothing_into_the_repository(harness, tmp_path):
    """Isolation, the other half of reproducibility.

    Redirecting only `aegis.config` looks like isolation and is not: a module
    that did `from aegis.config import X_DIR` bound the value at import time and
    never sees the change. The harness then writes into the live `data/`, and
    two runs of the "isolated" experiment share state.
    """
    from aegis.clock import frozen

    data = ROOT / "data"
    before = {path: path.stat().st_mtime_ns
              for path in data.rglob("*") if path.is_file()}

    with frozen() as clock:
        import aegis.config as cfg

        substrate = harness.build(True, tmp_path / "isolated")

        async def go():
            for _ in range(5):
                await substrate.tick()
                clock.advance(cfg.TICK_INTERVAL)

        harness._run(go())
        substrate._save_checkpoint()
        harness._run(substrate.cancel_background_tasks())

    after = {path: path.stat().st_mtime_ns
             for path in data.rglob("*") if path.is_file()}
    touched = sorted(str(path.relative_to(ROOT))
                     for path in set(before) | set(after)
                     if before.get(path) != after.get(path))
    assert touched == [], f"the harness wrote into the repository: {touched}"
