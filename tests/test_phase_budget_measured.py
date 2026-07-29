"""The phase budgets of §3.4, measured rather than described (spec §VII.7).

``tests/test_phase_budgets.py`` covers the budget *mechanism* — that the health
monitor records durations, judges them over a window and reports a sustained
overrun. This file asks the other question, the one §3.4 actually states: over
two hundred ticks of a real cognitive cycle, does each phase stay inside its
allowance?

    PERCEIVE ≤ 5 ms   EVALUATE ≤ 10 ms   DECIDE ≤ 30 ms
    ACT ≤ 20 ms       REFLECT ≤ 15 ms

The measurement is on the **mean**, as the spec says, and it excludes the two
things §3.4 itself excludes: external calls and heavy work, which the tick is
required to detach rather than perform. Anything left inside a phase is
something the cognitive cycle genuinely does every tick, and its cost is the
system's own.

The budgets are asserted as written, with no tolerance. That is affordable
because the measured margins are wide — the tightest phase sits at about a
quarter of its allowance — so a runner several times slower than this one still
passes. A tolerance would be buying headroom the system does not need, and the
first version of this file did exactly that: a factor of two quietly excused
ACT at 32 ms against a budget of 20, and PERCEIVE at 9.7 against 5.

**Measured in a clean subprocess, and that is not fastidiousness.** Run inside
the suite, ACT measured 4.4 ms alone and 28.9 ms after another test had built a
substrate — worker pools and detached tasks from the earlier run were still
competing for the same cores. A performance budget measured against that is a
measurement of the test suite, and it fails or passes for reasons that have
nothing to do with the code under test.

**On the real clock, deliberately.** Phase durations are measured with
``CLOCK.monotonic()``, and under the frozen clock the rest of the suite uses,
that does not advance — every phase records exactly 0 ms. Correct, and correct
for the reason the frozen clock exists: a duration must never reach the state
digest. It also means a budget test written under it passes by measuring
nothing, which is what the first version of this file did.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.config import PHASE_BUDGET_MS

#: §3.4 says "the mean of 200 ticks".
TICKS = 200

#: No headroom. See the module note: the margins are wide enough that the
#: budget itself is the honest assertion.
TOLERANCE = 1.0

PROBE = Path(__file__).with_name("_phase_budget_probe.py")


@pytest.fixture(scope="module")
def phase_means():
    """Mean duration per phase, measured by a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, str(PROBE), str(TICKS)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [row for row in result.stdout.strip().splitlines() if row.startswith("{")]
    assert line, f"the probe printed no measurement:\n{result.stdout[-2000:]}"
    return json.loads(line[-1])


def test_the_run_measured_every_phase(phase_means):
    """A guard on the test itself: a run that recorded nothing would pass every
    budget below by measuring nothing at all."""
    assert set(phase_means) >= set(PHASE_BUDGET_MS), sorted(phase_means)
    assert all(value > 0.0 for value in phase_means.values()), phase_means


@pytest.mark.parametrize("phase", sorted(PHASE_BUDGET_MS))
def test_the_phase_stays_within_its_budget(phase, phase_means):
    """One case per phase of §3.4, so a breach names the phase."""
    budget = PHASE_BUDGET_MS[phase]
    mean = phase_means[phase]
    assert mean <= budget * TOLERANCE, (
        f"{phase} averaged {mean:.2f} ms against a budget of {budget} ms "
        f"over {TICKS} ticks"
    )


def test_the_whole_tick_stays_within_the_sum_of_its_phases(phase_means):
    """The budgets are per phase; the tick is the thing an operator feels. If
    every phase passed and the total still overran, the time would be going
    somewhere no phase owns — which is exactly the kind of cost that grows
    unnoticed."""
    total_budget = sum(PHASE_BUDGET_MS.values())
    total_mean = sum(phase_means[phase] for phase in PHASE_BUDGET_MS)
    assert total_mean <= total_budget * TOLERANCE, (
        f"the cycle averaged {total_mean:.2f} ms against {total_budget} ms"
    )


def test_the_measured_profile_is_reported(phase_means):
    """Not an assertion about shape — a record of it.

    Which phase is expensive is a design question, not a contract: PERCEIVE
    carries sensor fusion and state encoding, ACT carries whatever the planner
    chose. Asserting an ordering here would encode today's balance as a rule
    and fail the day a contour legitimately moved cost from one phase to
    another. What is worth having is the numbers, visible when something else
    breaks.
    """
    profile = ", ".join(
        f"{phase} {phase_means[phase]:.2f}/{PHASE_BUDGET_MS[phase]:.0f}ms"
        for phase in sorted(PHASE_BUDGET_MS))
    assert all(phase_means[phase] <= PHASE_BUDGET_MS[phase] * TOLERANCE
               for phase in PHASE_BUDGET_MS), profile
