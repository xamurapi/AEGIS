"""REFLECT stays inside its budget with metacognition enabled (M11.10 #8).

The same probe and the same discipline as ``test_phase_budget_measured.py`` —
a clean subprocess, the real clock, the mean over 200 ticks — with one change:
``AEGIS_META_ENABLED=1``, so the tick-side hook of M11 (verdict folding, queue
upkeep) is actually running. The spec allots it ≤ 3 ms inside REFLECT's 15 and
forbids it the ablation; if the hook ever grew real work, this is the test
that names it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aegis.config import PHASE_BUDGET_MS

TICKS = 200
PROBE = Path(__file__).with_name("_phase_budget_probe.py")


@pytest.fixture(scope="module")
def phase_means():
    env = dict(os.environ)
    env["AEGIS_META_ENABLED"] = "1"
    result = subprocess.run(
        [sys.executable, str(PROBE), str(TICKS)],
        capture_output=True, text=True, timeout=600, env=env,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [row for row in result.stdout.strip().splitlines()
            if row.startswith("{")]
    assert line, f"the probe printed no measurement:\n{result.stdout[-2000:]}"
    return json.loads(line[-1])


def test_reflect_stays_within_budget_with_metacognition_on(phase_means):
    assert "reflect" in phase_means
    assert phase_means["reflect"] > 0.0
    budget = PHASE_BUDGET_MS["reflect"]
    assert phase_means["reflect"] <= budget, (
        f"REFLECT averaged {phase_means['reflect']:.2f} ms against "
        f"{budget} ms with metacognition enabled")
