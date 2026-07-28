"""Per-phase latency budgets (spec §3.4)."""
import asyncio

import pytest

from aegis.clock import frozen
from aegis.config import PHASE_BUDGET_WINDOW
from aegis.layers.health_monitor import PHASES, HealthMonitor
from aegis.layers.substrate import Substrate


@pytest.fixture
def health():
    return HealthMonitor()


# ── recording ────────────────────────────────────────────────────────

def test_every_phase_has_a_budget(health):
    for phase in PHASES:
        assert health.phase_budgets[phase] > 0


def test_record_phase_feeds_both_series(health):
    health.record_phase("decide", 4.0)
    report = health.phase_report()["decide"]
    assert report["samples_local"] == 1 and report["samples_all"] == 1


def test_external_sample_is_reported_but_not_budgeted(health):
    """A slow LLM call must never read as a slow cognitive cycle."""
    health.record_phase("act", 9000.0, external=True)
    report = health.phase_report()["act"]
    assert report["samples_all"] == 1
    assert report["samples_local"] == 0
    assert report["avg_all_ms"] == 9000.0
    assert report["avg_local_ms"] == 0.0
    assert not report["over_budget"]


def test_unknown_phase_is_ignored(health):
    health.record_phase("daydream", 1.0)
    assert "daydream" not in health.phase_report()


# ── budget evaluation ────────────────────────────────────────────────

def test_budget_not_judged_until_the_window_is_full(health):
    """One cold tick at start-up is not a regression."""
    health.record_phase("perceive", 10_000.0)
    assert not health.phase_report()["perceive"]["over_budget"]


def test_sustained_overrun_is_a_breach(health):
    budget = health.phase_budgets["decide"]
    for _ in range(PHASE_BUDGET_WINDOW):
        health.record_phase("decide", budget * 3)
    assert health.phase_report()["decide"]["over_budget"]


def test_within_budget_is_not_a_breach(health):
    budget = health.phase_budgets["decide"]
    for _ in range(PHASE_BUDGET_WINDOW * 2):
        health.record_phase("decide", budget * 0.5)
    assert not health.phase_report()["decide"]["over_budget"]


def test_one_spike_does_not_breach_a_healthy_window(health):
    budget = health.phase_budgets["reflect"]
    for _ in range(PHASE_BUDGET_WINDOW - 1):
        health.record_phase("reflect", budget * 0.1)
    health.record_phase("reflect", budget * 5)
    assert not health.phase_report()["reflect"]["over_budget"]


def test_breach_surfaces_as_a_warning_not_critical(health):
    """A phase that got slower is a regression to look at, not an emergency:
    escalating it to critical would push the system into emergency mode and
    stop it learning."""
    budget = health.phase_budgets["evaluate"]
    for _ in range(PHASE_BUDGET_WINDOW):
        health.record_phase("evaluate", budget * 4)
    report = health.check()
    assert any("evaluate over budget" in w for w in report["warnings"])
    assert not any("evaluate" in c for c in report["critical"])
    assert report["status"] != "critical"


def test_breaches_are_counted(health):
    budget = health.phase_budgets["act"]
    for _ in range(PHASE_BUDGET_WINDOW):
        health.record_phase("act", budget * 4)
    health.check()
    health.check()
    assert health.phase_report()["act"]["breaches"] == 2


def test_report_is_exposed_in_status(health):
    health.record_phase("perceive", 1.0)
    assert set(health.status()["phases"]) == set(PHASES)


def test_check_metrics_carry_the_phase_report(health):
    assert set(health.check()["metrics"]["phases"]) == set(PHASES)


# ── the real cycle ───────────────────────────────────────────────────

def _quiet(substrate: Substrate) -> Substrate:
    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.llm.enabled = False
    substrate.sensors.read_all = lambda: {"pinned": True}
    substrate.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    # "No subprocess work" includes evolution: a generation evaluates ten
    # variants in other processes, which is minutes against a 20 ms ACT budget.
    # It is detached in production for exactly that reason; here it is kept out
    # of the way entirely, like the network and the model.
    substrate.evolution.generation_running = True
    return substrate


def test_tick_records_every_phase(isolated_state):
    substrate = _quiet(Substrate())

    async def _one():
        await substrate.tick()

    asyncio.run(_one())
    report = substrate.health.phase_report()
    for phase in PHASES:
        assert report[phase]["samples_all"] >= 1, f"{phase} was not measured"


def test_deterministic_cycle_stays_inside_its_budgets(isolated_state):
    """The budget as an actual gate: the local cognitive cycle, with no network,
    no LLM and no subprocess work, must fit inside the per-phase limits."""
    with frozen() as clock:
        substrate = _quiet(Substrate())

        async def _drive():
            for _ in range(PHASE_BUDGET_WINDOW + 5):
                await substrate.tick()
                clock.advance(3.0)

        asyncio.run(_drive())

    report = substrate.health.phase_report()
    over = {p: d for p, d in report.items() if d["over_budget"]}
    assert not over, f"phases over budget: {over}"
