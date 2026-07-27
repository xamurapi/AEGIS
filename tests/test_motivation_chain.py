"""The whole motivational chain, end to end (spec M4.8).

    goal → value → priority → resource → action → reward → ROI

The development text names this chain and complains that the last two links do
not exist. These tests follow one objective the whole way round and check the
link that makes the rest binding: an action that cannot be paid for does not
happen — and the system carries on anyway.
"""
import asyncio

import pytest

import aegis.config as cfg
from aegis.clock import frozen
from aegis.layers.motivation import Candidate, ResourceCost
from aegis.layers.substrate import Substrate


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def substrate(isolated_state):
    s = Substrate()

    async def _no_agents():
        return []

    async def _no_learning(*a, **k):
        return {"success": False}

    s.agent_system.run_due_agents = _no_agents
    s.external_learning.learn_from_source = _no_learning
    s.environment.step = lambda: {"reward": 0.0, "solved": False, "task": None}
    s.health.check = lambda: {"status": "healthy", "warnings": [],
                              "critical": [], "metrics": {}}
    return s


# ── the chain is wired ───────────────────────────────────────────────

def test_the_substrate_owns_every_link(substrate):
    assert substrate.goal_intelligence is not None    # value
    assert substrate.priority is not None             # priority
    assert substrate.resources is not None            # resource
    assert substrate.actions is not None              # action
    assert substrate.roi is not None                  # return


def test_the_cortex_cannot_spend_without_a_lease(substrate):
    # This is the link that was missing: before it, every model call happened
    # because the tick counter said so.
    assert substrate.llm.cortex.resources is substrate.resources


def test_every_link_appears_in_the_status(substrate):
    status = substrate.full_status()
    for section in ("resources", "roi", "priority", "action_space"):
        assert section in status


# ── one objective, all the way round ─────────────────────────────────

def test_an_action_gets_a_lease_and_settles_it(substrate):
    lease = substrate.acquire("evaluate_state_llm")
    assert lease is not None
    assert substrate.resources.budgets["llm_tokens"].used > 0

    substrate.settle(lease, ResourceCost(llm_tokens=300, llm_calls=1), value=1.0)
    assert lease.active is False
    assert substrate.resources.budgets["llm_tokens"].used == 300
    assert substrate.roi.roi("evaluate_state_llm") > 0


def test_a_lease_is_priced_from_the_registry(substrate):
    lease = substrate.acquire("curiosity_explore")
    spec = substrate.actions.by_name["curiosity_explore"]
    assert lease.cost.llm_tokens == spec.cost.llm_tokens


def test_a_lease_carries_the_priority_it_won_with(substrate):
    lease = substrate.acquire("rest")
    assert isinstance(lease.priority, float)


def test_an_explicit_priority_is_honoured(substrate):
    lease = substrate.acquire("rest", priority=9.5)
    assert lease.priority == 9.5


def test_acquiring_an_unknown_action_is_refused(substrate):
    assert substrate.acquire("no_such_action") is None


def test_settling_nothing_is_harmless(substrate):
    substrate.settle(None)


def test_a_safety_critical_action_is_leased_as_such(substrate):
    lease = substrate.acquire("health_check")
    assert lease.safety_critical is True


# ── exhaustion: the system survives having no budget (§M4.7) ─────────

def test_with_no_token_budget_the_model_is_never_called(substrate):
    substrate.resources.budgets["llm_tokens"].limit = 0
    assert substrate.acquire("evaluate_state_llm") is None


def test_with_no_token_budget_the_system_keeps_ticking(substrate):
    substrate.resources.budgets["llm_tokens"].limit = 0
    substrate.llm.enabled = True
    errors_before = substrate.health.error_count
    with frozen() as clock:
        async def _drive():
            for _ in range(30):
                await substrate.tick()
                clock.advance(3.0)

        _run(_drive())
    assert substrate.tick_count == 30
    assert substrate.health.error_count == errors_before


def test_with_no_token_budget_free_work_still_happens(substrate):
    substrate.resources.budgets["llm_tokens"].limit = 0
    with frozen() as clock:
        async def _drive():
            for _ in range(20):
                await substrate.tick()
                clock.advance(3.0)

        _run(_drive())
    # The deterministic contours are unaffected — that is the whole point of
    # every contour having a path that needs no model.
    assert substrate.feedback_loop.resolved > 0
    assert len(substrate.memory.episodic) > 0


def test_refusals_are_visible_rather_than_silent(substrate):
    substrate.resources.budgets["llm_tokens"].limit = 0
    substrate.acquire("evaluate_state_llm")
    assert substrate.resources.denied >= 1
    assert substrate.actions.status()["blocked"].get("resources", 0) >= 1


# ── the tick boundary keeps the budget honest ────────────────────────

def test_the_tick_refills_the_per_tick_allowance(substrate):
    _run(substrate.tick())
    used_after_first = substrate.resources.budgets["wall_ms"].used
    _run(substrate.tick())
    assert substrate.resources.budgets["wall_ms"].used <= max(1, used_after_first)


def test_a_lease_left_open_is_settled_at_the_end_of_the_tick(substrate):
    async def _leak():
        substrate.acquire("env_step")      # taken and never settled

    substrate._perceive = _leak
    _run(substrate.tick())
    assert substrate.resources.open_leases() == []


def test_budgets_survive_a_restart(substrate, isolated_state):
    substrate.acquire("curiosity_explore")
    substrate._save_checkpoint()
    revived = Substrate()
    assert revived.resources.budgets["llm_tokens"].used > 0


# ── priority actually orders the queue ───────────────────────────────

def test_priority_ranks_by_learned_value(substrate):
    substrate.goal_intelligence.reward(1.0, "expand_knowledge")
    substrate.goal_intelligence.reward(0.0, "rest")
    ordered = substrate.priority.order(
        [Candidate("rest", drive="stability"),
         Candidate("expand_knowledge", drive="knowledge")],
        substrate._ctx)
    assert ordered[0].objective == "expand_knowledge"


def test_priority_prefers_the_urgent_drive_when_errors_climb(substrate):
    calm = substrate.priority.urgency("coherence", {"error_rate": 0.0})
    urgent = substrate.priority.urgency("coherence", {"error_rate": 0.5})
    assert urgent > calm


# ── ROI closes the loop ──────────────────────────────────────────────

def test_a_settled_action_is_scored(substrate):
    lease = substrate.acquire("curiosity_explore")
    substrate.settle(lease, ResourceCost(llm_tokens=1000), value=2.0)
    assert substrate.roi.roi("curiosity_explore") == pytest.approx(2.0)


def test_the_score_is_filed_under_the_actions_drive(substrate):
    lease = substrate.acquire("env_step")
    substrate.settle(lease, ResourceCost(wall_ms=100), value=1.0)
    assert substrate.roi.activities["env_step"].drive == "competence"


def test_a_broken_roi_tracker_cannot_break_settlement(substrate):
    class Exploding:
        def record(self, *a, **k):
            raise RuntimeError("ledger on fire")

    lease = substrate.acquire("rest")
    substrate.roi = Exploding()
    substrate.settle(lease, value=1.0)      # must not raise
    assert lease.active is False


# ── telemetry (Appendix G) ───────────────────────────────────────────

def test_resource_metrics_reach_the_time_series(substrate):
    from aegis.telemetry import metrics as M
    _run(substrate.tick())
    substrate.telemetry.flush()
    assert len(substrate.telemetry.series(M.RES_SPENT)) >= 1
    assert len(substrate.telemetry.series(M.RES_DENIED)) >= 1


def test_roi_shares_reach_the_time_series(substrate):
    from aegis.telemetry import metrics as M
    _run(substrate.tick())
    substrate.telemetry.flush()
    drives = {row["tags"].get("drive")
              for row in substrate.telemetry.series(M.RES_SHARE).rows()}
    assert drives == {"competence", "knowledge", "coherence", "stability"}


def test_starvation_is_reported(substrate):
    from aegis.telemetry import metrics as M
    _run(substrate.tick())
    substrate.telemetry.flush()
    assert len(substrate.telemetry.series(M.RES_STARVATION_TICKS)) >= 1
