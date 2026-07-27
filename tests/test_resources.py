"""Resources: the link that makes motivation binding (spec M4.3, M4.7).

An action that cannot get a lease does not run. Everything here checks that
this rule holds in both directions — that a refusal really stops the work, and
that a refusal never stops the *system*.
"""
import pytest

import aegis.config as cfg
from aegis.clock import FrozenClock, set_clock
from aegis.layers.motivation.resources import (
    CONCURRENT, HOUR_SECONDS, PER_TICK, WINDOWED, Lease, ResourceCost,
    ResourceManager,
)


@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


@pytest.fixture
def manager(tmp_path, frozen):
    return ResourceManager(store_path=tmp_path / "budgets.json")


# ── ResourceCost ─────────────────────────────────────────────────────

def test_a_cost_reports_every_kind():
    assert set(ResourceCost().as_dict()) == set(ResourceCost.KINDS)


def test_an_empty_cost_is_free():
    assert ResourceCost().is_free() is True
    assert ResourceCost(wall_ms=1).is_free() is False


def test_costs_add_kind_by_kind():
    total = ResourceCost(llm_tokens=10, wall_ms=5) + ResourceCost(llm_tokens=3, net_calls=1)
    assert total.llm_tokens == 13 and total.wall_ms == 5 and total.net_calls == 1


def test_a_cost_can_be_scaled():
    assert ResourceCost(llm_tokens=100).scaled(0.5).llm_tokens == 50


def test_a_cost_reads_defensively_from_a_dict():
    cost = ResourceCost.from_dict({"llm_tokens": "40", "wall_ms": None,
                                   "junk": 1})
    assert cost.llm_tokens == 40 and cost.wall_ms == 0


def test_a_malformed_cost_field_becomes_zero():
    assert ResourceCost.from_dict({"llm_tokens": "many"}).llm_tokens == 0


def test_a_cost_is_immutable():
    # Costs are shared between the registry, the lease and the ROI ledger; one
    # of them mutating a cost in place would silently rewrite the others.
    with pytest.raises(Exception):
        ResourceCost().llm_tokens = 5


def test_the_disk_budget_is_expressed_in_bytes(manager):
    # Configured in megabytes, held in bytes — getting the conversion backwards
    # would make the budget smaller than a single write.
    assert manager.budgets["disk_bytes"].limit == cfg.RES_DISK_MB * 1024 * 1024


def test_a_disk_write_is_charged(manager):
    manager.reserve(ResourceCost(disk_bytes=4096), "checkpoint")
    assert manager.budgets["disk_bytes"].used == 4096


# ── granting and refusing ────────────────────────────────────────────

def test_an_affordable_request_gets_a_lease(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=100), "thinking")
    assert lease is not None
    assert lease.purpose == "thinking"


def test_an_unaffordable_request_is_refused(manager):
    assert manager.reserve(ResourceCost(llm_tokens=10 ** 9), "greed") is None
    assert manager.denied == 1


def test_a_refusal_is_recorded_against_the_budget_that_ran_out(manager):
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "greed")
    assert manager.budgets["llm_tokens"].denials == 1


def test_a_granted_lease_takes_the_resource_out_of_the_budget(manager):
    manager.reserve(ResourceCost(llm_tokens=100), "thinking")
    assert manager.budgets["llm_tokens"].used == 100


def test_can_afford_agrees_with_reserve(manager):
    cost = ResourceCost(llm_tokens=10 ** 9)
    assert manager.can_afford(cost) is False
    assert manager.reserve(cost, "greed") is None


def test_can_afford_asks_as_ordinary_work_by_default(tmp_path, frozen):
    # Defaulting to safety-critical would report the reserved floor as spendable
    # to every caller that did not think to say otherwise.
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"llm_tokens": 1000})
    floor = int(1000 * cfg.RESOURCE_SAFETY_FLOOR)
    cost = ResourceCost(llm_tokens=1000 - floor + 1)
    assert manager.can_afford(cost) is False
    assert manager.can_afford(cost, safety_critical=True) is True


def test_a_lease_is_ordinary_unless_declared_otherwise(manager):
    assert manager.reserve(ResourceCost(llm_tokens=1), "ordinary").safety_critical is False


def test_lease_ids_are_unique(manager):
    ids = {manager.reserve(ResourceCost(llm_tokens=1), f"p{i}").id for i in range(5)}
    assert len(ids) == 5


# ── the safety floor (Appendix B, category 7) ────────────────────────

def test_ordinary_work_cannot_spend_the_safety_floor(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"llm_tokens": 1000})
    floor = int(1000 * cfg.RESOURCE_SAFETY_FLOOR)
    assert manager.reserve(ResourceCost(llm_tokens=1000 - floor + 1), "greedy") is None


def test_safety_critical_work_may_spend_the_floor(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"llm_tokens": 1000})
    assert manager.reserve(ResourceCost(llm_tokens=1000), "health_check",
                           safety_critical=True) is not None


def test_the_floor_survives_ordinary_work_draining_the_rest(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"wall_ms": 1000})
    floor = int(1000 * cfg.RESOURCE_SAFETY_FLOOR)
    manager.reserve(ResourceCost(wall_ms=1000 - floor), "ordinary")
    # Everything ordinary work could take is gone; the reserve is still there.
    assert manager.reserve(ResourceCost(wall_ms=1), "ordinary_more") is None
    assert manager.reserve(ResourceCost(wall_ms=floor), "checkpoint",
                           safety_critical=True) is not None


# ── settling ─────────────────────────────────────────────────────────

def test_committing_less_than_reserved_returns_the_difference(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=1000), "thinking")
    manager.commit(lease, ResourceCost(llm_tokens=200))
    assert manager.budgets["llm_tokens"].used == 200


def test_committing_more_than_reserved_charges_the_overspend(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=100), "thinking")
    manager.commit(lease, ResourceCost(llm_tokens=400))
    assert manager.budgets["llm_tokens"].used == 400


def test_committing_without_an_actual_charges_the_estimate(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=100), "thinking")
    manager.commit(lease)
    assert manager.budgets["llm_tokens"].used == 100


def test_a_committed_lease_is_no_longer_active(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=10), "thinking")
    manager.commit(lease)
    assert lease.active is False
    assert manager.open_leases() == []


def test_committing_twice_does_not_double_charge(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=100), "thinking")
    manager.commit(lease, ResourceCost(llm_tokens=100))
    manager.commit(lease, ResourceCost(llm_tokens=100))
    assert manager.budgets["llm_tokens"].used == 100


def test_releasing_hands_everything_back(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=500), "aborted")
    manager.release(lease)
    assert manager.budgets["llm_tokens"].used == 0


def test_releasing_an_already_settled_lease_is_harmless(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=10), "thinking")
    manager.commit(lease)
    manager.release(lease)
    assert manager.budgets["llm_tokens"].used == 10


def test_a_released_lease_is_no_longer_active(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=10), "aborted")
    manager.release(lease)
    assert lease.active is False
    assert manager.open_leases() == []


def test_releasing_twice_does_not_refund_twice(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=100), "aborted")
    manager.reserve(ResourceCost(llm_tokens=100), "other")
    manager.release(lease)
    manager.release(lease)
    assert manager.budgets["llm_tokens"].used == 100


def test_committing_a_none_lease_is_harmless(manager):
    manager.commit(None)
    manager.release(None)


def test_token_commits_accumulate_across_calls(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=5000), "thinking")
    manager.commit_tokens(lease, 100, calls=1)
    manager.commit_tokens(lease, 250, calls=1)
    assert lease.committed.llm_tokens == 350
    assert lease.committed.llm_calls == 2


def test_token_commits_on_a_dead_lease_are_ignored(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=10), "thinking")
    manager.commit(lease)
    manager.commit_tokens(lease, 999)
    assert lease.committed.llm_tokens == 10


def test_token_commits_on_a_none_lease_are_ignored(manager):
    manager.commit_tokens(None, 999)      # must not raise


def test_token_commits_need_an_object_that_declares_itself_live(manager):
    # An object with no `active` flag is not a lease this manager granted;
    # crediting it would let unmetered usage in through the side door.
    class NotALease:
        pass

    stranger = NotALease()
    manager.commit_tokens(stranger, 500)
    assert not hasattr(stranger, "committed")


# ── the tick boundary ────────────────────────────────────────────────

def test_per_tick_allowance_refills_every_tick(manager):
    manager.begin_tick(1)
    manager.reserve(ResourceCost(wall_ms=100), "work")
    assert manager.budgets["wall_ms"].used == 100
    manager.begin_tick(2)
    assert manager.budgets["wall_ms"].used == 0


def test_unused_tick_time_is_not_bankable(manager):
    # Milliseconds not spent were not saved; letting them accumulate would let
    # one tick spend a minute of wall clock it never earned.
    manager.begin_tick(1)
    manager.begin_tick(2)
    assert manager.budgets["wall_ms"].available() == manager.budgets["wall_ms"].limit


def test_the_hourly_window_slides(manager, frozen):
    manager.begin_tick(1)
    manager.reserve(ResourceCost(llm_tokens=1000), "thinking")
    frozen.advance(HOUR_SECONDS + 1)
    manager.begin_tick(2)
    assert manager.budgets["llm_tokens"].used == 0


def test_the_hourly_window_does_not_slide_early(manager, frozen):
    manager.begin_tick(1)
    manager.reserve(ResourceCost(llm_tokens=1000), "thinking")
    frozen.advance(HOUR_SECONDS - 10)
    manager.begin_tick(2)
    assert manager.budgets["llm_tokens"].used == 1000


def test_an_abandoned_lease_is_settled_at_the_end_of_the_tick(manager):
    # A phase that took a lease and then failed must not hold the reservation
    # forever; the budget would leak away one crash at a time.
    manager.begin_tick(1)
    lease = manager.reserve(ResourceCost(subprocess_slots=1), "crashed")
    manager.finalize_tick()
    assert lease.active is False
    assert manager.budgets["subprocess_slots"].held == 0


# ── concurrent slots ─────────────────────────────────────────────────

def test_slots_are_held_not_spent(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"subprocess_slots": 2})
    first = manager.reserve(ResourceCost(subprocess_slots=1), "a")
    manager.reserve(ResourceCost(subprocess_slots=1), "b")
    assert manager.reserve(ResourceCost(subprocess_slots=1), "c") is None
    manager.commit(first)
    assert manager.reserve(ResourceCost(subprocess_slots=1), "c") is not None


def test_a_released_slot_comes_straight_back(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"training_slots": 1})
    lease = manager.reserve(ResourceCost(training_slots=1), "train")
    manager.release(lease)
    assert manager.budgets["training_slots"].available() == 1


# ── exhaustion (§M4.7) ───────────────────────────────────────────────

def test_a_zero_token_budget_refuses_every_model_call(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"llm_tokens": 0})
    assert manager.reserve(ResourceCost(llm_tokens=1), "thinking") is None


def test_a_zero_token_budget_does_not_block_free_work(tmp_path, frozen):
    manager = ResourceManager(store_path=tmp_path / "b.json",
                              limits={"llm_tokens": 0})
    assert manager.reserve(ResourceCost(wall_ms=1), "local_work") is not None


# ── anti-starvation (§M4.7) ──────────────────────────────────────────

def test_waiting_starts_when_a_request_is_refused(manager):
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    assert manager.waiting_ticks("unlucky") == 0
    manager.begin_tick(1)
    assert manager.waiting_ticks("unlucky") == 1


def test_waiting_earns_priority(manager):
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    for tick in range(1, 21):
        manager.begin_tick(tick)
    assert manager.aging_bonus("unlucky") == pytest.approx(20 * cfg.PRIORITY_AGING)


def test_the_aging_bonus_is_capped(manager):
    # Unbounded aging turns anti-starvation into a different starvation: a
    # long-ignored trivial task would eventually outrank a health check.
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    for tick in range(1, 100_000, 97):
        manager.begin_tick(tick)
    assert manager.aging_bonus("unlucky") <= 1.0


def test_getting_the_lease_clears_the_wait(manager):
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    manager.begin_tick(1)
    manager.reserve(ResourceCost(llm_tokens=1), "unlucky")
    assert manager.waiting_ticks("unlucky") == 0


def test_starvation_beyond_the_limit_is_counted(manager):
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    for tick in range(1, cfg.PRIORITY_AGING_MAX_TICKS + 5):
        manager.begin_tick(tick)
    assert manager.starvation_ticks > 0


def test_an_unstarved_system_reports_none(manager):
    for tick in range(1, 50):
        manager.begin_tick(tick)
    assert manager.starvation_ticks == 0


def test_waiting_within_the_limit_is_not_starvation(manager):
    # Waiting is normal; only waiting *past the declared bound* is the failure
    # the metric is meant to surface.
    manager.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    for tick in range(1, cfg.PRIORITY_AGING_MAX_TICKS - 1):
        manager.begin_tick(tick)
    assert manager.starvation_ticks == 0


# ── persistence ──────────────────────────────────────────────────────

def test_usage_survives_a_restart(tmp_path, frozen):
    path = tmp_path / "b.json"
    first = ResourceManager(store_path=path)
    first.reserve(ResourceCost(llm_tokens=1234), "thinking")
    first.save()
    assert ResourceManager(store_path=path).budgets["llm_tokens"].used == 1234


def test_limits_come_from_configuration_not_from_disk(tmp_path, frozen):
    # A persisted limit would silently outrank an operator's env change.
    path = tmp_path / "b.json"
    first = ResourceManager(store_path=path, limits={"llm_tokens": 7})
    first.save()
    assert ResourceManager(store_path=path).budgets["llm_tokens"].limit \
        == cfg.RES_TOKENS_PER_HOUR


def test_a_corrupt_budget_file_starts_clean(tmp_path, frozen):
    path = tmp_path / "b.json"
    path.write_text("{ broken", encoding="utf-8")
    assert ResourceManager(store_path=path).budgets["llm_tokens"].used == 0


def test_a_malformed_budget_row_is_skipped(tmp_path, frozen):
    import json
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "budgets": {"llm_tokens": {"used": "lots"}, "wall_ms": {"used": 5}},
    }), encoding="utf-8")
    manager = ResourceManager(store_path=path)
    assert manager.budgets["llm_tokens"].used == 0
    assert manager.budgets["wall_ms"].used == 5


def test_a_budget_row_of_the_wrong_shape_is_skipped(tmp_path, frozen):
    import json
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "budgets": {"llm_tokens": "not an object", "wall_ms": {"used": 7}},
    }), encoding="utf-8")
    manager = ResourceManager(store_path=path)
    assert manager.budgets["llm_tokens"].used == 0
    assert manager.budgets["wall_ms"].used == 7


def test_a_budget_for_an_unknown_resource_is_ignored(tmp_path, frozen):
    import json
    path = tmp_path / "b.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "budgets": {"quantum_flux": {"used": 5}, "wall_ms": {"used": 7}},
    }), encoding="utf-8")
    manager = ResourceManager(store_path=path)
    assert "quantum_flux" not in manager.budgets
    assert manager.budgets["wall_ms"].used == 7


def test_waiting_state_survives_a_restart(tmp_path, frozen):
    path = tmp_path / "b.json"
    first = ResourceManager(store_path=path)
    first.reserve(ResourceCost(llm_tokens=10 ** 9), "unlucky")
    first.begin_tick(1)
    first.save()
    assert ResourceManager(store_path=path).waiting_ticks("unlucky") == 1


def test_corrupt_waiting_state_starts_clean(tmp_path, frozen):
    import json
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"schema_version": 2, "waiting": "not a dict"}),
                    encoding="utf-8")
    assert ResourceManager(store_path=path).waiting == {}


# ── reporting ────────────────────────────────────────────────────────

def test_status_reports_every_budget(manager):
    assert set(manager.status()["budgets"]) == set(ResourceCost.KINDS)


def test_status_lists_open_leases(manager):
    manager.reserve(ResourceCost(llm_tokens=1), "thinking")
    assert manager.status()["open_leases"][0]["purpose"] == "thinking"


def test_spent_reports_a_single_budget(manager):
    manager.reserve(ResourceCost(llm_tokens=42), "thinking")
    assert manager.spent("llm_tokens") == 42
    assert manager.spent("no_such_resource") == 0


def test_a_lease_exposes_its_token_allowance(manager):
    lease = manager.reserve(ResourceCost(llm_tokens=750), "thinking")
    assert lease.tokens == 750


def test_the_budget_modes_are_what_the_spec_declares():
    manager = ResourceManager(store_path=None)
    assert manager.budgets["llm_tokens"].mode == WINDOWED
    assert manager.budgets["wall_ms"].mode == PER_TICK
    assert manager.budgets["subprocess_slots"].mode == CONCURRENT
