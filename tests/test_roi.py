"""ROI and reallocation (spec M4.5, M4.7).

Measuring what each activity returns is what lets the system move its own
budget toward what works. The rule that makes it safe rather than degenerate:
nothing is ever cut to zero, because an activity with no budget produces no
results and its ROI could never recover.
"""
import pytest

import aegis.config as cfg
from aegis.clock import FrozenClock, set_clock
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.motivation.roi import (
    DRIVES, MIN_OBSERVATIONS, ActivityROI, ROITracker, normalize_cost,
)


@pytest.fixture
def frozen():
    clock = FrozenClock(1_000_000.0)
    previous = set_clock(clock)
    yield clock
    set_clock(previous)


@pytest.fixture
def tracker(tmp_path, frozen):
    return ROITracker(store_path=tmp_path / "roi.json")


# ── cost normalisation ───────────────────────────────────────────────

def test_a_free_action_normalises_to_zero():
    assert normalize_cost(ResourceCost()) == 0.0


def test_one_unit_is_a_thousand_tokens():
    # The absolute scale matters: every ROI number in the system is quoted
    # against it, so a change here silently rescales the whole ledger.
    assert normalize_cost(ResourceCost(llm_tokens=1000)) == pytest.approx(1.0)


def test_ten_seconds_of_local_work_is_one_unit():
    assert normalize_cost(ResourceCost(wall_ms=10_000)) == pytest.approx(1.0)


def test_a_training_slot_is_five_units():
    assert normalize_cost(ResourceCost(training_slots=1)) == pytest.approx(5.0)


def test_a_mixed_cost_adds_its_parts():
    cost = ResourceCost(llm_tokens=1000, wall_ms=10_000, llm_calls=2)
    assert normalize_cost(cost) == pytest.approx(1.0 + 1.0 + 1.0)


def test_a_megabyte_of_disk_is_one_unit():
    assert normalize_cost(ResourceCost(disk_bytes=1024 * 1024)) == pytest.approx(1.0)


def test_a_network_call_is_priced():
    assert normalize_cost(ResourceCost(net_calls=5)) == pytest.approx(1.0)


def test_a_subprocess_slot_is_priced():
    assert normalize_cost(ResourceCost(subprocess_slots=2)) == pytest.approx(1.0)


def test_tokens_dominate_the_normalised_cost():
    tokens = normalize_cost(ResourceCost(llm_tokens=10_000))
    millis = normalize_cost(ResourceCost(wall_ms=10_000))
    assert tokens > millis


def test_a_training_slot_is_expensive():
    assert normalize_cost(ResourceCost(training_slots=1)) > \
        normalize_cost(ResourceCost(subprocess_slots=1))


def test_normalised_cost_is_never_negative():
    assert normalize_cost(ResourceCost(llm_tokens=-5)) >= 0.0


# ── measurement ──────────────────────────────────────────────────────

def test_recording_returns_value_per_unit_cost(tracker):
    roi = tracker.record("synth", ResourceCost(llm_tokens=1000), value=2.0)
    assert roi == pytest.approx(2.0)          # 1000 tokens normalises to 1.0


def test_a_dearer_action_scores_a_lower_return_for_the_same_value(tracker):
    cheap = tracker.record("cheap", ResourceCost(llm_tokens=500), value=1.0)
    dear = tracker.record("dear", ResourceCost(llm_tokens=4000), value=1.0)
    assert cheap == pytest.approx(2.0)        # value / 0.5 units
    assert dear == pytest.approx(0.25)        # value / 4.0 units
    assert cheap > dear


def test_a_blank_drive_does_not_erase_the_one_already_recorded(tracker):
    tracker.record("bench", ResourceCost(wall_ms=100), 1.0, drive="competence")
    tracker.record("bench", ResourceCost(wall_ms=100), 1.0, drive="")
    assert tracker.activities["bench"].drive == "competence"


def test_a_cheap_success_is_not_infinitely_profitable(tracker):
    # Reporting infinity would let one zero-cost action capture the entire
    # budget on its first observation.
    roi = tracker.record("free_win", ResourceCost(), value=1.0)
    assert roi < float("inf")


def test_repeated_observations_average(tracker):
    tracker.record("a", ResourceCost(llm_tokens=1000), value=1.0)
    tracker.record("a", ResourceCost(llm_tokens=1000), value=3.0)
    assert tracker.roi("a") == pytest.approx(2.0)


def test_variance_is_tracked_alongside_the_mean(tracker):
    for value in (1.0, 3.0, 2.0):
        tracker.record("a", ResourceCost(llm_tokens=1000), value=value)
    assert tracker.activities["a"].variance() > 0


def test_welford_computes_the_exact_sample_variance(tracker):
    # 1, 3, 2 -> mean 2, sample variance 1. Checking the exact numbers rather
    # than "greater than zero" is what makes the incremental formula testable
    # at all; a running average that drifts is invisible to a loose assertion.
    for value in (1.0, 3.0, 2.0):
        tracker.record("a", ResourceCost(llm_tokens=1000), value=value)
    entry = tracker.activities["a"]
    assert entry.mean == pytest.approx(2.0)
    assert entry.variance() == pytest.approx(1.0)


def test_the_running_mean_matches_a_plain_average(tracker):
    values = [0.5, 2.5, 4.0, 1.0, 3.0]
    for value in values:
        tracker.record("a", ResourceCost(llm_tokens=1000), value=value)
    assert tracker.roi("a") == pytest.approx(sum(values) / len(values))


def test_observations_are_counted(tracker):
    for _ in range(4):
        tracker.record("a", ResourceCost(llm_tokens=1000), 1.0)
    assert tracker.activities["a"].n == 4


def test_a_single_observation_has_no_variance(tracker):
    tracker.record("a", ResourceCost(llm_tokens=1000), value=1.0)
    assert tracker.activities["a"].variance() == 0.0


def test_an_unknown_activity_returns_zero(tracker):
    assert tracker.roi("never_run") == 0.0


def test_totals_accumulate(tracker):
    tracker.record("a", ResourceCost(llm_tokens=1000), value=1.0)
    tracker.record("a", ResourceCost(llm_tokens=1000), value=1.0)
    entry = tracker.activities["a"]
    assert entry.total_value == pytest.approx(2.0)
    assert entry.total_cost == pytest.approx(2.0)


def test_an_activity_remembers_which_drive_it_serves(tracker):
    tracker.record("bench", ResourceCost(wall_ms=100), 1.0, drive="competence")
    assert tracker.activities["bench"].drive == "competence"


def test_a_few_observations_are_not_yet_trusted(tracker):
    tracker.record("a", ResourceCost(llm_tokens=1000), 1.0)
    assert tracker.activities["a"].trusted() is False


def test_enough_observations_become_trusted(tracker):
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("a", ResourceCost(llm_tokens=1000), 1.0)
    assert tracker.activities["a"].trusted() is True


def test_drive_roi_ignores_untrusted_activities(tracker):
    tracker.record("noisy", ResourceCost(llm_tokens=1000), 99.0, drive="knowledge")
    assert tracker.drive_roi("knowledge") == 0.0


def test_drive_roi_averages_its_trusted_activities(tracker):
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("a", ResourceCost(llm_tokens=1000), 2.0, drive="knowledge")
        tracker.record("b", ResourceCost(llm_tokens=1000), 4.0, drive="knowledge")
    assert tracker.drive_roi("knowledge") == pytest.approx(3.0)


# ── reallocation ─────────────────────────────────────────────────────

def test_shares_always_sum_to_one(tracker):
    tracker.reallocate(tick=1)
    assert sum(tracker.shares.values()) == pytest.approx(1.0)


# ── the allocation rule, stated exactly ──────────────────────────────

def test_equal_weights_produce_an_equal_split():
    shares = ROITracker._normalized({d: 1.0 for d in DRIVES})
    assert all(v == pytest.approx(0.25) for v in shares.values())


def test_the_floor_is_handed_out_before_anything_is_shared():
    # One drive with all the weight: the others get exactly the floor, and the
    # winner gets the floor plus everything left over.
    shares = ROITracker._normalized({"competence": 1.0, "knowledge": 0.0,
                                     "coherence": 0.0, "stability": 0.0})
    floor = cfg.RESOURCE_MIN_SHARE
    assert shares["knowledge"] == pytest.approx(floor)
    assert shares["competence"] == pytest.approx(floor + (1.0 - floor * 4))


def test_the_remainder_is_split_in_proportion_to_the_weights():
    shares = ROITracker._normalized({"competence": 3.0, "knowledge": 1.0,
                                     "coherence": 0.0, "stability": 0.0})
    floor = cfg.RESOURCE_MIN_SHARE
    remainder = 1.0 - floor * 4
    assert shares["competence"] == pytest.approx(floor + remainder * 0.75)
    assert shares["knowledge"] == pytest.approx(floor + remainder * 0.25)


def test_normalised_shares_always_sum_to_one():
    for weights in ({"competence": 99.0}, {}, {d: 0.0 for d in DRIVES},
                    {"knowledge": 1.0, "stability": 1.0}):
        shares = ROITracker._normalized(weights)
        assert sum(shares.values()) == pytest.approx(1.0)


def test_no_weights_at_all_produce_an_equal_split():
    shares = ROITracker._normalized({d: 0.0 for d in DRIVES})
    assert all(v == pytest.approx(0.25) for v in shares.values())


def test_negative_weights_are_treated_as_zero():
    shares = ROITracker._normalized({"competence": -5.0, "knowledge": 1.0,
                                     "coherence": 0.0, "stability": 0.0})
    assert shares["competence"] == pytest.approx(cfg.RESOURCE_MIN_SHARE)


def test_an_over_large_floor_degrades_to_an_equal_split(monkeypatch):
    # Floors alone must never over-subscribe the budget; the cap is what stops
    # a misconfigured minimum from making the arithmetic impossible.
    monkeypatch.setattr(cfg, "RESOURCE_MIN_SHARE", 0.9)
    shares = ROITracker._normalized({"competence": 1.0, "knowledge": 0.0,
                                     "coherence": 0.0, "stability": 0.0})
    assert sum(shares.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(0.25) for v in shares.values())


def test_the_floor_is_capped_at_an_equal_split(monkeypatch):
    monkeypatch.setattr(cfg, "RESOURCE_MIN_SHARE", 0.9)
    assert ROITracker.floor() == pytest.approx(0.25)


def test_the_floor_is_the_configured_minimum_when_it_fits():
    assert ROITracker.floor() == pytest.approx(cfg.RESOURCE_MIN_SHARE)


def test_a_negative_configured_floor_is_treated_as_none(monkeypatch):
    monkeypatch.setattr(cfg, "RESOURCE_MIN_SHARE", -0.5)
    assert ROITracker.floor() == 0.0


def test_an_allocation_sitting_exactly_on_the_floor_is_valid():
    # Off-by-one in the tolerance direction would reject a legitimate
    # allocation and silently renormalise it on every load.
    floor = ROITracker.floor()
    shares = {"competence": 1.0 - 3 * floor, "knowledge": floor,
              "coherence": floor, "stability": floor}
    assert ROITracker._is_valid_allocation(shares) is True


def test_an_allocation_below_the_floor_is_not_valid():
    floor = ROITracker.floor()
    shares = {"competence": 1.0 - 3 * (floor / 2), "knowledge": floor / 2,
              "coherence": floor / 2, "stability": floor / 2}
    assert ROITracker._is_valid_allocation(shares) is False


def test_an_allocation_that_does_not_sum_to_one_is_not_valid():
    assert ROITracker._is_valid_allocation({d: 0.5 for d in DRIVES}) is False


def test_with_no_evidence_the_declared_split_stands(tracker):
    before = dict(tracker.shares)
    tracker.reallocate(tick=1)
    assert tracker.shares == pytest.approx(before)


def test_budget_moves_toward_what_pays_off(tracker):
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("bench", ResourceCost(llm_tokens=1000), 10.0, drive="competence")
        tracker.record("browse", ResourceCost(llm_tokens=1000), 0.1, drive="knowledge")
    tracker.reallocate(tick=1)
    assert tracker.share("competence") > tracker.share("knowledge")


def test_a_worthless_activity_falls_to_the_floor_but_not_to_zero(tracker):
    # An activity with no budget never produces a result, so its ROI could
    # never recover — the reallocation would be a one-way ratchet.
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("bench", ResourceCost(llm_tokens=1000), 10.0, drive="competence")
        tracker.record("dead", ResourceCost(llm_tokens=1000), 0.0, drive="knowledge")
    tracker.reallocate(tick=1)
    assert tracker.share("knowledge") == pytest.approx(cfg.RESOURCE_MIN_SHARE, abs=1e-6)
    assert tracker.share("knowledge") > 0


def test_every_drive_keeps_at_least_the_floor(tracker):
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("only", ResourceCost(llm_tokens=1000), 50.0, drive="competence")
    tracker.reallocate(tick=1)
    for drive in DRIVES:
        assert tracker.share(drive) >= cfg.RESOURCE_MIN_SHARE - 1e-9


def test_negative_returns_do_not_invert_the_proportion(tracker):
    # Losing money should mean less funding, not funding backwards.
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("bad", ResourceCost(llm_tokens=1000), -5.0, drive="knowledge")
        tracker.record("good", ResourceCost(llm_tokens=1000), 5.0, drive="competence")
    tracker.reallocate(tick=1)
    assert all(share >= 0 for share in tracker.shares.values())
    assert tracker.share("competence") > tracker.share("knowledge")


def test_reallocation_is_rate_limited(tracker):
    assert tracker.should_reallocate(0) is False
    assert tracker.should_reallocate(cfg.RESOURCE_REALLOC_EVERY_N_TICKS) is True
    tracker.reallocate(tick=cfg.RESOURCE_REALLOC_EVERY_N_TICKS)
    assert tracker.should_reallocate(cfg.RESOURCE_REALLOC_EVERY_N_TICKS + 1) is False


def test_a_reallocation_reports_what_changed(tracker):
    result = tracker.reallocate(tick=7)
    assert set(result) == {"before", "after", "roi", "tick"}
    assert result["tick"] == 7


def test_reallocations_are_counted(tracker):
    tracker.reallocate(tick=1)
    tracker.reallocate(tick=2)
    assert tracker.reallocations == 2


def test_a_share_converts_into_a_concrete_budget(tracker):
    tracker.shares = {"competence": 0.5, "knowledge": 0.2,
                      "coherence": 0.2, "stability": 0.1}
    assert tracker.budget_for("competence", 1000) == 500


# ── persistence ──────────────────────────────────────────────────────

def test_roi_history_survives_a_restart(tmp_path, frozen):
    path = tmp_path / "roi.json"
    first = ROITracker(store_path=path)
    for _ in range(MIN_OBSERVATIONS):
        first.record("a", ResourceCost(llm_tokens=1000), 2.0)
    first.save()
    assert ROITracker(store_path=path).roi("a") == pytest.approx(2.0)


def test_shares_survive_a_restart_unchanged(tmp_path, frozen):
    # Exactly unchanged: re-applying the allocation rule on load would drag the
    # split toward the floor on every restart, so a long-lived system would
    # slowly forget what it had learned to fund.
    path = tmp_path / "roi.json"
    first = ROITracker(store_path=path)
    first.shares = first._normalized({"competence": 0.7, "knowledge": 0.1,
                                      "coherence": 0.1, "stability": 0.1})
    first.save()
    revived = ROITracker(store_path=path)
    for drive in DRIVES:
        assert revived.share(drive) == pytest.approx(first.share(drive))


def test_many_restarts_do_not_erode_the_allocation(tmp_path, frozen):
    path = tmp_path / "roi.json"
    tracker = ROITracker(store_path=path)
    tracker.shares = tracker._normalized({"competence": 4.0, "knowledge": 1.0,
                                          "coherence": 1.0, "stability": 1.0})
    expected = tracker.share("competence")
    for _ in range(10):
        tracker.save()
        tracker = ROITracker(store_path=path)
    assert tracker.share("competence") == pytest.approx(expected)


def test_a_stored_split_that_does_not_add_up_is_repaired(tmp_path, frozen):
    import json
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "shares": {"competence": 5.0, "knowledge": 5.0,
                   "coherence": 5.0, "stability": 5.0},
    }), encoding="utf-8")
    tracker = ROITracker(store_path=path)
    assert sum(tracker.shares.values()) == pytest.approx(1.0)


def test_a_tracker_with_no_path_uses_the_configured_store(frozen):
    tracker = ROITracker(store_path=None)
    assert tracker._store_path.name == "roi.json"
    assert tracker._store_path.parent == cfg.MOTIVATION_DIR


def test_a_corrupt_roi_file_starts_clean(tmp_path, frozen):
    path = tmp_path / "roi.json"
    path.write_text("{ broken", encoding="utf-8")
    tracker = ROITracker(store_path=path)
    assert tracker.activities == {}
    assert tracker.shares == ROITracker.default_shares()


def test_a_malformed_activity_row_is_skipped(tmp_path, frozen):
    import json
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "activities": [{"no_name": 1},
                       {"activity": "good", "n": 3, "mean": 1.5}],
    }), encoding="utf-8")
    tracker = ROITracker(store_path=path)
    assert "good" in tracker.activities and len(tracker.activities) == 1


def test_malformed_shares_fall_back_to_the_default(tmp_path, frozen):
    import json
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({
        "schema_version": 2,
        "shares": {"competence": "lots", "knowledge": 0.3,
                   "coherence": 0.2, "stability": 0.15},
    }), encoding="utf-8")
    tracker = ROITracker(store_path=path)
    assert sum(tracker.shares.values()) == pytest.approx(1.0)


def test_an_activity_row_round_trips():
    entry = ActivityROI("a", "knowledge", n=2, mean=1.0, m2=0.5)
    restored = ActivityROI.from_dict(entry.to_dict())
    assert restored.activity == "a" and restored.n == 2


def test_a_broken_activity_row_is_rejected():
    assert ActivityROI.from_dict({"activity": "a", "n": "many"}) is None


# ── reporting ────────────────────────────────────────────────────────

def test_status_ranks_activities_by_return(tracker):
    for _ in range(MIN_OBSERVATIONS):
        tracker.record("low", ResourceCost(llm_tokens=1000), 1.0)
        tracker.record("high", ResourceCost(llm_tokens=1000), 9.0)
    assert tracker.status()["top"][0]["activity"] == "high"


def test_status_reports_the_current_split(tracker):
    assert set(tracker.status()["shares"]) == set(DRIVES)
