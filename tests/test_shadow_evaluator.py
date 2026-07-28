"""The controlled trial and the two measurements behind it (spec M3.6).

`behaviour_delta_rate` is the direct answer to the development text's fifth
link, and it is the number that can be zero while the rule base looks healthy —
which is exactly the failure worth being able to see.

The interleave is ABAB in blocks rather than alternating ticks. Neighbouring
ticks are strongly correlated (energy, mood and focus barely move between two of
them), so alternating would put nearly identical situations in opposite arms and
measure noise instead of the rule.
"""
import json

import pytest

from aegis.layers.policy.counterfactual import TRIAL_BLOCKS, ShadowEvaluator
from aegis.layers.policy.rules import Rule


class _Rule:
    def __init__(self, id="r1", trial_started=0):
        self.id = id
        self.trial_started = trial_started


@pytest.fixture
def shadow(tmp_path):
    return ShadowEvaluator(store_path=tmp_path / "shadow.jsonl", trial_ticks=40)


# ── the interleave ───────────────────────────────────────────────────

def test_a_trial_is_divided_into_blocks_not_alternating_ticks(shadow):
    assert shadow.block_length() == 40 // TRIAL_BLOCKS == 10


def test_the_arms_alternate_by_block(shadow):
    rule = _Rule(trial_started=0)
    applied = [shadow.applies_this_tick(rule, tick) for tick in range(40)]
    assert applied[:10] == [True] * 10
    assert applied[10:20] == [False] * 10
    assert applied[20:30] == [True] * 10
    assert applied[30:40] == [False] * 10


def test_the_block_phase_starts_where_the_trial_did(shadow):
    """Two rules starting at different ticks must not march in lockstep, or
    each would confound the other's arms."""
    early, late = _Rule("early", trial_started=0), _Rule("late", trial_started=5)
    disagreements = sum(1 for tick in range(40)
                        if shadow.applies_this_tick(early, tick)
                        != shadow.applies_this_tick(late, tick))
    assert disagreements > 0


def test_a_rule_with_no_start_is_treated_as_starting_at_zero(shadow):
    assert shadow.applies_this_tick(_Rule(trial_started=None), 0) is True


def test_a_very_short_trial_still_has_a_block(tmp_path):
    shadow = ShadowEvaluator(store_path=tmp_path / "s.jsonl", trial_ticks=1)
    assert shadow.block_length() == 1
    rule = _Rule(trial_started=0)
    assert [shadow.applies_this_tick(rule, t) for t in range(4)] == \
        [True, False, True, False]


def test_a_trial_ends_after_its_configured_length(shadow):
    rule = _Rule(trial_started=100)
    assert not shadow.trial_finished(rule, 139)
    assert shadow.trial_finished(rule, 140)


def test_a_rule_that_never_started_has_not_finished(shadow):
    assert not shadow.trial_finished(_Rule(trial_started=None), 10_000)


# ── collecting ───────────────────────────────────────────────────────

def test_samples_land_in_the_arm_they_belong_to(shadow):
    shadow.record("r1", True, 0.8, tick=1)
    shadow.record("r1", False, 0.4, tick=11)
    applied, withheld = shadow.arms("r1")
    assert applied == [0.8] and withheld == [0.4]


def test_an_unusable_reward_is_not_a_sample(shadow):
    shadow.record("r1", True, "not a number")
    assert shadow.arms("r1") == ([], [])


def test_an_unknown_rule_has_empty_arms(shadow):
    assert shadow.arms("never_seen") == ([], [])


def test_a_concluded_rule_is_forgotten(shadow):
    shadow.record("r1", True, 0.8)
    shadow.forget("r1")
    assert shadow.arms("r1") == ([], [])


def test_an_arm_is_bounded(shadow):
    for index in range(6000):
        shadow.record("r1", True, index / 6000)
    applied, _ = shadow.arms("r1")
    assert len(applied) <= 5000
    # It keeps the *recent* end, which is the part describing the world now.
    assert applied[-1] == pytest.approx(5999 / 6000)


# ── behaviour change ─────────────────────────────────────────────────

def test_a_changed_top_choice_is_counted(shadow):
    assert shadow.note_decision("dream", "rest") is True
    assert shadow.ticks_changed == 1
    assert shadow.observed_delta_rate() == 1.0


def test_an_unchanged_top_choice_is_counted_too(shadow):
    assert shadow.note_decision("rest", "rest") is False
    assert shadow.ticks_seen == 1 and shadow.ticks_changed == 0
    assert shadow.observed_delta_rate() == 0.0


def test_the_delta_rate_is_the_share_of_ticks_the_policy_moved(shadow):
    for index in range(10):
        shadow.note_decision("dream" if index < 3 else "rest", "rest")
    assert shadow.observed_delta_rate() == 0.3


def test_the_rate_is_zero_before_any_decision(shadow):
    assert shadow.observed_delta_rate() == 0.0


def test_the_smoothed_rate_is_seeded_by_the_first_tick(shadow):
    """Seeding with the first observation rather than with zero: a fresh metric
    that reads 0.0 for its first few hundred ticks is at its most misleading
    exactly when somebody is watching it."""
    shadow.note_decision("dream", "rest")
    assert shadow.behaviour_delta_rate == 1.0


# ── counterfactual regret ────────────────────────────────────────────

def test_regret_is_what_the_road_not_taken_looked_worth(shadow):
    assert shadow.note_regret(0.4, 0.9) == pytest.approx(0.5)


def test_regret_smooths_across_ticks(shadow):
    shadow.note_regret(0.4, 0.9)
    second = shadow.note_regret(0.5, 0.5)
    assert 0.0 < second < 0.5


def test_unusable_regret_inputs_leave_the_estimate_alone(shadow):
    shadow.note_regret(0.4, 0.9)
    assert shadow.note_regret("x", None) == pytest.approx(0.5)


def test_regret_is_unknown_before_anything_happened(shadow):
    assert shadow.regret is None
    assert shadow.status()["counterfactual_regret"] is None


# ── verdicts ─────────────────────────────────────────────────────────

def test_a_verdict_compares_the_two_arms(shadow):
    for value in (0.8, 0.82, 0.79, 0.81):
        shadow.record("r1", True, value)
    for value in (0.4, 0.42, 0.39, 0.41):
        shadow.record("r1", False, value)
    verdict = shadow.verdict("r1")
    assert verdict.effect == pytest.approx(0.4, abs=0.01)
    assert verdict.significant(0.05)


def test_a_summary_reports_both_arms_in_full(shadow):
    shadow.record("r1", True, 0.8)
    shadow.record("r1", True, 0.9)
    shadow.record("r1", False, 0.4)
    shadow.record("r1", False, 0.5)
    summary = shadow.summary("r1")
    assert summary["n_applied"] == 2 and summary["n_withheld"] == 2
    assert summary["mean_applied"] == pytest.approx(0.85)
    assert summary["mean_withheld"] == pytest.approx(0.45)
    assert summary["effect"] == pytest.approx(0.4)


# ── persistence ──────────────────────────────────────────────────────

def test_rows_are_buffered_and_flushed_together(tmp_path):
    path = tmp_path / "shadow.jsonl"
    shadow = ShadowEvaluator(store_path=path)
    shadow.record("r1", True, 0.8, tick=3)
    shadow.record("r1", False, 0.4, tick=13)
    assert not path.exists()          # nothing on the critical path of a tick

    assert shadow.flush() == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["arm"] for row in rows] == ["applied", "withheld"]
    assert rows[0]["tick"] == 3 and rows[0]["rule"] == "r1"


def test_flushing_nothing_writes_nothing(tmp_path):
    shadow = ShadowEvaluator(store_path=tmp_path / "shadow.jsonl")
    assert shadow.flush() == 0


def test_an_unwritable_log_costs_the_rows_not_the_tick(tmp_path, monkeypatch):
    shadow = ShadowEvaluator(store_path=tmp_path / "nope" / "shadow.jsonl")
    shadow.record("r1", True, 0.5)

    def _explode(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.mkdir", _explode)
    assert shadow.flush() == 0        # swallowed, logged, the caller continues


def test_the_log_is_bounded(tmp_path, monkeypatch):
    import aegis.layers.policy.counterfactual as module

    monkeypatch.setattr(module, "MAX_SHADOW_ROWS", 10)
    path = tmp_path / "shadow.jsonl"
    shadow = ShadowEvaluator(store_path=path)
    for index in range(60):
        shadow.record("r1", index % 2 == 0, index / 60, tick=index)
        shadow.flush()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 20
    # The tail is kept: the recent rows are the ones describing the world now.
    assert json.loads(lines[-1])["tick"] == 59


def test_status_describes_the_trial_shape(shadow):
    shadow.note_decision("a", "b")
    shadow.record("r1", True, 0.5)
    status = shadow.status()
    assert status["ticks_seen"] == 1 and status["ticks_changed"] == 1
    assert status["trials"] == 1
    assert status["trial_ticks"] == 40 and status["block_length"] == 10


# ── the real Rule type works here too ────────────────────────────────

def test_a_real_rule_drives_the_interleave(shadow):
    rule = Rule(id="r", condition={"state": {}, "action": "rest"},
                effect="suppress", trial_started=7)
    assert shadow.applies_this_tick(rule, 7) is True
    assert shadow.applies_this_tick(rule, 17) is False
    assert shadow.trial_finished(rule, 47) is True
