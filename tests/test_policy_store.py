"""Preferences: the fast half of "experience changes behaviour" (spec M3.3).

Two properties carry the design, and both are testable in isolation:

* it learns **advantage**, not reward — otherwise it would learn that
  everything works in good states, which is true and decides nothing;
* its steps **saturate** — nothing learned from experience alone becomes
  certain, so the planner's other terms can always outvote a preference.
"""
import pytest

from aegis.layers.policy.store import BASELINE_ALPHA, MIN_OBSERVATIONS, PolicyStore
from aegis.layers.world.state import StateKey


@pytest.fixture
def store(tmp_path):
    return PolicyStore(store_path=tmp_path / "preferences.json")


HIGH = StateKey(energy="hi", mood="curious")
LOW = StateKey(energy="lo", mood="tired")


# ── the update rule ──────────────────────────────────────────────────

def test_the_first_observation_sets_the_baseline_and_moves_nothing(store):
    """With no baseline yet, the first reward *is* the baseline, so its
    advantage is zero — a single sample cannot be above or below average."""
    assert store.update(HIGH, "rest", 0.8) == 0.0
    assert store.baselines[HIGH.key()] == pytest.approx(0.8)


def test_a_reward_above_the_baseline_raises_the_weight(store):
    store.update(HIGH, "rest", 0.2)          # seeds the baseline at 0.2
    weight = store.update(HIGH, "rest", 0.9)
    assert weight > 0


def test_a_reward_below_the_baseline_lowers_the_weight(store):
    store.update(HIGH, "rest", 0.9)
    weight = store.update(HIGH, "rest", 0.1)
    assert weight < 0


def test_the_step_is_exactly_the_advantage_rule(tmp_path):
    """w ← w + η·(r − baseline)·(1 − |w|), with η = 0.5.

    Baseline after the first sample of 0.4 is 0.4; the second sample of 0.8 has
    advantage 0.4, so w = 0 + 0.5·0.4·1 = 0.2.
    """
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.5)
    store.update(HIGH, "rest", 0.4)
    assert store.update(HIGH, "rest", 0.8) == pytest.approx(0.2)


def test_the_baseline_follows_what_the_state_actually_pays(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json")
    store.update(HIGH, "rest", 0.5)
    assert store.baseline(HIGH) == pytest.approx(0.5)
    store.update(HIGH, "dream", 1.0)
    # One step of exponential smoothing towards 1.0.
    assert store.baseline(HIGH) == pytest.approx(
        0.5 * (1 - BASELINE_ALPHA) + 1.0 * BASELINE_ALPHA)


def test_states_keep_separate_baselines(store):
    store.update(HIGH, "rest", 0.9)
    store.update(LOW, "rest", 0.1)
    assert store.baseline(HIGH) > store.baseline(LOW)


def test_learning_is_advantage_not_reward(tmp_path):
    """The same action, paying the same amount, in two states that pay
    differently: it is the *better* choice in the poor state and the worse one
    in the rich state, and the weights have to say so."""
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.5)
    for _ in range(5):
        store.update(HIGH, "other", 0.9)     # a rich state
        store.update(LOW, "other", 0.1)      # a poor one
    for _ in range(5):
        store.update(HIGH, "rest", 0.5)
        store.update(LOW, "rest", 0.5)

    assert store.weight_for(HIGH, "rest") < 0
    assert store.weight_for(LOW, "rest") > 0


def test_the_weight_saturates_and_never_reaches_certainty(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.9)
    for _ in range(500):
        store.update(HIGH, "other", 0.0)
        store.update(HIGH, "rest", 1.0)
    weight = store.weight_for(HIGH, "rest")
    assert 0.5 < weight < 1.0


def test_an_unusable_reward_changes_nothing(store):
    store.update(HIGH, "rest", 0.5)
    before = dict(store.preferences)
    assert store.update(HIGH, "rest", "not a number") == 0.0
    assert store.preferences == before


def test_an_empty_action_is_not_a_preference(store):
    assert store.update(HIGH, "", 0.9) == 0.0
    assert store.preferences == {}


# ── reading ──────────────────────────────────────────────────────────

def test_a_single_observation_is_a_story_not_evidence(store):
    store.update(HIGH, "rest", 0.5)
    assert store.preferences[store.pair_key(HIGH, "rest")]["n"] == 1
    assert store.weight_for(HIGH, "rest") == 0.0     # below MIN_OBSERVATIONS
    assert MIN_OBSERVATIONS == 2


def test_an_unseen_pair_contributes_nothing(store):
    assert store.weight_for(HIGH, "never_tried") == 0.0
    assert store.delta(HIGH, "never_tried") == 0.0


def test_the_delta_is_the_weight_scaled_by_the_policy_weight(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.5,
                        weight=0.25)
    store.update(HIGH, "rest", 0.4)
    store.update(HIGH, "rest", 0.8)
    assert store.delta(HIGH, "rest") == pytest.approx(
        store.weight_for(HIGH, "rest") * 0.25)


def test_a_state_string_reads_the_same_as_a_state_key(store):
    store.update(HIGH, "rest", 0.4)
    store.update(HIGH, "rest", 0.9)
    assert store.weight_for(HIGH.key(), "rest") == store.weight_for(HIGH, "rest")
    assert store.baseline(HIGH.key()) == store.baseline(HIGH)


def test_ranking_is_deterministic_under_ties(store):
    ranked = store.preferred(HIGH, ["zebra", "alpha", "middle"])
    assert [name for name, _ in ranked] == ["alpha", "middle", "zebra"]


def test_the_best_preference_ranks_first(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.5)
    for _ in range(4):
        store.update(HIGH, "good", 0.9)
        store.update(HIGH, "bad", 0.1)
    ranked = store.preferred(HIGH, ["bad", "good"])
    assert ranked[0][0] == "good"


# ── capacity ─────────────────────────────────────────────────────────

def test_eviction_keeps_what_is_informative(tmp_path):
    """Informativeness, not recency: a strong weight backed by observations is
    the store's actual knowledge, and evicting by age discards exactly that."""
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.9,
                        max_preferences=3)
    for _ in range(10):
        store.update(HIGH, "strong", 1.0)
        store.update(HIGH, "other", 0.0)
    for index in range(6):
        store.update(HIGH, f"weak_{index}", store.baseline(HIGH))

    assert len(store.preferences) <= 3
    assert store.pair_key(HIGH, "strong") in store.preferences


def test_capacity_regulation_has_a_floor(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json")
    store.regulate_capacity(1)
    assert store.max_preferences == 100


# ── persistence ──────────────────────────────────────────────────────

def test_preferences_survive_a_restart(tmp_path):
    path = tmp_path / "preferences.json"
    store = PolicyStore(store_path=path, learning_rate=0.5)
    for _ in range(4):
        store.update(HIGH, "rest", 0.9)
        store.update(HIGH, "other", 0.1)
    store.save()

    restored = PolicyStore(store_path=path, learning_rate=0.5)
    assert restored.weight_for(HIGH, "rest") == pytest.approx(
        store.weight_for(HIGH, "rest"))
    assert restored.baseline(HIGH) == pytest.approx(store.baseline(HIGH))
    assert restored.updates == store.updates


def test_a_malformed_row_costs_its_own_preference_only(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text(
        '{"schema_version": 2,'
        ' "preferences": {"a#x": {"w": 0.5, "n": 4}, "b#y": "not a row",'
        '                 "c#z": {"w": "NaNsense", "n": 2}},'
        ' "baselines": {"a": 0.5, "b": "nope"}, "updates": "many"}',
        encoding="utf-8")
    store = PolicyStore(store_path=path)
    assert "a#x" in store.preferences
    assert "b#y" not in store.preferences and "c#z" not in store.preferences
    assert store.baselines == {"a": 0.5}
    assert store.updates == 0


def test_a_stored_weight_out_of_range_is_clamped(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text(
        '{"schema_version": 2, "preferences": {"a#x": {"w": 9.0, "n": 4}}}',
        encoding="utf-8")
    assert PolicyStore(store_path=path).preferences["a#x"]["w"] == 1.0


def test_a_missing_file_is_an_empty_store(tmp_path):
    store = PolicyStore(store_path=tmp_path / "absent.json")
    assert store.preferences == {} and store.baselines == {}


def test_status_reports_the_strongest_preferences(tmp_path):
    store = PolicyStore(store_path=tmp_path / "p.json", learning_rate=0.5)
    for _ in range(4):
        store.update(HIGH, "rest", 0.9)
        store.update(HIGH, "other", 0.1)
    status = store.status()
    assert status["preferences"] == 2
    assert status["updates"] == 8
    # Both pairs are reported, with the sign of what each one paid relative to
    # the state's own average.
    by_action = {row["pair"].rsplit("#", 1)[1]: row for row in status["strongest"]}
    assert by_action["rest"]["w"] > 0 > by_action["other"]["w"]
    assert abs(status["strongest"][0]["w"]) >= abs(status["strongest"][1]["w"])
