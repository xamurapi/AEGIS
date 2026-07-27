"""State encoding (spec M1.3, Appendix D).

Everything the predictive model stores is keyed on these labels, so two
properties decide whether any of it works: the same situation must always
encode to the same key, and the bucketing must be coarse enough that a state
is visited often enough to learn anything about.
"""
import pytest

import aegis.config as cfg
from aegis.layers.world.state import (
    FIELDS, LABELS, PREFIX, UNKNOWN, StateEncoder, StateKey, bucket,
    collect_state_inputs, sanitize,
)


@pytest.fixture
def encoder():
    return StateEncoder()


# ── the key ──────────────────────────────────────────────────────────

def test_the_key_lists_every_field():
    key = StateKey(energy="mid", error="low", mood="curious", mode="focused",
                   focus_kind="knowledge", perf="up", load="lo").key()
    assert key == "e=mid|err=low|mo=curious|md=focused|fk=knowledge|pf=up|ld=lo"


def test_the_key_round_trips():
    original = StateKey(energy="hi", error="none", mood="calm", mode="reflective",
                        focus_kind="stability", perf="flat", load="mid")
    assert StateKey.parse(original.key()) == original


def test_parsing_junk_yields_unknowns_rather_than_raising():
    parsed = StateKey.parse("complete nonsense")
    assert parsed.energy == UNKNOWN


def test_parsing_a_partial_key_fills_the_rest_with_unknown():
    parsed = StateKey.parse("e=hi|md=focused")
    assert parsed.energy == "hi" and parsed.mode == "focused"
    assert parsed.mood == UNKNOWN


def test_an_unrecognised_field_is_ignored():
    parsed = StateKey.parse("e=hi|zz=whatever")
    assert parsed.energy == "hi"


def test_a_state_is_hashable_and_immutable():
    state = StateKey(energy="hi")
    assert {state: 1}[state] == 1
    with pytest.raises(Exception):
        state.energy = "lo"


def test_every_field_has_a_prefix():
    assert set(PREFIX) == set(FIELDS)


def test_prefixes_are_unique():
    assert len(set(PREFIX.values())) == len(PREFIX)


def test_str_is_the_key():
    state = StateKey(energy="hi")
    assert str(state) == state.key()


# ── sanitising ───────────────────────────────────────────────────────

def test_a_label_is_lowercased_and_separator_free():
    # The separators are what the key format is built from; a mood containing
    # one would produce a key that parses back into something else entirely.
    assert sanitize("Deeply Curious") == "deeply_curious"
    assert "|" not in sanitize("a|b")
    assert "=" not in sanitize("a=b")


def test_an_empty_label_becomes_unknown():
    assert sanitize("") == UNKNOWN
    assert sanitize(None) == "none"          # the string "None", not emptiness


def test_a_very_long_label_is_bounded():
    assert len(sanitize("x" * 200)) <= 24


# ── bucketing ────────────────────────────────────────────────────────

def test_a_value_below_every_cut_takes_the_first_label():
    assert bucket(0.1, [0.33, 0.66], ("lo", "mid", "hi")) == "lo"


def test_a_value_above_every_cut_takes_the_last_label():
    assert bucket(0.9, [0.33, 0.66], ("lo", "mid", "hi")) == "hi"


def test_a_value_exactly_on_a_cut_belongs_to_the_upper_bucket():
    assert bucket(0.33, [0.33, 0.66], ("lo", "mid", "hi")) == "mid"


def test_more_cuts_than_labels_clamps_rather_than_indexing_out_of_range():
    assert bucket(99, [1, 2, 3, 4], ("lo", "hi")) == "hi"


def test_an_unparseable_value_is_unknown():
    assert bucket("lots", [0.5], ("lo", "hi")) == UNKNOWN
    assert bucket(None, [0.5], ("lo", "hi")) == UNKNOWN


def test_no_labels_yields_unknown():
    assert bucket(0.5, [0.5], ()) == UNKNOWN


# ── the configured bins (Appendix D) ─────────────────────────────────

def test_energy_uses_the_declared_thirds(encoder):
    assert encoder.energy_label(0.1) == "lo"
    assert encoder.energy_label(0.5) == "mid"
    assert encoder.energy_label(0.9) == "hi"


def test_energy_boundaries_are_where_the_appendix_puts_them(encoder):
    assert encoder.energy_label(0.32) == "lo"
    assert encoder.energy_label(0.33) == "mid"
    assert encoder.energy_label(0.65) == "mid"
    assert encoder.energy_label(0.66) == "hi"


def test_a_clean_run_is_distinguished_from_a_nearly_clean_one(encoder):
    # "none" and "low" are different situations: the first means nothing has
    # gone wrong at all, which is the state most of a healthy run is in.
    assert encoder.error_label(0.0) == "none"
    assert encoder.error_label(0.01) == "low"
    assert encoder.error_label(0.5) == "high"


def test_load_is_measured_against_the_health_threshold(encoder):
    # As a fraction, not in milliseconds — otherwise the model would learn the
    # hardware rather than the behaviour.
    assert encoder.load_label(100, 1000) == "lo"
    assert encoder.load_label(700, 1000) == "mid"
    assert encoder.load_label(1500, 1000) == "hi"


def test_load_with_no_threshold_is_unknown(encoder):
    assert encoder.load_label(100, 0) == UNKNOWN
    assert encoder.load_label(100, None) == UNKNOWN


def test_performance_direction_needs_two_points(encoder):
    assert encoder.perf_label([]) == "flat"
    assert encoder.perf_label([0.5]) == "flat"


def test_performance_direction_is_read_over_the_window(encoder):
    assert encoder.perf_label([0.1, 0.2, 0.3, 0.4, 0.5]) == "up"
    assert encoder.perf_label([0.5, 0.4, 0.3, 0.2, 0.1]) == "down"


def test_a_wobble_inside_the_band_reads_as_flat(encoder):
    # Without the band a series that moves in the fourth decimal would alternate
    # direction every tick and split the state space for no reason.
    assert encoder.perf_label([0.500, 0.5001, 0.5002]) == "flat"


def test_non_numeric_history_entries_are_ignored(encoder):
    assert encoder.perf_label([None, 0.1, "x", 0.9]) == "up"


def test_the_encoder_reads_its_bins_from_configuration():
    custom = StateEncoder({"energy": {"lo": 0.8, "hi": 0.9}})
    assert custom.energy_label(0.5) == "lo"


def test_a_malformed_bin_specification_is_survivable():
    custom = StateEncoder({"energy": "not an object"})
    assert custom.energy_label(0.5) in LABELS["energy"]


def test_a_missing_cut_point_is_skipped():
    custom = StateEncoder({"energy": {"lo": 0.5}})     # no "hi"
    assert custom.energy_label(0.9) == "mid"


# ── encoding ─────────────────────────────────────────────────────────

def test_encoding_is_pure_and_repeatable(encoder):
    inputs = {"energy": 0.5, "error_rate": 0.01, "mood": "curious",
              "mode": "focused", "focus_kind": "knowledge",
              "bench_history": [0.1, 0.9], "avg_tick_ms": 10,
              "tick_threshold_ms": 1000}
    assert encoder.encode(inputs) == encoder.encode(inputs)


def test_encoding_the_same_situation_gives_the_same_key(encoder):
    a = encoder.encode({"energy": 0.5, "mood": "calm"})
    b = encoder.encode({"energy": 0.55, "mood": "calm"})
    assert a.energy == b.energy      # both land in the same bucket


def test_a_different_situation_gives_a_different_key(encoder):
    a = encoder.encode({"energy": 0.1})
    b = encoder.encode({"energy": 0.9})
    assert a != b


def test_missing_readings_encode_as_unknown_not_as_a_guess(encoder):
    # A tick before the first benchmark genuinely IS a different situation
    # from one after it, and the model should be allowed to learn that.
    state = encoder.encode({})
    assert state.mood == UNKNOWN
    assert state.energy == UNKNOWN


def test_encoding_none_is_survivable(encoder):
    assert isinstance(encoder.encode(None), StateKey)


def test_the_state_space_is_small_enough_to_learn(encoder):
    # Appendix D's estimate: about 13 000 states, of which a real run visits
    # two orders of magnitude fewer. Finer buckets would give a model that has
    # seen every state once and can predict nothing.
    assert encoder.space_size() < 20_000


# ── reading it off the live system ───────────────────────────────────

@pytest.fixture
def substrate(isolated_state):
    from aegis.layers.substrate import Substrate
    return Substrate()


def test_the_live_readings_cover_every_encoded_field(substrate):
    inputs = collect_state_inputs(substrate)
    assert set(inputs) >= {"energy", "error_rate", "mood", "mode", "focus_kind",
                           "bench_history", "avg_tick_ms", "tick_threshold_ms"}


def test_the_live_readings_encode_without_unknowns(substrate, encoder):
    state = encoder.encode(collect_state_inputs(substrate))
    assert state.energy != UNKNOWN
    assert state.mood != UNKNOWN
    assert state.mode != UNKNOWN


def test_a_focusless_system_reports_an_idle_focus(substrate):
    substrate.goals.get_current_focus = lambda: None
    assert collect_state_inputs(substrate)["focus_kind"] == "idle"


def test_a_broken_drive_classifier_does_not_break_the_reading(substrate):
    def explode(name):
        raise RuntimeError("classifier down")

    substrate.goal_intelligence._classify_drive = explode
    assert collect_state_inputs(substrate)["focus_kind"] in ("idle", "knowledge")
