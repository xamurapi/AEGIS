"""Canonical form and stable digests (spec M9.4)."""
import math

import pytest

from aegis.util.canonical import (
    FLOAT_PLACES, TIME_KEYS, canonical, canonical_json, digest_of,
)


# ── scalars ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, True, False, 0, -5, "text", ""])
def test_plain_scalars_pass_through(value):
    assert canonical(value) == value


def test_floats_are_rounded():
    assert canonical(1 / 3) == round(1 / 3, FLOAT_PLACES)


def test_float_noise_below_the_threshold_is_erased():
    """Two runs summing the same numbers in a different order must not look
    like different states."""
    assert canonical(0.1 + 0.2) == canonical(0.3)


def test_real_differences_survive_rounding():
    assert canonical(0.5) != canonical(0.5001)


def test_negative_zero_matches_zero():
    assert canonical(-0.0) == canonical(0.0)
    assert canonical_json({"v": -0.0}) == canonical_json({"v": 0.0})


def test_nan_becomes_a_stable_token():
    """NaN never equals itself; without a token the digest would be undefined."""
    assert canonical(float("nan")) == "<nan>"
    assert digest_of({"x": float("nan")}) == digest_of({"x": float("nan")})


@pytest.mark.parametrize("value,expected", [
    (float("inf"), "<inf>"), (float("-inf"), "<-inf>"),
])
def test_infinities_become_tokens(value, expected):
    assert canonical(value) == expected


def test_unknown_objects_fall_back_to_str():
    class Thing:
        def __str__(self):
            return "a-thing"

    assert canonical(Thing()) == "a-thing"


# ── containers ───────────────────────────────────────────────────────

def test_dict_key_order_does_not_matter():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_list_order_does_matter():
    """Sequence order IS behaviour — the order goals were created in decides
    which one wins a tie."""
    assert canonical([1, 2]) != canonical([2, 1])


def test_tuples_normalise_to_lists():
    assert canonical((1, 2)) == [1, 2]


def test_sets_are_ordered_by_canonical_form():
    """Set iteration order depends on hash seeding; the digest must not."""
    assert canonical({3, 1, 2}) == canonical({2, 3, 1})


def test_sets_of_mixed_types_are_still_ordered():
    first = canonical({1, "a", 2.5})
    second = canonical({"a", 2.5, 1})
    assert first == second


def test_non_string_keys_are_stringified():
    assert canonical({1: "a"}) == {"1": "a"}


def test_nesting_is_normalised_all_the_way_down():
    value = {"outer": [{"b": 1, "a": (1, 2)}]}
    assert canonical(value) == {"outer": [{"a": [1, 2], "b": 1}]}


def test_depth_is_bounded():
    """A cyclic or pathologically deep structure must not blow the stack."""
    deep = current = {}
    for _ in range(60):
        current["next"] = {}
        current = current["next"]
    assert "<max-depth>" in canonical_json(deep)


def test_self_referencing_structure_terminates():
    node = {"name": "loop"}
    node["self"] = node
    assert isinstance(canonical_json(node), str)


def test_depth_is_bounded_through_lists_too():
    """The guard has to advance on every container kind, not just dicts —
    otherwise a deep list walks straight past it."""
    deep = current = []
    for _ in range(60):
        nxt = []
        current.append(nxt)
        current = nxt
    assert "<max-depth>" in canonical_json(deep)


def test_self_referencing_list_terminates():
    node = ["loop"]
    node.append(node)
    assert isinstance(canonical_json(node), str)


def test_depth_is_bounded_through_sets_too():
    deep = frozenset({"leaf"})
    for _ in range(60):
        deep = frozenset({deep})
    assert "<max-depth>" in canonical_json(deep)


# ── time exclusion ───────────────────────────────────────────────────

@pytest.mark.parametrize("key", sorted(TIME_KEYS))
def test_every_time_key_is_dropped(key):
    assert canonical({key: 12345.6, "kept": 1}) == {"kept": 1}


def test_time_keys_are_dropped_at_every_depth():
    value = {"rows": [{"event": "x", "timestamp": 1.0}]}
    assert canonical(value) == {"rows": [{"event": "x"}]}


def test_a_run_started_later_has_the_same_digest():
    first = {"tick": 3, "created": 1000.0, "updated": 1001.0}
    second = {"tick": 3, "created": 9000.0, "updated": 9001.0}
    assert digest_of(first) == digest_of(second)


def test_a_different_tick_has_a_different_digest():
    assert digest_of({"tick": 3}) != digest_of({"tick": 4})


def test_exclusion_set_can_be_overridden():
    assert canonical({"created": 1.0}, exclude_keys=frozenset()) == {"created": 1.0}


# ── digest ───────────────────────────────────────────────────────────

def test_digest_is_deterministic():
    value = {"a": [1, 2, {"b": 0.5}]}
    assert digest_of(value) == digest_of(value)


def test_digest_length_follows_size():
    assert len(digest_of({}, size=16)) == 32
    assert len(digest_of({}, size=8)) == 16


def test_canonical_json_is_compact_and_sorted():
    text = canonical_json({"b": 1, "a": 2})
    assert text == '{"a":2,"b":1}'
