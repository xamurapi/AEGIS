"""Tests for the untouchable set (spec Appendix B)."""
import pytest

from aegis.safety.immutable import (
    BOUNDED_PARAMS, CATEGORIES, IMMUTABLE_PARAMS, MONOTONIC_PARAMS,
    ImmutableParameterError, assert_mutable, category_of, check_change,
    digest, is_immutable, normalize, status,
)


# ── the set itself ───────────────────────────────────────────────────

def test_all_seven_categories_present():
    assert set(CATEGORIES) == {
        "ethics_and_stop", "self_preservation", "sandbox", "code_self_mod",
        "control_plane", "training_guards", "resource_floors",
    }


def test_every_category_is_non_empty():
    for name, members in CATEGORIES.items():
        assert members, f"category {name} is empty"


def test_immutable_set_is_the_union_of_categories():
    union = {n for members in CATEGORIES.values() for n in members}
    assert IMMUTABLE_PARAMS == union


def test_categories_do_not_overlap():
    seen: set[str] = set()
    for members in CATEGORIES.values():
        assert not (seen & set(members))
        seen |= set(members)


def test_protection_levels_are_disjoint():
    """A name that is both immutable and bounded would be ambiguous."""
    assert not (IMMUTABLE_PARAMS & set(BOUNDED_PARAMS))
    assert not (IMMUTABLE_PARAMS & set(MONOTONIC_PARAMS))


# ── name normalization ───────────────────────────────────────────────

def test_normalize_strips_contour_prefix():
    assert normalize("evolution/SANDBOX_TIMEOUT") == "SANDBOX_TIMEOUT"
    assert normalize("parametric/API_TOKEN") == "API_TOKEN"
    assert normalize("  API_HOST  ") == "API_HOST"


def test_prefixed_immutable_name_is_still_caught():
    """Contours label proposals; a raw lookup would miss every one of them."""
    assert is_immutable("evolution/ETHICAL_THRESHOLD_AUTO")
    assert is_immutable("parametric/API_TOKEN")


def test_unknown_name_is_not_immutable():
    assert not is_immutable("plan_beam")
    assert category_of("plan_beam") is None


# ── assert_mutable ───────────────────────────────────────────────────

def test_assert_mutable_raises_on_protected_name():
    with pytest.raises(ImmutableParameterError):
        assert_mutable("ETHICAL_THRESHOLD_AUTO")


def test_assert_mutable_message_names_category_and_context():
    with pytest.raises(ImmutableParameterError) as exc:
        assert_mutable("API_TOKEN", context="evolution")
    text = str(exc.value)
    assert "control_plane" in text and "evolution" in text


def test_assert_mutable_allows_ordinary_gene():
    assert_mutable("plan_beam")  # must not raise


# ── check_change ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(IMMUTABLE_PARAMS))
def test_no_immutable_parameter_can_be_changed(name):
    verdict = check_change(name, 1.0, 2.0)
    assert not verdict.allowed
    assert "immutable" in verdict.reason


def test_verdict_cannot_be_rewritten_after_it_is_issued():
    """A safety verdict is evidence, not a suggestion. If a caller could flip
    `allowed` on a refusal it received, the whole gate would be advisory."""
    verdict = check_change("ETHICAL_THRESHOLD_AUTO", 0.7, 0.1)
    assert not verdict.allowed
    with pytest.raises(Exception):
        verdict.allowed = True
    assert not verdict.allowed


def test_ordinary_change_is_allowed():
    verdict = check_change("plan_beam", 5, 7)
    assert verdict.allowed and verdict.value == 7.0 and not verdict.clamped


def test_non_numeric_value_is_refused():
    verdict = check_change("plan_beam", 5, "wide")
    assert not verdict.allowed


def test_bounded_parameter_is_clamped_not_refused():
    verdict = check_change("SANDBOX_TIMEOUT", 3.0, 120.0)
    assert verdict.allowed and verdict.clamped and verdict.value == 30.0


def test_bounded_parameter_inside_range_passes_untouched():
    verdict = check_change("SANDBOX_TIMEOUT", 3.0, 8.0)
    assert verdict.allowed and not verdict.clamped and verdict.value == 8.0


def test_bounded_parameter_below_floor_is_clamped_up():
    verdict = check_change("SANDBOX_TIMEOUT", 3.0, 0.01)
    assert verdict.allowed and verdict.clamped and verdict.value == 0.5


def test_monotonic_up_rejects_a_decrease():
    """A training cool-down may grow; shrinking it is how you melt a GPU."""
    verdict = check_change("TRAIN_MIN_INTERVAL_SECONDS", 3600, 60)
    assert not verdict.allowed and "increase" in verdict.reason


def test_monotonic_up_accepts_an_increase():
    assert check_change("TRAIN_MIN_INTERVAL_SECONDS", 3600, 7200).allowed


def test_monotonic_down_rejects_an_increase():
    verdict = check_change("TRAIN_VAL_LOSS_THRESHOLD", 0.5, 0.9)
    assert not verdict.allowed and "decrease" in verdict.reason


def test_monotonic_down_accepts_a_decrease():
    assert check_change("TRAIN_VAL_LOSS_THRESHOLD", 0.5, 0.3).allowed


def test_monotonic_with_non_numeric_current_value_is_refused():
    verdict = check_change("TRAIN_MIN_INTERVAL_SECONDS", None, 7200)
    assert not verdict.allowed


# ── digest / status ──────────────────────────────────────────────────

def test_digest_is_stable_across_calls():
    assert digest() == digest()


def test_digest_changes_when_the_contract_changes(monkeypatch):
    before = digest()
    monkeypatch.setitem(BOUNDED_PARAMS, "SANDBOX_TIMEOUT", (0.5, 31.0))
    assert digest() != before


def test_status_reports_every_level():
    st = status()
    assert st["immutable_count"] == len(IMMUTABLE_PARAMS)
    assert set(st["categories"]) == set(CATEGORIES)
    assert "SANDBOX_TIMEOUT" in st["bounded"]
    assert st["monotonic"]["TRAIN_VAL_LOSS_THRESHOLD"] == "down"
    assert st["digest"] == digest()
