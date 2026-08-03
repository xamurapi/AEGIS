"""The skeleton catalogue (spec M11.6.2, M11.6.5).

Pinned: six structurally distinct shapes (pairwise distance >= META_FAR), a
deterministic fill that the credit table can steer, and permanent retirement
per (skeleton, features) — never per skeleton.
"""
from itertools import combinations

import aegis.config as cfg
from aegis.layers.metacognition.distance import canonical_hash, distance
from aegis.layers.metacognition.mechanism import CreditTable, mechanism_of_transform
from aegis.layers.metacognition.skeletons import (
    SKELETON_NAMES, SkeletonCatalog, fill, skeleton_template,
)
from aegis.layers.reasoning.dsl import validate

FEATURES = ("family=arithmetic_chain", "steps>=4")


def test_the_catalogue_holds_at_least_six_shapes():
    assert len(SKELETON_NAMES) >= 6


def test_every_skeleton_validates_against_the_dsl():
    credit = CreditTable()
    for name in SKELETON_NAMES:
        assert validate(fill(name, FEATURES, credit)) == [], name


def test_skeletons_are_pairwise_far():
    """The catalogue line: distance >= META_FAR for every pair, or the far
    generator is a neighbourhood generator wearing a costume."""
    credit = CreditTable()
    forms = {name: fill(name, FEATURES, credit) for name in SKELETON_NAMES}
    for a, b in combinations(sorted(forms), 2):
        assert distance(forms[a], forms[b]) >= cfg.META_FAR, (a, b)


def test_the_fill_is_deterministic():
    credit = CreditTable()
    for name in SKELETON_NAMES:
        assert canonical_hash(fill(name, FEATURES, credit)) \
            == canonical_hash(fill(name, FEATURES, credit))


def test_the_fill_depends_on_the_features():
    """Different weaknesses may fill differently — the table of M11.6.2 is
    keyed by the weakness's features, not global."""
    credit = CreditTable()
    changed = 0
    for name in SKELETON_NAMES:
        a = canonical_hash(fill(name, ("family=logic_grid",), credit))
        b = canonical_hash(fill(name, ("incomplete",), credit))
        changed += 1 if a != b else 0
    assert changed > 0


def test_the_credit_table_steers_the_fill():
    """Accumulated credit changes the rank and the rank feeds the fill: at
    least one skeleton fills differently once credit has moved."""
    empty = CreditTable()
    earned = CreditTable()
    for name in SKELETON_NAMES:
        mechanism = mechanism_of_transform(f"skeleton:{name}")
        for _ in range(3):
            earned.note_attempt(mechanism, FEATURES)
    earned.note_accepted(
        mechanism_of_transform("skeleton:vote_of_alternatives"), FEATURES, 0.2)
    changed = sum(
        1 for name in SKELETON_NAMES
        if canonical_hash(fill(name, FEATURES, empty))
        != canonical_hash(fill(name, FEATURES, earned)))
    assert changed > 0


def test_templates_are_fresh_copies():
    a = skeleton_template("vote_of_alternatives")
    a[0]["n"] = 99
    assert skeleton_template("vote_of_alternatives")[0]["n"] != 99


# ── retirement (M11.6.5) ─────────────────────────────────────────────

def test_retirement_is_per_pair_not_per_skeleton():
    catalog = SkeletonCatalog(retire_after=3)
    for _ in range(3):
        catalog.note_failure("vote_of_alternatives", FEATURES)
    assert catalog.is_retired("vote_of_alternatives", FEATURES)
    # The same shape on different features is still available.
    assert not catalog.is_retired("vote_of_alternatives", ("incomplete",))
    # And other skeletons on the same features are untouched.
    assert not catalog.is_retired("decompose_solve_verify", FEATURES)


def test_retirement_is_permanent():
    catalog = SkeletonCatalog(retire_after=2)
    catalog.note_failure("reflect_retry_bounded", FEATURES)
    catalog.note_failure("reflect_retry_bounded", FEATURES)
    assert catalog.is_retired("reflect_retry_bounded", FEATURES)
    catalog.note_success("reflect_retry_bounded", FEATURES)
    assert catalog.is_retired("reflect_retry_bounded", FEATURES), \
        "a win must not resurrect a retired pair"


def test_a_success_resets_the_failure_count_before_retirement():
    catalog = SkeletonCatalog(retire_after=3)
    catalog.note_failure("decompose_solve_verify", FEATURES)
    catalog.note_failure("decompose_solve_verify", FEATURES)
    catalog.note_success("decompose_solve_verify", FEATURES)
    catalog.note_failure("decompose_solve_verify", FEATURES)
    assert not catalog.is_retired("decompose_solve_verify", FEATURES)


def test_available_filters_the_retired_pair_only():
    catalog = SkeletonCatalog(retire_after=1)
    catalog.note_failure("retrieve_compute_verify", FEATURES)
    available = catalog.available(FEATURES)
    assert "retrieve_compute_verify" not in available
    assert set(available) | {"retrieve_compute_verify"} == set(SKELETON_NAMES)
    assert "retrieve_compute_verify" in catalog.available(("incomplete",))


def test_feature_order_does_not_split_the_registry():
    catalog = SkeletonCatalog(retire_after=1)
    catalog.note_failure("vote_of_alternatives",
                         ("steps>=4", "family=arithmetic_chain"))
    assert catalog.is_retired("vote_of_alternatives",
                              ("family=arithmetic_chain", "steps>=4"))


def test_the_registry_round_trips_through_dict():
    catalog = SkeletonCatalog(retire_after=2)
    catalog.note_failure("vote_of_alternatives", FEATURES)
    catalog.note_failure("vote_of_alternatives", FEATURES)
    restored = SkeletonCatalog.from_dict(catalog.to_dict())
    assert restored.is_retired("vote_of_alternatives", FEATURES)
    assert restored.retire_after == 2
    assert restored.retired_report() == catalog.retired_report()
