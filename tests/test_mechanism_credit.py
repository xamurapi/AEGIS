"""The mechanism vocabulary and the credit table (spec M11.5.3, M11.6.3).

Pinned: the mapping is total over every (op, kind) pair the diff can produce
and lands on exactly one mechanism; every mechanism is reachable (a mechanism
no edit can produce is decorative); the UCB gives +inf to the untried; the
order is deterministic and actually moves when credit accumulates.
"""
import pytest

from aegis.layers.metacognition.mechanism import (
    EDIT_KINDS, FIXED_TRANSFORM_ORDER, MECHANISMS, CreditTable,
    mechanism_for, mechanism_of_transform, reverse_mapping,
)
from aegis.layers.reasoning.dsl import OPS

FEATURES = ("family=arithmetic_chain", "steps>=4")


# ── the mapping (M11.5.3) ────────────────────────────────────────────

def test_every_producible_pair_maps_to_exactly_one_mechanism():
    for op in sorted(OPS):
        for kind in EDIT_KINDS:
            mechanism = mechanism_for(op, kind)
            assert mechanism in MECHANISMS, (op, kind, mechanism)


def test_the_spec_table_rows_hold():
    assert mechanism_for("DECOMPOSE", "insert") == "decomposition_shortened_chain"
    assert mechanism_for("DECOMPOSE", "param") == "decomposition_shortened_chain"
    assert mechanism_for("VERIFY", "insert") == "verification_caught_error"
    assert mechanism_for("ABSTAIN", "insert") == "abstention_avoided_confident_error"
    assert mechanism_for("VOTE", "insert") == "voting_reduced_variance"
    assert mechanism_for("VOTE", "param") == "voting_reduced_variance"
    assert mechanism_for("COMPUTE", "param") == "computation_replaced_guess"
    assert mechanism_for("PREDICT", "insert") == "prediction_pruned_branch"
    assert mechanism_for("BRANCH", "insert") == "prediction_pruned_branch"
    assert mechanism_for("RETRIEVE", "insert") == "retrieval_supplied_missing_fact"
    for op in sorted(OPS):
        assert mechanism_for(op, "reorder") == "reflection_reordered_work"
        assert mechanism_for(op, "wrap") == "reflection_reordered_work"


def test_no_mechanism_is_decorative():
    """The reverse mapping is non-empty for every mechanism."""
    table = reverse_mapping()
    assert set(table) == set(MECHANISMS)
    for mechanism, pairs in table.items():
        assert pairs, f"{mechanism} is producible by no edit"


def test_unknown_pairs_raise_rather_than_invent():
    with pytest.raises(KeyError):
        mechanism_for("EXEC", "insert")
    with pytest.raises(KeyError):
        mechanism_for("SOLVE", "sideways")


def test_every_generator_names_a_real_mechanism():
    for name in FIXED_TRANSFORM_ORDER:
        assert mechanism_of_transform(name) in MECHANISMS
    assert mechanism_of_transform("skeleton:vote_of_alternatives") in MECHANISMS
    assert mechanism_of_transform("no_such_generator") == ""


# ── the credit table (M11.6.3) ───────────────────────────────────────

def test_untried_mechanisms_score_infinite():
    table = CreditTable(mechanism_c=0.7)
    assert table.score("verification_caught_error", FEATURES) == float("inf")
    table.note_attempt("verification_caught_error", FEATURES)
    assert table.score("verification_caught_error", FEATURES) < float("inf")
    # A mechanism tried on one feature but not another is still unexplored.
    assert table.score("verification_caught_error",
                       FEATURES + ("incomplete",)) == float("inf")


def test_the_order_is_deterministic():
    table = CreditTable(mechanism_c=0.7)
    for _ in range(3):
        table.note_attempt("abstention_avoided_confident_error", FEATURES)
    table.note_accepted("abstention_avoided_confident_error", FEATURES, 0.1)
    assert table.order(FEATURES) == table.order(FEATURES)


def test_credit_moves_the_order():
    """The measurable 'behaviour changed': with every mechanism tried equally,
    the one with wins comes first; with no credit at all the tie falls to the
    canonical (fixed) order."""
    table = CreditTable(mechanism_c=0.7)
    names = list(FIXED_TRANSFORM_ORDER)
    for name in names:
        mechanism = mechanism_of_transform(name)
        for _ in range(4):
            table.note_attempt(mechanism, FEATURES)
    table.note_accepted(mechanism_of_transform("compute_instead_of_llm"),
                        FEATURES, 0.2)
    ordered = table.order(FEATURES, names)
    assert ordered[0] == "compute_instead_of_llm"
    assert table.order_differs(FEATURES, names)


def test_without_credit_ties_break_canonically():
    table = CreditTable(mechanism_c=0.7)
    names = sorted(FIXED_TRANSFORM_ORDER)
    assert table.order(FEATURES, names) == tuple(names)


def test_acceptance_counts_and_effects_accumulate_per_feature():
    table = CreditTable()
    table.note_attempt("voting_reduced_variance", FEATURES)
    table.note_accepted("voting_reduced_variance", FEATURES, 0.07)
    for feature in FEATURES:
        row = table.rows[("voting_reduced_variance", feature)]
        assert row.attempts == 1 and row.accepted == 1
        assert row.total_effect == pytest.approx(0.07)


def test_unknown_mechanisms_earn_nothing():
    table = CreditTable()
    table.note_attempt("plausible_sounding_story", FEATURES)
    assert not table.rows


def test_credit_round_trips_through_dict():
    table = CreditTable(mechanism_c=1.1)
    table.note_attempt("verification_caught_error", FEATURES)
    table.note_accepted("verification_caught_error", FEATURES, 0.05)
    restored = CreditTable.from_dict(table.to_dict())
    assert restored.mechanism_c == pytest.approx(1.1)
    assert restored.rows.keys() == table.rows.keys()
    for key in table.rows:
        assert restored.rows[key].as_dict() == table.rows[key].as_dict()


def test_win_rates_aggregate_across_features():
    table = CreditTable()
    table.note_attempt("verification_caught_error", ("a", "b"))
    table.note_accepted("verification_caught_error", ("a",), 0.1)
    rates = table.win_rates()
    entry = rates["verification_caught_error"]
    assert entry["attempts"] == 2 and entry["accepted"] == 1
    assert entry["win_rate"] == pytest.approx(0.5)


def test_mechanism_c_zero_is_pure_exploitation():
    table = CreditTable(mechanism_c=0.0)
    mech_a = mechanism_of_transform("add_verify")
    mech_b = mechanism_of_transform("add_predict")
    for _ in range(2):
        table.note_attempt(mech_a, FEATURES)
        table.note_attempt(mech_b, FEATURES)
    table.note_accepted(mech_a, FEATURES, 0.1)
    assert table.score(mech_a, FEATURES) == pytest.approx(0.5)
    assert table.score(mech_b, FEATURES) == pytest.approx(0.0)
