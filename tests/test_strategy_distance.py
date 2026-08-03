"""Canonical form, distance and the strategy archive (spec M11.6.1, M11.11).

The properties pinned here are what make "principally different" a measurable
claim instead of a vibe: zero exactly for canonically equal strategies,
symmetry, strict positivity for different canonical hashes, invariance to the
order independent parameters are written in — and an archive that cuts a
near-duplicate before it costs an arena run.
"""
import pytest

from aegis.layers.metacognition.distance import (
    StrategyArchive, canonical_hash, canonicalize, distance, op_multiset,
)
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES, Library


def _step(op, **fields):
    return {"op": op, **fields}


DIRECT = [_step("SOLVE")]
VERIFY_TAIL = [_step("SOLVE"), _step("VERIFY", checker="confidence")]
DECOMPOSED = [_step("DECOMPOSE", max_parts=4), _step("SOLVE")]


# ── canonicalisation ─────────────────────────────────────────────────

def test_field_order_does_not_change_the_canonical_form():
    a = [{"op": "RETRIEVE", "source": "memory", "k": 5}]
    b = [{"k": 5, "op": "RETRIEVE", "source": "memory"}]
    assert canonicalize(a) == canonicalize(b)
    assert canonical_hash(a) == canonical_hash(b)


def test_integral_floats_normalise_to_integers():
    assert canonical_hash([_step("RETRIEVE", source="memory", k=5.0)]) \
        == canonical_hash([_step("RETRIEVE", source="memory", k=5)])


def test_a_trailing_reflect_is_removed():
    """A REFLECT nothing reads has no observable effect (M11.6.1)."""
    assert canonicalize([_step("SOLVE"), _step("REFLECT")]) \
        == canonicalize([_step("SOLVE")])


def test_a_reflect_that_is_read_survives():
    steps = [_step("SOLVE"), _step("REFLECT"),
             _step("BRANCH", cond="insufficient",
                   then=[_step("ABSTAIN")])]
    assert any(step.get("op") == "REFLECT" for step in canonicalize(steps))


def test_unknown_operations_are_dropped_not_kept():
    assert canonicalize([_step("SOLVE"), _step("EXEC")]) \
        == canonicalize([_step("SOLVE")])


# ── the distance ─────────────────────────────────────────────────────

def test_distance_to_self_is_exactly_zero():
    for steps in BUILTIN_STRATEGIES.values():
        assert distance(steps, steps) == 0.0


def test_distance_is_symmetric():
    names = sorted(BUILTIN_STRATEGIES)
    for a in names:
        for b in names:
            assert distance(BUILTIN_STRATEGIES[a], BUILTIN_STRATEGIES[b]) \
                == pytest.approx(distance(BUILTIN_STRATEGIES[b],
                                          BUILTIN_STRATEGIES[a]))


def test_different_hashes_mean_positive_distance():
    names = sorted(BUILTIN_STRATEGIES)
    for a in names:
        for b in names:
            if canonical_hash(BUILTIN_STRATEGIES[a]) \
                    != canonical_hash(BUILTIN_STRATEGIES[b]):
                assert distance(BUILTIN_STRATEGIES[a],
                                BUILTIN_STRATEGIES[b]) > 0.0


def test_distance_is_bounded_by_one():
    assert 0.0 <= distance(DIRECT, BUILTIN_STRATEGIES["predictive_check"]) <= 1.0


def test_parameter_permutation_is_distance_zero():
    a = [{"op": "VOTE", "n": 3, "agg": "majority", "body": [_step("SOLVE")]}]
    b = [{"agg": "majority", "body": [_step("SOLVE")], "n": 3, "op": "VOTE"}]
    assert distance(a, b) == 0.0


def test_permuting_the_same_operations_is_caught_by_the_structural_half():
    """Jaccard alone scores a permutation zero; the edit half must not."""
    a = [_step("SOLVE"), _step("VERIFY", checker="type")]
    b = [_step("VERIFY", checker="type"), _step("SOLVE")]
    assert distance(a, b) > 0.0


def test_swapping_one_operation_is_caught_by_the_compositional_half():
    a = [_step("SOLVE"), _step("VERIFY", checker="type")]
    b = [_step("SOLVE"), _step("PREDICT", horizon=1)]
    assert distance(a, b) > 0.0


def test_op_multiset_counts_through_nesting():
    counts = op_multiset([_step("VOTE", n=3, body=[_step("SOLVE")])])
    assert counts["VOTE"] == 1 and counts["SOLVE"] == 1


def test_distance_accepts_strategy_objects():
    class Holder:
        steps = DIRECT

    assert distance(Holder(), DIRECT) == 0.0


# ── the archive ──────────────────────────────────────────────────────

def test_archive_cuts_exact_duplicates():
    archive = StrategyArchive(near=0.05)
    archive.add("direct", DIRECT)
    assert archive.seen(DIRECT)
    assert not archive.is_novel(DIRECT)


def test_archive_cuts_near_duplicates_before_evaluation():
    archive = StrategyArchive(near=0.2)
    archive.add("verify_tail", VERIFY_TAIL)
    near = [_step("SOLVE"), _step("VERIFY", checker="type")]
    assert 0.0 < distance(VERIFY_TAIL, near) < 0.2
    assert not archive.is_novel(near)


def test_archive_admits_the_genuinely_different():
    archive = StrategyArchive(near=0.2)
    archive.add("direct", DIRECT)
    far = BUILTIN_STRATEGIES["predictive_check"]
    assert archive.is_novel(far)


def test_archive_min_distance_is_infinite_when_empty():
    assert StrategyArchive().min_distance(DIRECT) == float("inf")


def test_archive_deduplicates_on_the_real_library(tmp_path):
    """The acceptance line of stage 13: on the shipped library, every stored
    strategy is an exact-duplicate hit and none is 'novel' against itself."""
    library = Library(store_path=tmp_path / "strategies.json")
    archive = StrategyArchive(near=0.05)
    for strategy in library.strategies.values():
        archive.add(strategy.name, strategy.steps)
    for strategy in library.strategies.values():
        assert archive.seen(strategy.steps), strategy.name
        assert archive.min_distance(strategy.steps) == 0.0


def test_archive_round_trips_through_its_dict_form():
    archive = StrategyArchive(near=0.1)
    archive.add("direct", DIRECT)
    archive.note_skip()
    restored = StrategyArchive.from_dict(archive.to_dict())
    assert restored.seen(DIRECT)
    assert restored.skips == 1
    assert restored.near == pytest.approx(0.1)


def test_archive_capacity_is_bounded():
    archive = StrategyArchive(near=0.0, capacity=3)
    for index in range(6):
        archive.add(f"s{index}", [_step("RETRIEVE", source="memory", k=index + 1)])
    assert len(archive.entries) == 3
