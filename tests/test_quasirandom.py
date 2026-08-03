"""Deterministic replacements for randomness (spec §3.1).

Two properties have to hold or the whole zero-randomness guarantee is theatre:
the sequences must actually SPREAD (otherwise they are a constant dressed up as
exploration) and they must be REPRODUCIBLE across instances and processes.
"""
import pytest

from aegis.util.quasirandom import (
    PRIMES, bootstrap_indices, halton, halton_sequence, hash_choice, hash_index,
    hash_unit, scaled, signed, van_der_corput,
)


# ── van der Corput ───────────────────────────────────────────────────

def test_van_der_corput_base2_matches_known_values():
    # Radical inverse in base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8
    expected = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]
    assert [van_der_corput(i, 2) for i in range(8)] == expected


def test_van_der_corput_base3_matches_known_values():
    assert van_der_corput(1, 3) == pytest.approx(1 / 3)
    assert van_der_corput(2, 3) == pytest.approx(2 / 3)
    assert van_der_corput(3, 3) == pytest.approx(1 / 9)


def test_van_der_corput_rejects_bad_base_and_index():
    with pytest.raises(ValueError):
        van_der_corput(1, 1)
    with pytest.raises(ValueError):
        van_der_corput(-1, 2)


def test_van_der_corput_stays_in_unit_interval():
    assert all(0.0 <= van_der_corput(i, 5) < 1.0 for i in range(200))


# ── Halton ───────────────────────────────────────────────────────────

def test_halton_uses_one_prime_base_per_dimension():
    point = halton(1, 3)
    assert point == pytest.approx([1 / 2, 1 / 3, 1 / 5])
    assert PRIMES[:3] == (2, 3, 5)


def test_halton_zero_dimensions_is_empty_not_an_error():
    assert halton(5, 0) == []


def test_halton_rejects_more_dimensions_than_bases():
    with pytest.raises(ValueError):
        halton(1, len(PRIMES) + 1)


def test_halton_sequence_starts_at_one_by_default():
    # Index 0 is the origin — as a mutation step it means "no change", which
    # would waste a slot in every generation.
    assert halton_sequence(1, 2)[0] != [0.0, 0.0]
    assert halton_sequence(1, 2, start=0)[0] == [0.0, 0.0]


def test_halton_sequence_is_reproducible():
    assert halton_sequence(20, 4) == halton_sequence(20, 4)


def test_halton_sequence_covers_the_interval_better_than_it_clusters():
    # 64 points, 8 buckets: a low-discrepancy sequence fills every bucket.
    values = [p[0] for p in halton_sequence(64, 1)]
    buckets = {int(v * 8) for v in values}
    assert buckets == set(range(8))


def test_halton_sequence_count_is_honoured_and_clamped():
    assert len(halton_sequence(7, 2)) == 7
    assert halton_sequence(0, 2) == []
    assert halton_sequence(-3, 2) == []


# ── scaling helpers ──────────────────────────────────────────────────

def test_scaled_maps_onto_the_requested_range():
    assert scaled(0.0, 2.0, 6.0) == 2.0
    assert scaled(1.0, 2.0, 6.0) == 6.0
    assert scaled(0.5, 2.0, 6.0) == 4.0


def test_scaled_tolerates_inverted_bounds_and_clamps_input():
    assert scaled(0.0, 6.0, 2.0) == 2.0
    assert scaled(5.0, 0.0, 1.0) == 1.0
    assert scaled(-5.0, 0.0, 1.0) == 0.0


def test_signed_gives_a_direction():
    assert signed(0.0) == -1.0
    assert signed(0.5) == 0.0
    assert signed(1.0) == 1.0


# ── hash-indexed choices ─────────────────────────────────────────────

def test_hash_unit_is_in_range_and_stable():
    first = hash_unit("genome", 7)
    assert 0.0 <= first < 1.0
    assert hash_unit("genome", 7) == first


def test_hash_unit_separates_material_unambiguously():
    # Joining with a separator means ("ab","c") and ("a","bc") are different
    # seeds — without it they would collide and two distinct contexts would
    # deterministically make the same "random" choice.
    assert hash_unit("ab", "c") != hash_unit("a", "bc")


def test_hash_index_stays_within_range():
    assert all(0 <= hash_index(5, "x", i) < 5 for i in range(50))


def test_hash_index_rejects_empty_range():
    with pytest.raises(ValueError):
        hash_index(0, "x")


def test_hash_choice_picks_stably_and_survives_empty():
    options = ["a", "b", "c", "d"]
    assert hash_choice(options, "seed") == hash_choice(options, "seed")
    assert hash_choice(options, "seed") in options
    assert hash_choice([], "seed") is None


def test_hash_index_actually_spreads():
    picks = {hash_index(4, "tick", i) for i in range(40)}
    assert picks == {0, 1, 2, 3}


# ── bootstrap ────────────────────────────────────────────────────────

def test_bootstrap_indices_are_in_range_and_reproducible():
    first = bootstrap_indices(10, 32)
    assert len(first) == 32
    assert all(0 <= i < 10 for i in first)
    assert bootstrap_indices(10, 32) == first


def test_bootstrap_replicates_differ():
    assert bootstrap_indices(10, 16, replicate=0) != bootstrap_indices(10, 16, replicate=1)


def test_bootstrap_resamples_with_replacement():
    # 32 draws from 10 items must repeat — sampling without replacement here
    # would make the interval far too narrow.
    assert len(set(bootstrap_indices(10, 32))) < 32


def test_bootstrap_degenerate_inputs_are_empty():
    assert bootstrap_indices(0, 5) == []
    assert bootstrap_indices(5, 0) == []


def test_bootstrap_resample_means_have_with_replacement_spread():
    """The property a bootstrap exists for. Low-discrepancy draws covered
    nearly every index exactly once, so every "resample" was almost the
    original sample — the SD of resample means collapsed to 0.28 against the
    correct ~1.29 and any CI built on it was ~4.5x too narrow. Genuine
    with-replacement resampling must reproduce the theoretical sigma/sqrt(n)
    spread, not merely stay in range."""
    import math
    import statistics

    data = list(range(20))
    means = [statistics.mean(data[i] for i in bootstrap_indices(20, 20, replicate=r))
             for r in range(200)]
    spread = statistics.pstdev(means)
    theoretical = statistics.pstdev(data) / math.sqrt(20)
    assert spread > 0.75 * theoretical, (spread, theoretical)
    assert spread < 1.50 * theoretical, (spread, theoretical)


def test_bootstrap_draws_leave_gaps_like_a_real_resample():
    """With replacement, each of n items is missed with probability
    (1-1/n)^n ~ 36% per resample. Near-complete coverage of the index space is
    the signature of the old quasirandom bug."""
    distinct = [len(set(bootstrap_indices(20, 20, replicate=r)))
                for r in range(100)]
    average = sum(distinct) / len(distinct)
    # True with-replacement expectation is ~12.6 of 20; the broken version
    # measured ~16.6. Split the difference with margin on both sides.
    assert 10.5 < average < 15.0, average
