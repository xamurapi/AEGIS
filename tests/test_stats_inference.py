"""The inference primitives, against analytically known answers (spec M7.7).

These decide whether a behaviour rule activates, whether a strategy is accepted
and whether a discovery is registered. If ``p_value`` is wrong the whole edifice
of "significant" above it is decoration — so every function here is checked
against a value that can be derived by hand or is published, not against
whatever the implementation happens to return.
"""
import math

import pytest

from aegis.util.stats import (
    TTest, benjamini_hochberg, cohens_d, student_t_sf, welch_t,
)


# ── Student's t survival function ────────────────────────────────────

def test_the_t_distribution_is_symmetric_about_zero():
    assert student_t_sf(0.0, 10) == pytest.approx(0.5)
    assert student_t_sf(-2.0, 10) == pytest.approx(student_t_sf(2.0, 10))


def test_one_degree_of_freedom_is_the_cauchy_distribution():
    """For df = 1 the survival function is ``½ − atan(t)/π``, exactly."""
    for t in (0.5, 1.0, 2.0, 5.0):
        expected = 0.5 - math.atan(t) / math.pi
        assert student_t_sf(t, 1) == pytest.approx(expected, abs=1e-9)


def test_two_degrees_of_freedom_have_a_closed_form():
    """For df = 2 the survival function is ``½·(1 − t/√(2+t²))``."""
    for t in (0.5, 1.0, 3.0):
        expected = 0.5 * (1 - t / math.sqrt(2 + t * t))
        assert student_t_sf(t, 2) == pytest.approx(expected, abs=1e-9)


def test_large_degrees_of_freedom_approach_the_normal():
    """At df → ∞ the 97.5th percentile is 1.959964 — the z everything else
    in this package uses."""
    assert student_t_sf(1.959963984540054, 1e7) == pytest.approx(0.025, abs=1e-5)


def test_the_survival_function_falls_monotonically():
    values = [student_t_sf(t, 12) for t in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values, reverse=True)


# ── Welch's t-test ───────────────────────────────────────────────────

def test_a_textbook_comparison_matches_the_published_answer():
    """``t.test(1:5, 6:10)`` gives t = −5, df = 8, p = 0.001053."""
    result = welch_t([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert result.t == pytest.approx(-5.0, abs=1e-9)
    assert result.df == pytest.approx(8.0, abs=1e-9)
    assert result.p_value == pytest.approx(0.001053, abs=1e-5)
    assert result.effect == pytest.approx(-5.0)


def test_the_degrees_of_freedom_follow_welch_satterthwaite():
    """Hand-computed: var 4.5714/8 + 6/8 = 1.32143, df = 13.749."""
    result = welch_t([2, 4, 4, 4, 5, 5, 7, 9], [1, 2, 3, 4, 5, 6, 7, 8])
    assert result.t == pytest.approx(0.43496, abs=1e-5)
    assert result.df == pytest.approx(13.7489, abs=1e-4)
    assert not result.significant(0.05)


def test_identical_samples_are_not_a_difference():
    result = welch_t([1, 2, 3], [1, 2, 3])
    assert result.effect == 0.0
    assert result.p_value == pytest.approx(1.0)
    assert not result.significant()


def test_two_constants_carry_no_uncertainty_to_report():
    """Zero variance in both arms: the difference may be huge, but there is
    nothing here a t-test can speak about, so it declines to."""
    result = welch_t([5, 5, 5], [1, 1, 1])
    assert result.p_value == 1.0
    assert result.effect == 4.0


@pytest.mark.parametrize("a,b", [([], []), ([1], [2]), ([1, 2], []),
                                 ([1], [1, 2, 3])])
def test_an_arm_with_too_little_data_yields_no_confidence(a, b):
    result = welch_t(a, b)
    assert result.p_value == 1.0
    assert not result.significant()


def test_the_effect_is_the_difference_of_means_in_order():
    result = welch_t([10, 12, 14], [1, 3, 5])
    assert result.effect == pytest.approx(9.0)
    assert welch_t([1, 3, 5], [10, 12, 14]).effect == pytest.approx(-9.0)


def test_cohens_d_is_the_effect_in_pooled_standard_deviations():
    """Means 2 apart, both arms with sd 1 → d = 2."""
    a = [1.0, 2.0, 3.0]          # sd = 1
    b = [3.0, 4.0, 5.0]          # sd = 1
    assert cohens_d(a, b) == pytest.approx(-2.0, abs=1e-9)


def test_a_report_knows_its_own_sample_sizes():
    result = welch_t([1, 2, 3, 4], [5, 6])
    assert (result.n_a, result.n_b) == (4, 2)
    assert isinstance(result, TTest)


def test_significance_is_read_against_the_alpha_it_is_given():
    # Means 2 apart, both arms var 2.5 over n = 5: se = 1, t = −2, df = 8,
    # so p = 2·sf(2, 8) = 0.0805.
    result = welch_t([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert result.t == pytest.approx(-2.0, abs=1e-9)
    assert result.df == pytest.approx(8.0, abs=1e-9)
    assert result.p_value == pytest.approx(0.08052, abs=1e-5)
    assert not result.significant(0.05)
    assert result.significant(0.10)


# ── Benjamini–Hochberg ───────────────────────────────────────────────

def test_the_published_example_rejects_exactly_two():
    """Benjamini & Hochberg (1995), α = 0.05: thresholds are i·α/m, and the
    largest i whose p-value clears its threshold is 2."""
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    assert benjamini_hochberg(p_values, 0.05) == [
        True, True, False, False, False, False, False, False]


def test_results_come_back_in_the_order_they_were_given():
    """The procedure sorts internally; the caller's order is what identifies
    which hypothesis is which, so it has to survive."""
    assert benjamini_hochberg([0.205, 0.001, 0.008], 0.05) == [False, True, True]


def test_uniform_noise_yields_no_discoveries():
    """The property the rule miner depends on: p-values spread evenly over
    [0, 1] — which is what pure noise produces — must reject nothing."""
    noise = [(i + 0.5) / 200 for i in range(200)]
    assert not any(benjamini_hochberg(noise, 0.05))


def test_everything_significant_is_kept():
    assert benjamini_hochberg([0.0001, 0.0002, 0.0003], 0.05) == [True] * 3


def test_no_tests_is_no_discoveries():
    assert benjamini_hochberg([], 0.05) == []


def test_a_single_test_is_just_the_threshold():
    assert benjamini_hochberg([0.04], 0.05) == [True]
    assert benjamini_hochberg([0.06], 0.05) == [False]


def test_a_step_up_procedure_keeps_everything_below_the_largest_survivor():
    """The "step up" part: once the largest rank clears, every smaller p-value
    is rejected too — even ones that fail their own threshold. Testing each
    p-value against its own line independently is the classic mis-implementation.
    """
    # rank 3 clears (0.045 <= 3·0.05/3 = 0.05) so all three are rejected,
    # although 0.04 > 2·0.05/3 = 0.0333.
    assert benjamini_hochberg([0.02, 0.04, 0.045], 0.05) == [True, True, True]


def test_a_stricter_alpha_rejects_less():
    p_values = [0.001, 0.008, 0.039]
    assert sum(benjamini_hochberg(p_values, 0.05)) >= \
        sum(benjamini_hochberg(p_values, 0.005))
