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
    bic, bootstrap_ci, r_squared, required_n,
    TTest, benjamini_hochberg, cohens_d, compare_samples, mann_whitney_u,
    normal_sf, student_t_sf, welch_t,
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


# ── Mann–Whitney U (M7.7) ────────────────────────────────────────────
#
# Required by the spec alongside Welch's t, and reached in exactly the place
# Welch's t cannot go: two arms with no variance at all. Every expectation below
# is derived by hand from the definition of U rather than read off the
# implementation.

def test_two_completely_separated_samples_give_u_of_zero():
    """a = 1..5, b = 6..10. Every b outranks every a, so the ranks of a are
    1..5, their sum is 15, and U_a = 15 − 5·6/2 = 0 — the extreme of the
    statistic. With no ties the variance is n_a·n_b·(n+1)/12 = 25·11/12, and the
    continuity-corrected z is (12.5 − 0.5)/√22.9166… = 2.50686…
    """
    result = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    variance = 25 * 11 / 12
    expected_z = (12.5 - 0.5) / math.sqrt(variance)
    assert result.t == pytest.approx(-expected_z)
    assert result.p_value == pytest.approx(2.0 * normal_sf(expected_z))
    assert result.effect == pytest.approx(-5.0)


def test_the_sign_says_which_sample_ranked_higher():
    low, high = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert mann_whitney_u(high, low).t > 0
    assert mann_whitney_u(low, high).t < 0


def test_swapping_the_samples_does_not_change_the_p_value():
    a, b = [3.0, 1.0, 4.0, 1.0, 5.0], [9.0, 2.0, 6.0, 5.0, 3.0]
    assert mann_whitney_u(a, b).p_value == pytest.approx(
        mann_whitney_u(b, a).p_value)


def test_two_constant_arms_at_different_levels_are_separated_evidence():
    """The case the docstring stands on. Each arm is three identical readings,
    so there are two tie groups of three: the correction is 2·(3³−3) = 48, the
    variance is 9/12·(7 − 48/30) = 4.05, U_a = 0, and z = (4.5 − 0.5)/√4.05.

    Welch's t is undefined here. A comparison that called the cleanest possible
    evidence insignificant would make consistency count for less than noise.
    """
    result = mann_whitney_u([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    expected_z = 4.0 / math.sqrt(4.05)
    assert result.t == pytest.approx(-expected_z)
    assert result.p_value == pytest.approx(2.0 * normal_sf(expected_z))
    assert result.p_value < 0.05


def test_the_tie_correction_is_applied_and_not_merely_computed():
    """Ties shrink the variance, so a tied comparison has a *larger* z than the
    same comparison would show if the correction were dropped. Checked against
    the uncorrected variance, which is what a missing correction would produce.
    """
    result = mann_whitney_u([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    uncorrected = 3 * 3 * (6 + 1) / 12.0
    assert abs(result.t) > 4.0 / math.sqrt(uncorrected)


def test_two_identical_arms_carry_no_evidence():
    """Every value tied with every other: the corrected variance is exactly
    zero, and there is nothing to report but "no difference"."""
    result = mann_whitney_u([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert result.p_value == 1.0
    assert result.t == 0.0
    assert result.effect == 0.0


def test_an_empty_sample_is_not_a_comparison():
    result = mann_whitney_u([], [1.0, 2.0])
    assert result.p_value == 1.0
    assert result.n_a == 0 and result.n_b == 2


def test_one_reading_each_cannot_reach_significance():
    """n_a = n_b = 1 gives a variance of 2/12 and |U − mean| = 0.5, which the
    continuity correction takes to zero. Two single observations are not
    evidence, and the statistic says so on its own."""
    assert mann_whitney_u([1.0], [2.0]).p_value == pytest.approx(1.0)


# ── compare_samples: which test runs, decided on shape ───────────────

def test_a_normal_comparison_goes_to_welch():
    a, b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 9.0]
    assert compare_samples(a, b) == welch_t(a, b)


def test_two_variance_free_arms_fall_through_to_the_rank_test():
    """Welch's t divides by a standard error of zero here and reports nothing.
    The rank test needs no variance estimate and answers the same question."""
    a, b = [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]
    assert compare_samples(a, b) == mann_whitney_u(a, b)
    assert compare_samples(a, b).p_value < 0.05
    assert welch_t(a, b).p_value == 1.0


def test_one_variance_free_arm_is_still_welch():
    """The fallback is for the case Welch's t cannot answer at all. With one arm
    varying the standard error is positive and the t-test is defined, so the
    switch must not fire — otherwise which test ran would depend on noise."""
    a, b = [1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0]
    assert compare_samples(a, b) == welch_t(a, b)


def test_a_sample_too_short_to_have_a_variance_goes_to_welch():
    assert compare_samples([1.0], [2.0, 3.0]) == welch_t([1.0], [2.0, 3.0])


# ── model fit and study design (M7.7) ────────────────────────────────

def test_a_perfect_model_explains_everything():
    assert r_squared([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_predicting_the_mean_explains_nothing():
    assert r_squared([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)


def test_a_model_worse_than_the_mean_reports_a_negative_r_squared():
    """RSS = 2² + 0 + 2² = 8 against TSS = 1 + 0 + 1 = 2, so R² = 1 − 4 = −3.

    Not clamped: "worse than guessing the average" and "no better than it" are
    different findings, and the symbolic search picks between formulas on this
    number.
    """
    assert r_squared([1, 2, 3], [3, 2, 1]) == pytest.approx(-3.0)


def test_constant_data_is_explained_only_by_reproducing_it():
    assert r_squared([5, 5, 5], [5, 5, 5]) == pytest.approx(1.0)
    assert r_squared([5, 5, 5], [5, 5, 4]) == pytest.approx(0.0)


@pytest.mark.parametrize("actual,predicted", [([], []), ([1.0], [1.0]),
                                              ([1, 2, 3], [1, 2])])
def test_too_little_or_mismatched_data_explains_nothing(actual, predicted):
    assert r_squared(actual, predicted) == 0.0


def test_the_bic_is_the_textbook_expression():
    """n·ln(RSS/n) + k·ln(n) = 10·ln(0.1) + 2·ln(10) = −18.42068…"""
    assert bic(1.0, 10, 2) == pytest.approx(
        10 * math.log(0.1) + 2 * math.log(10))


def test_the_bic_charges_for_every_extra_parameter():
    """The property the symbolic search depends on: same fit, more nodes, worse
    score — otherwise nothing stops a formula from reproducing noise exactly."""
    assert bic(1.0, 100, 5) > bic(1.0, 100, 2)


def test_a_better_fit_scores_lower():
    assert bic(0.5, 100, 3) < bic(5.0, 100, 3)


def test_the_penalty_for_complexity_grows_with_the_data():
    """ln(n) is the multiplier, so one extra parameter costs more at n = 1000
    than at n = 10. More evidence should make an elaborate explanation harder
    to justify, not easier."""
    small = bic(1.0, 10, 3) - bic(1.0, 10, 2)
    large = bic(1.0, 1000, 3) - bic(1.0, 1000, 2)
    assert large > small


def test_no_observations_cannot_be_scored():
    assert bic(1.0, 0, 2) == float("inf")


def test_a_bootstrap_interval_is_the_same_on_every_run():
    """The property the whole scheme exists for (§3.1). An interval that moved
    between runs would make replication — which is what the interval supports —
    impossible to test."""
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_a_bootstrap_interval_brackets_the_statistic():
    values = [float(v) for v in range(1, 21)]
    low, high = bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high


def test_a_wider_confidence_level_gives_a_wider_interval():
    values = [float(v) for v in range(1, 41)]
    narrow = bootstrap_ci(values, confidence=0.50)
    wide = bootstrap_ci(values, confidence=0.99)
    assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])


def test_one_observation_is_its_own_interval():
    assert bootstrap_ci([7.0]) == (7.0, 7.0)


def test_no_observations_is_an_empty_interval():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_a_bootstrap_takes_the_statistic_it_is_given():
    values = [1.0, 2.0, 3.0, 100.0]
    assert bootstrap_ci(values, statistic=max)[1] == 100.0


def test_the_required_sample_size_matches_the_published_table():
    """The standard two-sample figures at α = 0.05, power = 0.8: 63 per arm for
    a medium effect and 25 for a large one under the normal approximation."""
    assert required_n(0.5) == 63
    assert required_n(0.8) == 25


def test_a_smaller_effect_needs_more_observations():
    assert required_n(0.2) > required_n(0.5) > required_n(1.0)


def test_more_power_costs_more_observations():
    assert required_n(0.5, power=0.95) > required_n(0.5, power=0.80)


def test_a_stricter_alpha_costs_more_observations():
    assert required_n(0.5, alpha=0.01) > required_n(0.5, alpha=0.05)


def test_an_effect_of_zero_can_never_be_established():
    """No sample size detects nothing. Reporting a finite n here would let a
    preregistration commit to a study that cannot succeed."""
    assert required_n(0.0) >= 1_000_000


def test_the_direction_of_an_effect_does_not_change_its_cost():
    assert required_n(-0.5) == required_n(0.5)


def test_a_bootstrap_handles_more_observations_than_there_are_prime_bases():
    """A resample needs one coordinate per observation, and quasirandom points
    run out of dimensions long before a sample runs out of members. Two hundred
    observations is an ordinary experiment, not an edge case."""
    values = [float(v) for v in range(200)]
    low, high = bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high
