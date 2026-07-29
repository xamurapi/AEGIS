"""The three ways this engine asks whether two things are related (spec M7.7).

Each answers a different question, and the difference is the point of having
three. Pearson sees a line. Spearman sees any monotone shape. Mutual
information sees a relationship of any shape at all, which is why a ``U`` — a
squared law, the most ordinary form a real law takes — is the case that
separates them: both correlations score it at approximately zero and MI does
not.

Every expectation is against a value that is known analytically or published,
not against what the implementation returns.
"""
import math

import pytest

from aegis.layers.discovery.statistics import (
    MI_BINS, _chi2_sf, mutual_information, normal_sf, pearson, spearman,
    student_t_sf,
)
from aegis.util.quasirandom import hash_unit


# ── Pearson ──────────────────────────────────────────────────────────

def test_a_perfect_line_correlates_perfectly():
    xs = list(range(1, 21))
    r, p = pearson(xs, [2 * x + 1 for x in xs])
    assert r == pytest.approx(1.0)
    assert p == pytest.approx(0.0)


def test_a_perfect_falling_line_correlates_at_minus_one():
    xs = list(range(1, 21))
    assert pearson(xs, [-3 * x for x in xs])[0] == pytest.approx(-1.0)


def test_a_hand_computed_correlation_matches():
    """Derived from the definition rather than recalled from a table.

    Means 5.5 and 6.4; the cross products sum to 75.0, the squared deviations
    to 82.5 and 72.4. So ``r = 75/√(82.5·72.4) = 75/√5973 = 0.970432…``
    """
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [2, 4, 5, 4, 5, 7, 8, 9, 9, 11]
    assert pearson(xs, ys)[0] == pytest.approx(75.0 / math.sqrt(82.5 * 72.4),
                                               abs=1e-9)
    assert pearson(xs, ys)[0] == pytest.approx(0.970432, abs=1e-6)


def test_a_constant_series_correlates_with_nothing():
    """Reporting zero rather than dividing is the difference between "no
    relationship" and a crash halfway through a scan of two hundred pairs."""
    assert pearson([5] * 10, list(range(10))) == (0.0, 1.0)


def test_too_few_points_are_not_a_correlation():
    assert pearson([1, 2], [1, 2]) == (0.0, 1.0)


def test_unusable_readings_are_dropped_pairwise():
    xs = [1.0, None, 3.0, 4.0, float("nan"), 6.0]
    ys = [2.0, 5.0, 6.0, 8.0, 1.0, 12.0]
    assert pearson(xs, ys)[0] == pytest.approx(1.0)


def test_a_boolean_is_not_a_reading():
    assert pearson([True, False, True], [1.0, 2.0, 3.0]) == (0.0, 1.0)


# ── Spearman ─────────────────────────────────────────────────────────

def test_a_monotone_curve_ranks_perfectly_even_where_pearson_weakens():
    """``y = x³`` is a monotone relationship, and that is what Spearman is for."""
    xs = list(range(1, 21))
    cubes = [x ** 3 for x in xs]
    assert spearman(xs, cubes)[0] == pytest.approx(1.0)
    assert pearson(xs, cubes)[0] < 0.95


def test_a_falling_monotone_curve_ranks_at_minus_one():
    xs = list(range(1, 21))
    assert spearman(xs, [1.0 / x for x in xs])[0] == pytest.approx(-1.0)


def test_ranks_average_over_ties():
    """Without the tie correction a series of repeated readings produces ranks
    that depend on the order they arrived in."""
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2])[0] == pytest.approx(1.0)


def test_spearman_is_not_dragged_by_one_outlier():
    xs = list(range(1, 21))
    ys = [float(x) for x in xs]
    ys[-1] = 10_000.0
    assert spearman(xs, ys)[0] == pytest.approx(1.0)


def test_too_few_points_are_not_a_rank_correlation():
    assert spearman([1, 2], [2, 1]) == (0.0, 1.0)


# ── mutual information ───────────────────────────────────────────────

def test_a_squared_law_is_invisible_to_correlation_and_visible_to_mi():
    """The case that justifies having a third measure. A ``U`` is the ordinary
    shape of a real law, and both correlations score it at zero."""
    xs = [value / 10.0 for value in range(-50, 51)] * 4
    ys = [x * x for x in xs]
    assert abs(pearson(xs, ys)[0]) < 0.05
    information, p_value = mutual_information(xs, ys)
    assert information > 0.5
    assert p_value < 0.01


def test_independent_series_carry_almost_no_information():
    xs = [hash_unit("a", index) for index in range(400)]
    ys = [hash_unit("b", index) for index in range(400)]
    assert mutual_information(xs, ys)[1] > 0.01


def test_information_is_never_negative():
    xs = [hash_unit("a", index) for index in range(200)]
    ys = [hash_unit("b", index) for index in range(200)]
    assert mutual_information(xs, ys)[0] >= 0.0


def test_a_constant_series_carries_no_information():
    assert mutual_information([1.0] * 100, list(range(100))) == (0.0, 1.0)


def test_too_little_data_for_the_bins_is_not_measured():
    """MI is biased upward when bins outnumber the data, so a scan whose
    measure grew with the number of empty bins would find its strongest
    "relationships" in its smallest datasets."""
    assert mutual_information([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == (0.0, 1.0)


def test_the_bin_count_can_be_set():
    xs = [value / 10.0 for value in range(-50, 51)] * 4
    ys = [x * x for x in xs]
    assert mutual_information(xs, ys, bins=3)[0] > 0.0
    assert mutual_information(xs, ys, bins=MI_BINS)[0] > 0.0


def test_fewer_than_two_bins_is_still_two():
    xs = [value / 10.0 for value in range(-50, 51)] * 4
    assert mutual_information(xs, [x * x for x in xs], bins=1)[0] >= 0.0


# ── the χ² tail ──────────────────────────────────────────────────────

@pytest.mark.parametrize("statistic,df,expected", [
    (3.841458820694124, 1, 0.05),
    (6.634896601021213, 1, 0.01),
    (5.991464547107979, 2, 0.05),
    (9.487729036781154, 4, 0.05),
    (18.307038053275146, 10, 0.05),
])
def test_the_chi_square_tail_matches_the_published_critical_values(
        statistic, df, expected):
    assert _chi2_sf(statistic, df) == pytest.approx(expected, abs=1e-4)


def test_a_statistic_of_zero_is_certainly_not_significant():
    assert _chi2_sf(0.0, 1) == 1.0
    assert _chi2_sf(-5.0, 3) == 1.0


def test_the_tail_falls_monotonically():
    values = [_chi2_sf(statistic, 3) for statistic in (0.5, 1, 2, 5, 10, 20)]
    assert values == sorted(values, reverse=True)


def test_a_very_large_statistic_is_vanishingly_unlikely():
    """Past where the series converges usefully the Wilson–Hilferty
    approximation takes over; it has to stay a probability."""
    tail = _chi2_sf(5000.0, 10)
    assert 0.0 <= tail < 1e-6


def test_many_degrees_of_freedom_use_the_approximation_and_stay_sane():
    tail = _chi2_sf(250.0, 300)
    assert 0.0 <= tail <= 1.0
    assert tail > 0.5, "a statistic below its df should not look significant"


def test_the_tail_is_a_probability_everywhere():
    for statistic in (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0):
        for df in (1, 2, 5, 50, 500):
            assert 0.0 <= _chi2_sf(statistic, df) <= 1.0


def test_the_inference_surface_is_re_exported():
    """The engine imports its estimators from one place; a missing re-export
    would be found at the worst possible moment."""
    from aegis.layers.discovery import statistics

    for name in ("welch_t", "mann_whitney_u", "benjamini_hochberg", "bic",
                 "r_squared", "bootstrap_ci", "required_n", "wilson_lower",
                 "two_proportion_z", "compare_samples", "cohens_d"):
        assert callable(getattr(statistics, name)), name


# ── the p-value behind a correlation ─────────────────────────────────
#
# The scan turns every correlation into a p-value and hands the lot to
# Benjamini–Hochberg. If the transform is wrong the ranking is wrong, the
# correction is applied to the wrong numbers, and "significant" stops meaning
# anything — while every correlation still looks perfectly reasonable.

def test_the_correlation_p_value_is_the_t_transform_of_r():
    """``t = r·√((n−2)/(1−r²))`` on ``n−2`` degrees of freedom, two-sided."""
    from aegis.layers.discovery.statistics import _correlation_p

    for r, n in ((0.5, 20), (-0.3, 50), (0.8, 12), (0.05, 200)):
        t = r * math.sqrt((n - 2) / (1.0 - r * r))
        assert _correlation_p(r, n) == pytest.approx(
            min(1.0, 2.0 * student_t_sf(t, n - 2)), abs=1e-12)


def test_a_stronger_correlation_on_the_same_data_is_more_significant():
    from aegis.layers.discovery.statistics import _correlation_p

    assert _correlation_p(0.9, 30) < _correlation_p(0.5, 30) < _correlation_p(0.1, 30)


def test_more_data_makes_the_same_correlation_more_significant():
    from aegis.layers.discovery.statistics import _correlation_p

    assert _correlation_p(0.3, 500) < _correlation_p(0.3, 30)


def test_a_perfect_correlation_is_certain_and_two_points_are_nothing():
    from aegis.layers.discovery.statistics import _correlation_p

    assert _correlation_p(1.0, 50) == 0.0
    assert _correlation_p(-1.0, 50) == 0.0
    assert _correlation_p(0.9, 2) == 1.0


def test_the_covariance_is_of_deviations_not_of_sums():
    """Both factors are deviations from their own mean. Adding either mean
    instead produces a number of the right magnitude and no meaning, and on
    centred data it would look almost right."""
    xs, ys = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]
    assert pearson(xs, ys)[0] == pytest.approx(1.0)

    shifted = [value + 1000.0 for value in ys]
    assert pearson(xs, shifted)[0] == pytest.approx(1.0), \
        "the correlation moved when the data was only shifted"


# ── one bad reading spoils only its own pair ─────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_an_unusable_value_in_either_series_drops_that_pair(bad):
    good_x = [1.0, 2.0, 3.0, 4.0, 5.0]
    good_y = [2.0, 4.0, 6.0, 8.0, 10.0]

    xs = list(good_x); xs[2] = bad
    assert pearson(xs, good_y)[0] == pytest.approx(1.0)

    ys = list(good_y); ys[2] = bad
    assert pearson(good_x, ys)[0] == pytest.approx(1.0)


def test_a_boolean_in_either_series_drops_that_pair():
    good_x = [1.0, 2.0, 3.0, 4.0, 5.0]
    good_y = [2.0, 4.0, 6.0, 8.0, 10.0]

    xs = list(good_x); xs[1] = True
    assert pearson(xs, good_y)[0] == pytest.approx(1.0)

    ys = list(good_y); ys[1] = False
    assert pearson(good_x, ys)[0] == pytest.approx(1.0)


# ── ranks, to the number ─────────────────────────────────────────────

def test_tied_values_take_the_average_of_the_ranks_they_span():
    """Three values tied across positions 2, 3 and 4 all rank 3 — the average
    of the ranks they occupy. Ranks that started at zero, or that averaged the
    wrong span, would still be monotone and would still be wrong."""
    from aegis.layers.discovery.statistics import _ranks

    assert _ranks([10.0, 20.0, 20.0, 20.0, 30.0]) == [1.0, 3.0, 3.0, 3.0, 5.0]
    assert _ranks([5.0, 5.0]) == [1.5, 1.5]
    assert _ranks([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]


def test_ranks_start_at_one():
    from aegis.layers.discovery.statistics import _ranks

    assert min(_ranks([7.0, 8.0, 9.0])) == 1.0


# ── binning ──────────────────────────────────────────────────────────

def test_the_top_of_the_range_falls_in_the_last_bin():
    """Not one past it. An index of ``bins`` would be a bin nobody allocated,
    and the joint table would carry a row that the margins do not."""
    from aegis.layers.discovery.statistics import _bin_index

    assert _bin_index(1.0, 0.0, 1.0, 5) == 4
    assert _bin_index(0.0, 0.0, 1.0, 5) == 0
    assert _bin_index(0.5, 0.0, 1.0, 5) == 2
    assert _bin_index(99.0, 0.0, 1.0, 5) == 4, "a value past the top was not clamped"


def test_a_degenerate_range_collapses_to_one_bin():
    from aegis.layers.discovery.statistics import _bin_index

    assert _bin_index(5.0, 5.0, 5.0, 5) == 0


def test_either_series_being_constant_means_no_information():
    """Both directions. A check that needed *both* to be constant would compute
    a mutual information against a variable that never varies."""
    varying = [float(index % 7) for index in range(100)]
    assert mutual_information([1.0] * 100, varying) == (0.0, 1.0)
    assert mutual_information(varying, [1.0] * 100) == (0.0, 1.0)


# ── mutual information, against a hand-computed value ────────────────

def test_a_perfect_two_by_two_dependency_carries_one_bit():
    """``x`` and ``y`` agree exactly and split evenly, so each cell holds ½ and
    each margin ½: ``I = 2 · ½ · ln(½ / ¼) = ln 2 = 0.693147…`` nats."""
    xs = [float(index % 2) for index in range(200)]
    ys = list(xs)
    information, p_value = mutual_information(xs, ys, bins=2)
    assert information == pytest.approx(math.log(2), abs=1e-9)
    assert p_value == pytest.approx(_chi2_sf(2 * 200 * math.log(2), 1), abs=1e-12)


def test_the_degrees_of_freedom_come_from_the_occupied_bins():
    """``(bins_x − 1)(bins_y − 1)``, multiplied. Adding them instead would give
    two degrees of freedom to every 2×2 table and understate every finding."""
    xs = [float(index % 2) for index in range(200)]
    information, p_value = mutual_information(xs, list(xs), bins=2)
    assert p_value == pytest.approx(_chi2_sf(2 * 200 * information, 1), abs=1e-12)

    xs = [float(index % 3) for index in range(300)]
    information, p_value = mutual_information(xs, list(xs), bins=3)
    assert p_value == pytest.approx(_chi2_sf(2 * 300 * information, 4), abs=1e-12)


# ── the χ² approximation branch ──────────────────────────────────────

def test_the_wilson_hilferty_branch_matches_its_formula():
    """Past where the series converges usefully the cube-root normal
    approximation takes over, and it has to be that approximation rather than
    something with the same shape."""
    from aegis.layers.discovery.statistics import _chi2_sf

    for statistic, df in ((2000.0, 10), (5000.0, 50), (300.0, 250)):
        cube = (statistic / df) ** (1.0 / 3.0)
        z = (cube - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        assert _chi2_sf(statistic, df) == pytest.approx(normal_sf(z), abs=1e-12)


def test_the_series_branch_is_accurate_to_the_published_percentiles():
    """The series has to actually converge, not merely stop. Checked against
    the 1%, 5%, 50% and 90% points of three distributions."""
    from aegis.layers.discovery.statistics import _chi2_sf

    for statistic, df, expected in (
            (0.004, 1, 0.95), (0.455, 1, 0.50), (2.706, 1, 0.10),
            (0.211, 2, 0.90), (1.386, 2, 0.50), (4.605, 2, 0.10),
            (2.204, 5, 0.82), (4.351, 5, 0.50), (9.236, 5, 0.10)):
        assert _chi2_sf(statistic, df) == pytest.approx(expected, abs=2e-3), \
            f"chi2_sf({statistic}, {df})"


def test_an_unusable_reading_is_dropped_rather_than_poisoning_the_result():
    """The check that matters, on data whose correlation is *not* one.

    NaN propagates through every sum, and the clamp at the end turns the
    resulting NaN into 1.0 — so a series that kept its unusable readings comes
    back claiming a perfect correlation. Against perfectly correlated data that
    is indistinguishable from working; against this data it is obvious.
    """
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ys = [2, 4, 5, 4, 5, 7, 8, 9, 9, 11]
    clean = pearson(xs, ys)[0]
    assert clean == pytest.approx(0.970432, abs=1e-6)

    for bad in (float("nan"), float("inf"), float("-inf"), True, None, "text"):
        assert pearson(xs + [bad], ys + [6.0])[0] == pytest.approx(clean), \
            f"{bad!r} in x was not dropped"
        assert pearson(xs + [6.0], ys + [bad])[0] == pytest.approx(clean), \
            f"{bad!r} in y was not dropped"


def test_the_degrees_of_freedom_are_the_product_of_the_occupied_bins_less_one():
    """Checked where the answer is discriminating.

    On a perfect dependency the statistic is enormous and every plausible
    number of degrees of freedom rounds the tail to zero, so that case cannot
    tell 1 from 3. This one is weak enough that the two disagree.
    """
    from aegis.layers.discovery.statistics import _chi2_sf

    # Mostly independent, with a slight tilt: enough association to measure,
    # little enough that the tail lands in a readable range.
    xs, ys = [], []
    for index in range(400):
        x = float(index % 2)
        y = float((index % 2) if index % 13 == 0 else ((index // 2) % 2))
        xs.append(x)
        ys.append(y)

    information, p_value = mutual_information(xs, ys, bins=2)
    statistic = 2 * len(xs) * information
    assert 0.5 < statistic < 30.0, f"statistic {statistic} is not discriminating"
    assert p_value == pytest.approx(_chi2_sf(statistic, 1), abs=1e-12)
    assert p_value != pytest.approx(_chi2_sf(statistic, 3), abs=1e-6), \
        "one and three degrees of freedom are indistinguishable here"


def test_the_series_converges_rather_than_merely_stopping():
    """The convergence test is relative to the running total, and the total is
    minuscule at high degrees of freedom — the first term is ``1/Γ(a+1)``, which
    at 100 degrees of freedom is around 1e-158.

    A threshold that grew as the total shrank would be met on the first
    iteration and the series would stop before it had summed anything, which
    looks like a probability and is not one. A χ² statistic equal to its own
    degrees of freedom sits near the middle of the distribution, so the answer
    has to be near a half.
    """
    from aegis.layers.discovery.statistics import _chi2_sf

    for df in (50, 100, 150, 200):
        tail = _chi2_sf(float(df), df)
        assert 0.40 < tail < 0.60, f"chi2_sf({df}, {df}) = {tail}"


def test_the_series_matches_the_published_percentiles_at_high_degrees_of_freedom():
    from aegis.layers.discovery.statistics import _chi2_sf

    # 95th and 5th percentiles of chi-square with 100 degrees of freedom.
    assert _chi2_sf(124.342, 100) == pytest.approx(0.05, abs=2e-3)
    assert _chi2_sf(77.929, 100) == pytest.approx(0.95, abs=2e-3)
