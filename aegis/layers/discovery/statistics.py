"""Association measures, and the inference surface the engine works against
(spec M7.7).

The general estimators — Welch, Mann–Whitney, Wilson, Benjamini–Hochberg, BIC,
R², bootstrap, power — live in :mod:`aegis.util.stats`, because the outcome
model and the rule miner need the same ones and "significant" must not mean two
different things in two contours. They are re-exported here so the discovery
engine has one import surface.

What is genuinely new here is the three ways this engine asks "are these two
things related at all", and each answers a different question:

* **Pearson** — is one a straight-line function of the other. Cheap, and the
  right test when the relationship really is linear.
* **Spearman** — is one a *monotone* function of the other. Correlates the ranks
  instead of the values, so it sees ``y = x³`` and ``y = log x`` where Pearson
  sees a weakened line, and it is not dragged around by a single outlier.
* **Mutual information** — is one informative about the other *at all*. On
  binned data, so it sees a U-shape, which both correlations score at
  approximately zero. The engine looks for laws, and a law shaped like ``x²``
  is exactly the case a correlation scan misses.

All three come back with a p-value, because a scan produces hundreds of these
and the Benjamini–Hochberg step that follows needs p-values, not scores.
"""
from __future__ import annotations

import math

from aegis.util.stats import (  # noqa: F401  — re-exported inference surface
    TTest, Welford, benjamini_hochberg, bic, bootstrap_ci, clamp, cohens_d,
    compare_samples, mann_whitney_u, mean, normal_sf, r_squared, required_n,
    student_t_sf, two_proportion_z, welch_t, wilson_interval, wilson_lower,
)

#: Bins for the mutual-information estimate. Few enough that a few hundred
#: points still fill them — MI is biased upward when bins outnumber the data,
#: and a scan whose measure grows with the number of empty bins would find its
#: strongest "relationships" in its smallest datasets.
MI_BINS = 5


def _clean_pairs(xs, ys) -> tuple[list[float], list[float]]:
    """Positions where both series carry a usable number."""
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        if isinstance(x, bool) or isinstance(y, bool):
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf"):
            continue
        out_x.append(float(x))
        out_y.append(float(y))
    return out_x, out_y


def _correlation_p(r: float, n: int) -> float:
    """Two-sided p-value for a correlation, via the t transform.

    ``t = r·√((n−2)/(1−r²))`` on ``n−2`` degrees of freedom. A perfect
    correlation makes the denominator zero, which is a p-value of zero rather
    than a division error — with three points that is a statement about how
    little data there is, and the support check upstream is what guards it.
    """
    n = int(n)
    if n < 3:
        return 1.0
    r = clamp(float(r), -1.0, 1.0)
    if abs(r) >= 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return min(1.0, 2.0 * student_t_sf(t, n - 2))


def pearson(xs, ys) -> tuple[float, float]:
    """Linear correlation and its p-value. ``(0.0, 1.0)`` when undefined."""
    x, y = _clean_pairs(xs, ys)
    n = len(x)
    if n < 3:
        return (0.0, 1.0)
    mean_x, mean_y = mean(x), mean(y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0.0 or var_y <= 0.0:
        # A constant series carries no information about anything. Reporting
        # zero rather than dividing is the difference between "no relationship"
        # and a crash halfway through a scan of two hundred pairs.
        return (0.0, 1.0)
    r = cov / math.sqrt(var_x * var_y)
    return (clamp(r, -1.0, 1.0), _correlation_p(r, n))


def _ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged — the standard correction."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and \
                values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(xs, ys) -> tuple[float, float]:
    """Rank correlation and its p-value — monotone, not merely linear."""
    x, y = _clean_pairs(xs, ys)
    if len(x) < 3:
        return (0.0, 1.0)
    return pearson(_ranks(x), _ranks(y))


def _bin_index(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        return 0
    scaled = (value - low) / (high - low) * bins
    return int(min(bins - 1, max(0, int(scaled))))


def mutual_information(xs, ys, bins: int = MI_BINS) -> tuple[float, float]:
    """Mutual information in nats, with a p-value from the G-test.

    ``G = 2·N·I`` is distributed as χ² with ``(bins_x − 1)(bins_y − 1)`` degrees
    of freedom under independence, which is what turns a score into something
    the false-discovery step can use.

    Only occupied bins count toward the degrees of freedom. A dataset that fills
    three of five bins has the degrees of freedom of a three-bin table, and
    charging it for the empty ones would make every sparse variable look
    significant.
    """
    x, y = _clean_pairs(xs, ys)
    n = len(x)
    bins = max(2, int(bins))
    if n < bins * 2:
        return (0.0, 1.0)

    low_x, high_x = min(x), max(x)
    low_y, high_y = min(y), max(y)
    if high_x <= low_x or high_y <= low_y:
        return (0.0, 1.0)

    joint: dict[tuple[int, int], int] = {}
    margin_x: dict[int, int] = {}
    margin_y: dict[int, int] = {}
    for a, b in zip(x, y):
        i = _bin_index(a, low_x, high_x, bins)
        j = _bin_index(b, low_y, high_y, bins)
        joint[(i, j)] = joint.get((i, j), 0) + 1
        margin_x[i] = margin_x.get(i, 0) + 1
        margin_y[j] = margin_y.get(j, 0) + 1

    information = 0.0
    for (i, j), count in joint.items():
        p_joint = count / n
        p_x = margin_x[i] / n
        p_y = margin_y[j] / n
        information += p_joint * math.log(p_joint / (p_x * p_y))
    information = max(0.0, information)

    df = (len(margin_x) - 1) * (len(margin_y) - 1)
    if df <= 0:
        return (information, 1.0)
    return (information, _chi2_sf(2.0 * n * information, df))


def _chi2_sf(statistic: float, df: int) -> float:
    """Upper tail of the χ² distribution — P(X > statistic).

    Series expansion of the regularised lower incomplete gamma for the small
    statistics this sees, and the Wilson–Hilferty cube-root normal
    approximation past where the series stops converging usefully. Both are
    standard; neither needs scipy.
    """
    statistic = max(0.0, float(statistic))
    df = max(1, int(df))
    if statistic <= 0.0:
        return 1.0
    if statistic > 1000.0 or df > 200:
        # Wilson–Hilferty: (X/df)^(1/3) is approximately normal.
        cube = (statistic / df) ** (1.0 / 3.0)
        z = (cube - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        return normal_sf(z)

    a = df / 2.0
    x = statistic / 2.0
    term = 1.0 / math.gamma(a + 1.0)
    total = term
    for k in range(1, 500):
        term *= x / (a + k)
        total += term
        if term < 1e-16 * total:
            break
    lower = total * math.exp(-x + a * math.log(x))
    return clamp(1.0 - lower, 0.0, 1.0)
