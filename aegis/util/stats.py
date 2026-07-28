"""Small statistical primitives, implemented rather than imported.

Three contours need the same handful of estimators — the outcome model (M1),
the rule miner (M3) and the discovery engine (M7) — and all three need them to
behave identically, or "significant" would mean something different depending
on who asked. They live here once.

No external dependency: numpy and scipy are optional everywhere in this package
and a statistical claim that silently changes when a library is present is
worse than no claim at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: z for a two-sided 95% interval. The default everywhere in the system, so a
#: "95% confident" statement means the same thing in every contour.
Z_95 = 1.959963984540054


@dataclass
class Welford:
    """Running mean and variance in one pass.

    Used instead of keeping the samples because these accumulate per
    state/action pair, and the number of pairs grows without bound while the
    memory for each must not.

    ``n`` is an *effective weight*, not a count, and is deliberately a float:
    ageing multiplies it by a factor just below one, and truncating that to an
    integer sends a weight of 1 straight to 0. The next observation then starts
    from nothing, so the mean becomes whatever was seen last and the variance
    collapses to zero — the statistics look healthy and describe one sample.
    """

    n: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1.0
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def variance(self) -> float:
        """Sample variance; zero until there are two observations."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    def sd(self) -> float:
        return math.sqrt(self.variance())

    def scale(self, factor: float) -> None:
        """Age the accumulated evidence by a decay factor.

        The mean is left alone and only the *weight* behind it shrinks: a model
        following a drifting world should become less certain of an old
        estimate, not start believing a different one.
        """
        factor = max(0.0, min(1.0, float(factor)))
        self.n *= factor
        self.m2 *= factor

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, data: dict | None) -> Welford:
        """Restore from disk, degrading to "no observations" on anything odd.

        A malformed sub-object must not take the record that contains it down
        with it: losing an action's success counts because its reward statistics
        were corrupt would discard the useful half along with the broken one.
        """
        if not isinstance(data, dict):
            return cls()
        try:
            return cls(n=max(0.0, float(data.get("n", 0.0))),
                       mean=float(data.get("mean", 0.0)),
                       m2=max(0.0, float(data.get("m2", 0.0))))
        except (TypeError, ValueError):
            return cls()


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Confidence interval for a proportion, Wilson's method.

    Wilson rather than the textbook normal interval because the estimates here
    are routinely made from a handful of observations and often sit near 0 or
    1 — exactly where the normal interval produces bounds outside [0, 1] and
    collapses to zero width at the extremes.
    """
    if trials <= 0:
        return 0.0, 1.0
    successes = max(0, min(int(successes), int(trials)))
    n = float(trials)
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_lower(successes: int, trials: int, z: float = Z_95) -> float:
    """Lower bound of the Wilson interval.

    This is what a decision should use when choosing: it answers "how well does
    this work, pessimistically?", so one lucky success out of one cannot look
    like a certainty.
    """
    return wilson_interval(successes, trials, z)[0]


def laplace_rate(successes: int, trials: int, alpha: float = 1.0) -> float:
    """Smoothed success rate — the point estimate, for reporting.

    Add-alpha smoothing keeps an unobserved pair at 0.5 rather than at 0, which
    is the difference between "no evidence" and "evidence of failure".
    """
    if trials < 0 or alpha <= 0:
        return 0.5
    return (successes + alpha) / (trials + 2 * alpha)


def brier_score(probability: float, outcome: bool) -> float:
    """Squared error of one probabilistic forecast.

    Proper: it is minimised only by reporting the probability actually
    believed, so a model cannot improve its score by hedging toward 0.5.
    """
    return (float(probability) - (1.0 if outcome else 0.0)) ** 2


def expected_calibration_error(pairs: list[tuple[float, bool]],
                               bins: int = 10) -> float:
    """How far predicted confidence sits from observed frequency.

    Brier conflates being *right* with being *calibrated*; ECE isolates the
    second. A model that says 70% and is right 70% of the time scores zero here
    even though its Brier score is far from zero — and that is the property the
    planner depends on, because it multiplies these probabilities together.
    """
    if not pairs or bins <= 0:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, outcome in pairs:
        p = max(0.0, min(1.0, float(probability)))
        index = min(bins - 1, int(p * bins))
        buckets[index].append((p, bool(outcome)))

    total = len(pairs)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, o in bucket if o) / len(bucket)
        error += (len(bucket) / total) * abs(confidence - accuracy)
    return error


def calibration_curve(pairs: list[tuple[float, bool]],
                      bins: int = 10) -> list[dict]:
    """Per-bin predicted-vs-observed, for the reliability plot."""
    if bins <= 0:
        return []
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, outcome in pairs or []:
        p = max(0.0, min(1.0, float(probability)))
        buckets[min(bins - 1, int(p * bins))].append((p, bool(outcome)))
    curve = []
    for index, bucket in enumerate(buckets):
        lower = index / bins
        row = {"bin": index, "from": round(lower, 4),
               "to": round(lower + 1.0 / bins, 4), "n": len(bucket)}
        if bucket:
            row["predicted"] = round(sum(p for p, _ in bucket) / len(bucket), 4)
            row["observed"] = round(sum(1 for _, o in bucket if o) / len(bucket), 4)
        else:
            row["predicted"] = None
            row["observed"] = None
        curve.append(row)
    return curve


def exponential_smooth(current: float | None, sample: float,
                       alpha: float = 0.1) -> float:
    """One step of exponential smoothing, seeded by the first sample.

    Seeding with the sample rather than with zero matters: starting at zero
    makes every freshly-created metric look excellent for its first few hundred
    observations, which is precisely when someone is watching it.
    """
    if current is None:
        return float(sample)
    alpha = max(0.0, min(1.0, float(alpha)))
    return (1.0 - alpha) * float(current) + alpha * float(sample)


def safe_log(value: float, floor: float = 1e-12) -> float:
    """``log`` with a floor, for negative log-likelihoods.

    A probability of exactly zero is always a modelling artefact — it means the
    event was not in the table, not that it is impossible — and an infinite
    surprise would swamp every average it entered.
    """
    return math.log(max(float(floor), float(value)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ── comparing two samples ────────────────────────────────────────────

@dataclass(frozen=True)
class TTest:
    """Outcome of a two-sample comparison."""

    t: float
    df: float
    p_value: float
    effect: float           # difference of means, a − b
    cohens_d: float
    n_a: int
    n_b: int

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def welch_t(sample_a, sample_b) -> TTest:
    """Welch's t-test: two means, without assuming equal variance.

    Welch rather than Student because the two arms here are never symmetric —
    a rule fires on a subset of ticks chosen by the rule itself, so the arm it
    creates is both smaller and differently spread than the one it left behind.
    Student's pooled variance would understate the standard error in exactly
    that situation and turn noise into a "significant" rule.

    An arm with fewer than two observations, or two arms with no variance at
    all, yield ``p = 1.0``: there is nothing here to be confident about.
    """
    a = [float(v) for v in sample_a]
    b = [float(v) for v in sample_b]
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return TTest(0.0, 0.0, 1.0, mean(a) - mean(b), 0.0, n_a, n_b)

    mean_a, mean_b = mean(a), mean(b)
    var_a = sum((v - mean_a) ** 2 for v in a) / (n_a - 1)
    var_b = sum((v - mean_b) ** 2 for v in b) / (n_b - 1)
    effect = mean_a - mean_b

    standard_error_squared = var_a / n_a + var_b / n_b
    if standard_error_squared <= 0.0:
        return TTest(0.0, 0.0, 1.0, effect, 0.0, n_a, n_b)

    t = effect / math.sqrt(standard_error_squared)
    # Welch–Satterthwaite degrees of freedom.
    denominator = ((var_a / n_a) ** 2 / (n_a - 1)
                   + (var_b / n_b) ** 2 / (n_b - 1))
    df = standard_error_squared ** 2 / denominator if denominator > 0 else 1.0

    pooled_sd = math.sqrt((var_a + var_b) / 2)
    d = effect / pooled_sd if pooled_sd > 0 else 0.0
    return TTest(t, df, student_t_sf(abs(t), df) * 2, effect, d, n_a, n_b)


def student_t_sf(t: float, df: float) -> float:
    """One-sided survival function of Student's t — P(T > t).

    Computed through the regularised incomplete beta function rather than
    imported: scipy is not a dependency here, and a p-value that changed
    depending on whether it happened to be installed would make "significant"
    an environment-dependent claim.
    """
    t, df = abs(float(t)), max(1e-9, float(df))
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b), by continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    # The fraction converges fast for x < (a+1)/(a+b+2); outside that range the
    # symmetry I_x(a,b) = 1 − I_{1−x}(b,a) puts it back inside.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    return front * _beta_cf(a, b, x) / a


def _beta_cf(a: float, b: float, x: float, iterations: int = 200,
             epsilon: float = 1e-12) -> float:
    """Lentz's algorithm for the beta continued fraction."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def normal_sf(z: float) -> float:
    """One-sided tail of the standard normal — P(Z > z)."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def mann_whitney_u(sample_a, sample_b) -> TTest:
    """Rank-based comparison of two samples, with a tie correction.

    Distribution-free, so it stays defined exactly where Welch's t does not:
    when a sample has no variance at all. Repeated identical readings are not
    "no information" — two arms of constant, different values are as separated
    as data can be — and a test that called that insignificant would make the
    strongest evidence the weakest result.

    Reported in the same shape as :func:`welch_t` so a caller can swap between
    them without special-casing. ``t`` carries the normal-approximation z.
    """
    a = [float(v) for v in sample_a]
    b = [float(v) for v in sample_b]
    n_a, n_b = len(a), len(b)
    effect = mean(a) - mean(b)
    if n_a < 1 or n_b < 1:
        return TTest(0.0, 0.0, 1.0, effect, 0.0, n_a, n_b)

    combined = sorted((value, group) for group, values in
                      ((0, a), (1, b)) for value in values)
    ranks: list[float] = [0.0] * len(combined)
    index = 0
    tie_correction = 0.0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average_rank
        group_size = end - index + 1
        if group_size > 1:
            tie_correction += group_size ** 3 - group_size
        index = end + 1

    rank_sum_a = sum(rank for rank, (_, group) in zip(ranks, combined) if group == 0)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    total = n_a * n_b
    mean_u = total / 2.0
    n = n_a + n_b
    variance = total / 12.0 * ((n + 1) - tie_correction / (n * (n - 1))) \
        if n > 1 else 0.0
    if variance <= 0:
        return TTest(0.0, 0.0, 1.0, effect, 0.0, n_a, n_b)

    # Continuity correction: U is discrete, the normal approximation is not.
    z = (abs(u_a - mean_u) - 0.5) / math.sqrt(variance)
    z = max(0.0, z)
    signed_z = z if u_a >= mean_u else -z
    return TTest(signed_z, float(n - 2), min(1.0, 2.0 * normal_sf(z)),
                 effect, 0.0, n_a, n_b)


def compare_samples(sample_a, sample_b) -> TTest:
    """Welch's t where it is defined, rank-based where it is not.

    The behaviour policy compares reward samples, and a rule whose effect is
    perfectly consistent produces two arms with zero variance. Welch's t is
    undefined there — the standard error is zero — and reporting ``p = 1`` for
    it would mean the cleanest possible evidence never activated a rule, while
    noisier evidence did. So a degenerate arm falls through to Mann–Whitney,
    which needs no variance estimate and answers the same question.

    The switch is on the *shape of the data*, never on the result: which test
    runs is decided before either has been computed.
    """
    a = [float(v) for v in sample_a]
    b = [float(v) for v in sample_b]
    if len(a) < 2 or len(b) < 2:
        return welch_t(a, b)
    mean_a, mean_b = mean(a), mean(b)
    var_a = sum((v - mean_a) ** 2 for v in a)
    var_b = sum((v - mean_b) ** 2 for v in b)
    if var_a <= 0.0 and var_b <= 0.0:
        return mann_whitney_u(a, b)
    return welch_t(a, b)


def two_proportion_z(successes_a: int, trials_a: int,
                     successes_b: int, trials_b: int) -> TTest:
    """Pooled two-proportion z-test — the right tool for success counts.

    The rule miner compares "how often does this work here" against "how often
    does it work elsewhere", and those are proportions, not measurements. Under
    the null the two groups share one rate, so the variance is pooled — which
    also means an all-fail group against an all-succeed group has a perfectly
    well-defined statistic, where a t-test on the 0/1 indicators would divide by
    a zero standard error and report no evidence at all.
    """
    trials_a, trials_b = int(trials_a), int(trials_b)
    if trials_a < 1 or trials_b < 1:
        return TTest(0.0, 0.0, 1.0, 0.0, 0.0, trials_a, trials_b)
    successes_a = max(0, min(int(successes_a), trials_a))
    successes_b = max(0, min(int(successes_b), trials_b))

    rate_a = successes_a / trials_a
    rate_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    variance = pooled * (1.0 - pooled) * (1.0 / trials_a + 1.0 / trials_b)
    effect = rate_a - rate_b
    if variance <= 0.0:
        # Both groups are entirely successes or entirely failures: identical
        # rates, so there is nothing to distinguish.
        return TTest(0.0, 0.0, 1.0, effect, 0.0, trials_a, trials_b)

    z = effect / math.sqrt(variance)
    d = 2.0 * (math.asin(math.sqrt(rate_a)) - math.asin(math.sqrt(rate_b)))
    return TTest(z, float(trials_a + trials_b - 2), min(1.0, 2.0 * normal_sf(abs(z))),
                 effect, d, trials_a, trials_b)


def cohens_d(sample_a, sample_b) -> float:
    """Standardised difference of means — effect size, not significance.

    Kept separate from the p-value because they answer different questions, and
    a rule needs both: "unlikely to be chance" and "big enough to bother
    changing behaviour over".
    """
    return welch_t(sample_a, sample_b).cohens_d


# ── many comparisons at once ─────────────────────────────────────────

def benjamini_hochberg(p_values, alpha: float = 0.05) -> list[bool]:
    """Which of many tests survive false-discovery-rate control.

    The rule miner tests hundreds of (state, action) combinations per
    generation, and at α = 0.05 roughly one in twenty pure-noise combinations
    clears an uncorrected threshold. Without this, a system that mined enough
    would always find "evidence" and would fill its policy with rules describing
    nothing. Benjamini–Hochberg rather than Bonferroni because the goal is to
    keep the *proportion* of false rules low while still discovering real ones —
    Bonferroni at this many tests would reject everything.

    Returns one boolean per input, in the input's order.
    """
    values = [float(p) for p in p_values]
    total = len(values)
    if total == 0:
        return []
    order = sorted(range(total), key=lambda i: (values[i], i))
    survives = [False] * total
    largest_k = 0
    for rank, index in enumerate(order, start=1):
        if values[index] <= alpha * rank / total:
            largest_k = rank
    for rank, index in enumerate(order, start=1):
        if rank <= largest_k:
            survives[index] = True
    return survives


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def trend(values, flat_band: float = 0.0) -> str:
    """Direction of a short series: ``"up"``, ``"down"`` or ``"flat"``.

    The band is what stops noise from reading as a trend; without it a series
    that wobbles in the fourth decimal alternates direction every tick.
    """
    values = [float(v) for v in values]
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    if abs(delta) < flat_band:
        return "flat"
    return "up" if delta > 0 else "down"
