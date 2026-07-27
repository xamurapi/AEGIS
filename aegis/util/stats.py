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
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
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
        self.n = int(self.n * factor)
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
            return cls(n=max(0, int(data.get("n", 0))),
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
