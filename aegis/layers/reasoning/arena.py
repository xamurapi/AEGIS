"""Judging a proposed strategy before it gets any traffic (spec M6.8).

A candidate has to clear three gates *at once*, and the conjunction is the whole
design:

* **It helps.** ``Δpass`` on the held-out half of the weak class is at least
  ``REASON_MIN_GAIN``. Measured on problems that were not used to find the
  weakness or to debug the candidate — otherwise the gain is the candidate
  fitting the examples it was shown.
* **It breaks nothing.** ``Δpass`` on a general set is not worse than −0.01. A
  strategy that wins its class by losing everywhere else is a strategy that
  should not exist, and without this gate the system would accumulate them one
  weakness at a time.
* **It is affordable.** Its cost is within ``REASON_COST_TOLERANCE`` of the
  incumbent's. Wrapping everything in a five-way vote will usually buy a point
  of accuracy; the cost gate is what stops that being a free win.

Passing all three does not make a strategy active. It makes it a **trial**: it
gets every k-th request of its class, and after ``REASON_TRIAL_N`` applications
its accumulated record is compared with the incumbent's. That second comparison
is on live traffic, which is the only place a strategy can be wrong in a way an
arena run cannot see.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning.dsl import cost_of
from aegis.util.stats import two_proportion_z, wilson_lower

logger = logging.getLogger("aegis.reasoning")

#: Problems per arena run, per set. Enough that a five-point difference is not
#: two lucky tasks, small enough that a generation is seconds rather than minutes.
ARENA_TASKS = 48

#: Where the arena's problems come from. Disjoint from the working queue (which
#: walks up from zero) and from the held-out score (which walks down from ten
#: million), so a candidate is never judged on something it was tuned on.
TRAIN_BASE = 1_000_000
HOLDOUT_BASE = 2_000_000
REGRESSION_BASE = 3_000_000


@dataclass
class Verdict:
    """What the arena concluded, and every number behind it."""

    accepted: bool = False
    reasons: list[str] = field(default_factory=list)
    train_gain: float = 0.0
    holdout_gain: float = 0.0
    overall_delta: float = 0.0
    cost_ratio: float = 1.0
    candidate_holdout: float = 0.0
    incumbent_holdout: float = 0.0
    candidate_overall: float = 0.0
    incumbent_overall: float = 0.0
    confident_error_delta: float = 0.0
    p_value: float = 1.0

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "reasons": list(self.reasons),
                "train_gain": round(self.train_gain, 4),
                "holdout_gain": round(self.holdout_gain, 4),
                "overall_delta": round(self.overall_delta, 4),
                "cost_ratio": round(self.cost_ratio, 4),
                "candidate_holdout": round(self.candidate_holdout, 4),
                "incumbent_holdout": round(self.incumbent_holdout, 4),
                "candidate_overall": round(self.candidate_overall, 4),
                "incumbent_overall": round(self.incumbent_overall, 4),
                "confident_error_delta": round(self.confident_error_delta, 4),
                "p_value": round(self.p_value, 6)}


@dataclass
class _Score:
    solved: int = 0
    total: int = 0
    confident_errors: int = 0
    elapsed_ms: float = 0.0

    @property
    def rate(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def confident_error_rate(self) -> float:
        return self.confident_errors / self.total if self.total else 0.0


class Arena:
    """Runs a candidate against the incumbent on three sets and judges it."""

    def __init__(self, interpreter, *, min_gain: float | None = None,
                 cost_tolerance: float | None = None,
                 regression_limit: float = 0.01, tasks: int = ARENA_TASKS,
                 budget: int | None = None):
        self.interpreter = interpreter
        self.min_gain = float(cfg.REASON_MIN_GAIN if min_gain is None else min_gain)
        self.cost_tolerance = float(
            cfg.REASON_COST_TOLERANCE if cost_tolerance is None else cost_tolerance)
        self.regression_limit = float(regression_limit)
        self.tasks = int(tasks)
        self.budget = budget
        self.runs = 0
        self.accepted = 0

    # ── the task sets ────────────────────────────────────────────────

    def sets_for(self, family: str) -> dict[str, list]:
        """Train, held-out and regression sets for one weak class.

        The first two are the weak family; the third spans everything, because
        "does not break anything else" is a claim about everything else.
        """
        half = max(1, self.tasks // 2)
        if family and family in bench.FAMILIES:
            train = bench.build_family(family, half, start=TRAIN_BASE)
            holdout = bench.build_family(family, half, start=HOLDOUT_BASE)
        else:
            train = bench.benchmark(half, start=TRAIN_BASE)
            holdout = bench.benchmark(half, start=HOLDOUT_BASE)
        return {"train": train, "holdout": holdout,
                "regression": bench.benchmark(self.tasks, start=REGRESSION_BASE)}

    # ── scoring ──────────────────────────────────────────────────────

    def score(self, strategy, tasks) -> _Score:
        result = _Score()
        for task in tasks:
            trace = self.interpreter.run(strategy, task, budget=self.budget)
            result.total += 1
            result.elapsed_ms += trace.elapsed_ms
            if trace.solved:
                result.solved += 1
            elif not trace.abstained and trace.answer is not None:
                result.confident_errors += 1
        return result

    # ── the verdict ──────────────────────────────────────────────────

    def evaluate(self, candidate, weakness, incumbent) -> Verdict:
        """Judge one candidate against the strategy it would displace."""
        self.runs += 1
        family = getattr(weakness, "family", "") or ""
        sets = self.sets_for(family)
        steps = getattr(candidate, "steps", candidate)

        candidate_train = self.score(steps, sets["train"])
        incumbent_train = self.score(incumbent, sets["train"])
        candidate_holdout = self.score(steps, sets["holdout"])
        incumbent_holdout = self.score(incumbent, sets["holdout"])
        candidate_overall = self.score(steps, sets["regression"])
        incumbent_overall = self.score(incumbent, sets["regression"])

        verdict = Verdict(
            train_gain=candidate_train.rate - incumbent_train.rate,
            holdout_gain=candidate_holdout.rate - incumbent_holdout.rate,
            overall_delta=candidate_overall.rate - incumbent_overall.rate,
            cost_ratio=self._cost_ratio(steps, incumbent),
            candidate_holdout=candidate_holdout.rate,
            incumbent_holdout=incumbent_holdout.rate,
            candidate_overall=candidate_overall.rate,
            incumbent_overall=incumbent_overall.rate,
            confident_error_delta=(candidate_overall.confident_error_rate
                                   - incumbent_overall.confident_error_rate),
            p_value=two_proportion_z(candidate_holdout.solved,
                                     candidate_holdout.total,
                                     incumbent_holdout.solved,
                                     incumbent_holdout.total).p_value)

        reasons = []
        if verdict.holdout_gain < self.min_gain:
            reasons.append(
                f"held-out gain {verdict.holdout_gain:+.3f} below the required "
                f"{self.min_gain:+.3f}")
        if verdict.overall_delta < -self.regression_limit:
            reasons.append(
                f"the general benchmark falls {verdict.overall_delta:+.3f}, "
                f"past the {-self.regression_limit:+.3f} limit")
        if verdict.cost_ratio > self.cost_tolerance:
            reasons.append(
                f"costs {verdict.cost_ratio:.2f}x the incumbent, past "
                f"{self.cost_tolerance:.2f}x")
        verdict.reasons = reasons
        verdict.accepted = not reasons
        if verdict.accepted:
            self.accepted += 1
        return verdict

    def _cost_ratio(self, steps, incumbent) -> float:
        """Priced from the declared cost of the steps, before either runs.

        Wall time would be the honest measure of what a strategy costs, and it
        is also the measure that changes with machine load — a candidate judged
        on a busy minute would be rejected for the machine's reasons, not its
        own. The declared cost is a property of the strategy.
        """
        incumbent_steps = getattr(incumbent, "steps", incumbent)
        mine = cost_of(steps)
        theirs = cost_of(incumbent_steps)
        mine_total = mine.llm_tokens + mine.wall_ms
        theirs_total = theirs.llm_tokens + theirs.wall_ms
        if theirs_total <= 0:
            return 1.0 if mine_total <= 0 else float("inf")
        return mine_total / theirs_total

    def status(self) -> dict:
        return {"runs": self.runs, "accepted": self.accepted,
                "min_gain": self.min_gain,
                "cost_tolerance": self.cost_tolerance,
                "regression_limit": self.regression_limit,
                "tasks": self.tasks}


def conclude_trial(trial, incumbent, family: str, *, min_uses: int | None = None):
    """Whether a finished trial should take over, be kept waiting, or go.

    Compared on the lower bound of each interval rather than on the point
    estimates. A trial that got a favourable run of problems has a wide interval
    and a low bound; the incumbent, with hundreds of attempts behind it, does
    not. That asymmetry is the point — displacing a proven strategy should need
    more evidence than matching it.
    """
    min_uses = int(cfg.REASON_TRIAL_N if min_uses is None else min_uses)
    used = trial.used(family)
    if used < min_uses:
        return "trial", f"{used}/{min_uses} applications"

    trial_lower = wilson_lower(trial.solved(family), used)
    if incumbent is None or incumbent.used(family) == 0:
        # Nothing to compare against. Promote only on evidence that stands on
        # its own, rather than by default because the field was empty.
        if trial_lower > 0.5:
            return "active", f"no incumbent; lower bound {trial_lower:.3f}"
        return "retired", f"no incumbent and lower bound only {trial_lower:.3f}"

    incumbent_lower = wilson_lower(incumbent.solved(family),
                                   incumbent.used(family))
    if trial_lower > incumbent_lower:
        return "active", (f"lower bound {trial_lower:.3f} beats "
                          f"{incumbent.name} at {incumbent_lower:.3f}")
    return "retired", (f"lower bound {trial_lower:.3f} does not beat "
                       f"{incumbent.name} at {incumbent_lower:.3f}")
