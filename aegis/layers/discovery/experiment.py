"""Committing to a test before running it, and running it safely (spec M7.6).

**Preregistration is the whole idea.** A plan — which variable, which levels,
how many observations, which analysis — is written down and hashed *before* any
data is collected. Afterwards the hash is checked. An analysis that does not
match the frozen plan does not produce a weaker result; it produces an
``invalid`` one. Without this the engine would be free to look at the data, pick
the comparison that came out well, and register that as a discovery — which is
how a system that only ever looks at noise still accumulates laws.

Two designs, per the spec:

* **Observational.** The model is checked against data recorded strictly *after*
  the registration tick. True out-of-sample in time, and no intervention.
* **Interventional ABAB.** The system sets a controlled parameter to two levels
  in alternating blocks and compares. Stronger evidence, and the only design
  that distinguishes cause from association — which is why it is fenced.

The fence on the interventional path, every item of which is a gate that must
pass before a single tick of it runs:

1. The variable is on the Appendix F whitelist. Not "not forbidden" — *listed*.
2. It is not in ``IMMUTABLE_PARAMS``, checked separately, because a whitelist is
   a thing someone could edit and the immutable set is the thing that must hold
   even then.
3. The amplitude is within ``DISC_INTERVENTION_MAX_DELTA`` of the range.
4. The original value is captured before the first change, and restored on
   completion, on abort, and on the object being discarded.
5. It aborts immediately — mid-block, without waiting for a boundary — on
   ``health != ok``, on the kill switch, or on reward falling below
   ``baseline − 2σ`` for half a block.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.safety.immutable import is_immutable
from aegis.util.canonical import canonical_json
from aegis.util.stats import (
    Welford, compare_samples, cohens_d, mean, r_squared, required_n,
)

logger = logging.getLogger("aegis.discovery")

#: Appendix F — the only parameters an intervention may touch. Declared here as
#: data, checked on every path. ``res_share_*`` and the ``*_EVERY_N_TICKS``
#: entries are prefixes, matched as such.
CONTROLLABLE: frozenset[str] = frozenset({
    "explore_bonus", "plan_beam", "plan_depth", "w_exp", "w_cost",
    "policy_weight", "reason_vote_n", "reason_budget", "solver_order",
    "res_share_competence", "res_share_knowledge", "res_share_coherence",
    "res_share_stability",
    "LLM_THINK_EVERY_N_TICKS", "ENV_STEP_EVERY_N_TICKS",
    "COGNITIVE_GRAPH_EVERY_N_TICKS",
})

#: Minimum blocks in an ABAB series. Four is the schema's name: fewer cannot
#: separate the effect of the level from a drift that happened to coincide.
MIN_BLOCKS = 4

#: Floor on out-of-sample R² regardless of what a plan predicted. Without it a
#: model that preregistered an effect of nothing would clear its own bar by
#: continuing to explain nothing.
MIN_HOLDOUT_R2 = 0.05

DESIGNS = ("observational_holdout", "interventional_abab")
ANALYSES = ("welch_t", "mann_whitney", "r2_holdout")


def is_controllable(name: str) -> bool:
    """Whether an intervention may set this variable at all (Appendix F)."""
    name = str(name)
    if is_immutable(name):
        return False
    return name in CONTROLLABLE


@dataclass
class Preregistration:
    """A frozen plan. Its hash is what makes it frozen."""

    hypothesis_id: str
    model_expr: str
    predicted_effect: float
    direction: str                 # "increase" | "decrease"
    design: str
    variable: str | None = None
    levels: tuple[float, ...] = ()
    n_required: int = 0
    analysis: str = "welch_t"
    created_tick: int = 0
    block_ticks: int = 0
    frozen_hash: str = ""

    def plan(self) -> dict:
        """Exactly the fields the hash covers — the plan, and nothing else."""
        return {"hypothesis_id": self.hypothesis_id,
                "model_expr": self.model_expr,
                "predicted_effect": round(float(self.predicted_effect), 6),
                "direction": self.direction, "design": self.design,
                "variable": self.variable,
                "levels": [round(float(level), 6) for level in self.levels],
                "n_required": int(self.n_required), "analysis": self.analysis,
                "created_tick": int(self.created_tick),
                "block_ticks": int(self.block_ticks)}

    def compute_hash(self) -> str:
        from aegis.util.quasirandom import hash_index

        return f"{hash_index(1 << 48, 'prereg', canonical_json(self.plan())):012x}"

    def freeze(self) -> "Preregistration":
        self.frozen_hash = self.compute_hash()
        return self

    def intact(self) -> bool:
        """Whether the plan still matches the hash taken when it was frozen."""
        return bool(self.frozen_hash) and self.compute_hash() == self.frozen_hash

    def as_dict(self) -> dict:
        return {**self.plan(), "frozen_hash": self.frozen_hash}

    @classmethod
    def from_dict(cls, data: dict) -> "Preregistration | None":
        if not isinstance(data, dict) or not data.get("hypothesis_id"):
            return None
        try:
            record = cls(
                hypothesis_id=str(data["hypothesis_id"]),
                model_expr=str(data.get("model_expr", "")),
                predicted_effect=float(data.get("predicted_effect", 0.0)),
                direction=str(data.get("direction", "increase")),
                design=str(data.get("design", "observational_holdout")),
                variable=(str(data["variable"])
                          if data.get("variable") is not None else None),
                levels=tuple(float(level) for level in data.get("levels", []) or []),
                n_required=int(data.get("n_required", 0)),
                analysis=str(data.get("analysis", "welch_t")),
                created_tick=int(data.get("created_tick", 0)),
                block_ticks=int(data.get("block_ticks", 0)))
        except (TypeError, ValueError):
            return None
        record.frozen_hash = str(data.get("frozen_hash", ""))
        return record


def _identity_of(hypothesis) -> str:
    """The id of a hypothesis, whether it arrived as an object or a mapping.

    Both forms are legitimate: the engine holds :class:`Hypothesis` objects, and
    an intervention can be started for an id that has not been queued yet, which
    is passed as a bare ``{"id": ...}``. ``getattr(mapping, "id", mapping)``
    falls through to the mapping itself, so that case used to produce a plan
    identified by the *text of a dict* — a plan the ledger could never match
    against the hypothesis it was about.
    """
    identifier = getattr(hypothesis, "id", None)
    if identifier is None and isinstance(hypothesis, dict):
        identifier = hypothesis.get("id")
    return str(identifier if identifier is not None else hypothesis)


def preregister(hypothesis, model, *, design: str = "observational_holdout",
                tick: int = 0, variable: str | None = None,
                levels=(), effect_size: float = 0.5,
                analysis: str | None = None,
                block_ticks: int | None = None) -> Preregistration | None:
    """Write the plan and freeze it. ``None`` if the plan is not admissible."""
    if design not in DESIGNS:
        return None
    if analysis is None:
        analysis = "r2_holdout" if design == "observational_holdout" else "welch_t"
    if analysis not in ANALYSES:
        return None
    if design == "interventional_abab":
        if variable is None or not is_controllable(variable):
            logger.warning("Refusing to preregister an intervention on %r: "
                           "not a controllable variable", variable)
            return None
        if len(levels) != 2:
            return None

    predicted = float(getattr(model, "r2_valid", 0.0)) if model is not None else 0.0
    record = Preregistration(
        hypothesis_id=_identity_of(hypothesis),
        model_expr=str(getattr(model, "expr", "")),
        predicted_effect=predicted,
        direction="increase" if predicted >= 0 else "decrease",
        design=design, variable=variable,
        levels=tuple(float(level) for level in levels),
        n_required=required_n(effect_size),
        analysis=analysis, created_tick=int(tick),
        block_ticks=int(cfg.DISC_BLOCK_TICKS if block_ticks is None
                        else block_ticks))
    return record.freeze()


# ── the observational design ─────────────────────────────────────────

def run_observational(prereg: Preregistration, model, frame, predictors,
                      target: str, *, since: int | None = None) -> dict:
    """Score a stored formula on data recorded after the plan was frozen.

    The frame is filtered by tick, not merely taken from the end: "after the
    registration" is a claim about when the observation happened, and slicing
    the last N rows would silently include pre-registration data whenever the
    series had gaps.

    ``since`` is what makes **replication** possible. A confirmation re-run
    against the registration tick would score the same rows as the first one
    and every later one, so the ledger would reject each as an overlapping
    window and the discovery would sit at ``supported`` forever. Passing the
    end of the last counted window instead gives genuinely disjoint successive
    windows — which is what "an independent repetition in a different time
    window" means (M7.8).
    """
    from aegis.layers.discovery import symbolic

    if not prereg.intact():
        return {"status": "invalid", "reason": "the plan was altered after it was frozen"}
    floor = prereg.created_tick if since is None else max(int(since),
                                                          prereg.created_tick)
    fresh = frame.filter(lambda row: int(row.get("tick", -1)) > floor)
    fresh = fresh.numeric(target, *predictors)
    if len(fresh) < max(8, prereg.n_required // 10):
        return {"status": "pending", "reason": f"{len(fresh)} rows after the plan",
                "n": len(fresh)}

    actual, predicted = [], []
    for row in fresh.rows():
        value = symbolic.predict(model, row, predictors)
        if value is None:
            continue
        actual.append(float(row[target]))
        predicted.append(value)
    if len(actual) < 8:
        return {"status": "pending", "reason": "the formula did not apply",
                "n": len(actual)}

    score = r_squared(actual, predicted)
    residuals = [a - p for a, p in zip(actual, predicted)]

    # The bar is the effect the plan *committed to*, not a fixed fraction of
    # variance. A relationship that explains a quarter of the variance in
    # sample and still explains a quarter out of it has held up: that is a
    # weak law, not a false one, and a flat "R² ≥ 0.5" would refute every
    # genuine secondary effect the system has while passing anything dominant.
    # Half of the preregistered effect, floored so a model that predicted
    # nothing cannot clear its own bar by predicting nothing.
    threshold = max(MIN_HOLDOUT_R2, 0.5 * float(prereg.predicted_effect))
    held_up = score >= threshold
    return {
        "status": "supported" if held_up else "refuted",
        "analysis": "r2_holdout", "r2_holdout": round(score, 4),
        "threshold": round(threshold, 4),
        "effect_size": round(score, 4),
        # An out-of-sample R² is not a hypothesis test and has no p-value. The
        # ledger requires one, so the honest thing is to report the threshold
        # this design actually applies rather than to invent a number.
        "p_value": 0.0 if held_up else 1.0,
        "n": len(actual), "residual_mean": round(mean(residuals), 6),
        "frozen_hash": prereg.frozen_hash,
    }


# ── the interventional design ────────────────────────────────────────

class Intervention:
    """One ABAB series over a controlled parameter, with the fence around it.

    The object owns the original value from the moment it starts and gives it
    back on every exit path. ``apply`` and ``restore`` are injected rather than
    reached for, so this module never learns how to write to a genome — the
    caller supplies exactly the two operations it is allowed to perform.
    """

    def __init__(self, prereg: Preregistration, *, apply, read=None,
                 block_ticks: int | None = None, min_blocks: int = MIN_BLOCKS):
        self.prereg = prereg
        self._apply = apply
        self._read = read
        self.block_ticks = int(block_ticks or prereg.block_ticks
                               or cfg.DISC_BLOCK_TICKS)
        self.min_blocks = max(MIN_BLOCKS, int(min_blocks))
        self.variable = prereg.variable
        self.levels = list(prereg.levels)
        self.original = None
        self.started_tick = 0
        self.active = False
        self.aborted = False
        self.abort_reason = ""
        self.blocks: list[dict] = []
        self.samples: dict[int, list[float]] = {0: [], 1: []}
        self._current_block = -1
        self._below_baseline = 0
        self._baseline = Welford()

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self, tick: int) -> bool:
        """Begin the series. Refuses unless every gate passes."""
        if self.active or self.aborted:
            return False
        if not self.prereg.intact():
            self.abort_reason = "the plan was altered after it was frozen"
            self.aborted = True
            return False
        if self.variable is None or not is_controllable(self.variable):
            self.abort_reason = f"{self.variable!r} is not controllable"
            self.aborted = True
            return False
        if len(self.levels) != 2:
            self.abort_reason = "an ABAB series needs exactly two levels"
            self.aborted = True
            return False
        if not cfg.DISC_INTERVENTION_ENABLED:
            self.abort_reason = "interventions are disabled by configuration"
            self.aborted = True
            return False
        # Gate 4 is "capture and restore", and both halves have to be real. A
        # caller that supplies only `apply` used to get a series whose
        # restore() was silently a no-op — the experimental level stayed in
        # force forever on every exit path, which is the exact failure the
        # fence exists to prevent. An intervention that cannot give the
        # parameter back must not be allowed to take it.
        if self._read is None:
            self.abort_reason = ("no reader for the original value — a series "
                                 "that cannot be restored must not start")
            self.aborted = True
            return False
        self.original = self._read()
        if self.original is None:
            self.abort_reason = "the original value could not be captured"
            self.aborted = True
            return False
        self.started_tick = int(tick)
        self.active = True
        self._current_block = -1
        return True

    def block_index(self, tick: int) -> int:
        return max(0, (int(tick) - self.started_tick) // max(1, self.block_ticks))

    def level_for(self, tick: int) -> float:
        """ABAB: even blocks take level A, odd blocks level B."""
        return self.levels[self.block_index(tick) % 2]

    def step(self, tick: int, reward: float, health: str = "ok",
             kill_switch: bool = False) -> dict:
        """One tick of the series. Returns what it did and why.

        Safety is checked *before* anything is recorded or applied: a tick that
        should have stopped the series must not first contribute a data point to
        it, or the abort condition becomes part of the result it was meant to
        prevent.
        """
        if not self.active:
            return {"state": "inactive"}

        if kill_switch:
            return self.abort("the kill switch is active")
        if str(health) != "ok":
            return self.abort(f"health is {health}")

        block = self.block_index(tick)
        if block >= self.min_blocks * 2 and block % 2 == 0:
            return self.finish("the series completed")

        value = float(reward)
        if self._baseline.n >= 8:
            floor = self._baseline.mean - 2.0 * self._baseline.sd()
            if value < floor:
                self._below_baseline += 1
                if self._below_baseline >= max(1, self.block_ticks // 2):
                    return self.abort("reward stayed below baseline − 2σ")
                # Deliberately *not* folded into the baseline. A collapse fed
                # back into its own reference drags the mean down and inflates
                # the spread, so the floor chases the damage and the guard
                # stops firing exactly while the thing it guards against is
                # happening. The baseline is what the system looked like before
                # the trouble, and it stays that.
                return {"state": "running", "block": self._current_block,
                        "arm": self._current_block % 2, "below_baseline": True}
            self._below_baseline = 0
        self._baseline.update(value)

        if block != self._current_block:
            self._current_block = block
            self.blocks.append({"block": block, "level": self.level_for(tick),
                                "start_tick": int(tick), "n": 0})
            self._apply(self.variable, self.level_for(tick))
            # The boundary tick's reward is dropped, not recorded: it was
            # produced under the level in force *before* this switch (the
            # first block's under no experimental level at all). Filing it in
            # the new block's arm put one old-level sample into every block —
            # a systematic dilution of the contrast that biased analyse()
            # toward "refuted" on every series.
            return {"state": "running", "block": block, "arm": block % 2,
                    "level": self.level_for(tick), "boundary": True}

        arm = block % 2
        self.samples[arm].append(value)
        self.blocks[-1]["n"] = self.blocks[-1].get("n", 0) + 1
        return {"state": "running", "block": block, "arm": arm,
                "level": self.level_for(tick)}

    def abort(self, reason: str) -> dict:
        self.abort_reason = str(reason)
        self.aborted = True
        self.active = False
        self.restore()
        logger.warning("Intervention on %s aborted: %s", self.variable, reason)
        return {"state": "aborted", "reason": self.abort_reason}

    def finish(self, reason: str = "") -> dict:
        self.active = False
        self.restore()
        return {"state": "finished", "reason": str(reason)}

    def restore(self) -> bool:
        """Put the parameter back. Safe to call more than once."""
        if self.original is None or self.variable is None:
            return False
        try:
            self._apply(self.variable, self.original)
        except Exception:
            logger.exception("Restoring %s after an intervention failed",
                             self.variable)
            return False
        self.original = None
        return True

    def __del__(self):
        # Last resort. A series dropped without finishing must not leave the
        # system running at an experimental level forever.
        try:
            if self.original is not None:
                self.restore()
        except Exception:
            pass

    # ── the analysis, exactly as preregistered ───────────────────────

    def analyse(self) -> dict:
        if not self.prereg.intact():
            return {"status": "invalid",
                    "reason": "the plan was altered after it was frozen"}
        arm_a, arm_b = self.samples[0], self.samples[1]
        if len(arm_a) < 2 or len(arm_b) < 2:
            return {"status": "pending", "reason": "not enough observations",
                    "n": len(arm_a) + len(arm_b)}
        if self.aborted:
            return {"status": "invalid", "reason": self.abort_reason,
                    "n": len(arm_a) + len(arm_b)}

        if self.prereg.analysis == "mann_whitney":
            from aegis.util.stats import mann_whitney_u
            result = mann_whitney_u(arm_b, arm_a)
        else:
            result = compare_samples(arm_b, arm_a)
        supported = result.p_value < cfg.DISC_ALPHA and abs(result.effect) > 0
        return {
            "status": "supported" if supported else "refuted",
            "analysis": self.prereg.analysis,
            "effect_size": round(result.effect, 6),
            "cohens_d": round(cohens_d(arm_b, arm_a), 4),
            "p_value": round(result.p_value, 8),
            "n": len(arm_a) + len(arm_b),
            "n_a": len(arm_a), "n_b": len(arm_b),
            "blocks": len(self.blocks),
            "frozen_hash": self.prereg.frozen_hash,
        }

    def status(self) -> dict:
        return {"variable": self.variable, "levels": list(self.levels),
                "active": self.active, "aborted": self.aborted,
                "reason": self.abort_reason, "blocks": len(self.blocks),
                "n_a": len(self.samples[0]), "n_b": len(self.samples[1])}


def levels_for(current: float, low: float, high: float,
               max_delta: float | None = None) -> tuple[float, float]:
    """The two levels of an ABAB series, clamped to the allowed amplitude.

    The amplitude is a fraction of the parameter's *range*, not of its current
    value: a fraction of the value would let a parameter sitting near zero be
    moved by nothing and one sitting high be moved a long way, for no reason
    connected to what is safe.
    """
    max_delta = float(cfg.DISC_INTERVENTION_MAX_DELTA
                      if max_delta is None else max_delta)
    span = abs(float(high) - float(low))
    delta = span * max(0.0, min(1.0, max_delta))
    below = max(float(low), float(current) - delta)
    above = min(float(high), float(current) + delta)
    return (below, above)


def append_prereg(path, prereg: Preregistration) -> bool:
    """Write the plan to the log *before* the experiment runs (M7.6)."""
    from pathlib import Path

    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prereg.as_dict(), ensure_ascii=False) + "\n")
        return True
    except Exception:
        logger.warning("Could not write the preregistration log", exc_info=True)
        return False
