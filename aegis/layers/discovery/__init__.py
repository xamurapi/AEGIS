"""Creating knowledge nobody supplied (spec M7).

``known data → new hypothesis → mathematical model → experiment → discovery``,
as one object with the five stages wired together and each of them separately
testable.

The engine owns the loop; the five modules own the work:

* :mod:`datapool` — the observations, with a declared schema and provenance
* :mod:`hypothesis` — what might be true, and the count that makes the
  false-discovery correction honest
* :mod:`symbolic` — the formula, searched deterministically and written down
* :mod:`experiment` — the plan, frozen before the data, and the fence around
  any intervention
* :mod:`ledger` — what is known, what was refuted, and where it has been applied

The thing that makes this an engine rather than a report generator is the last
step, :meth:`DiscoveryEngine.applications`. A confirmed relationship is pushed
back into the systems it is about — a prior in the world model, a rule in the
behaviour policy, a narrowed range in evolution, a shifted budget in motivation
— and the application is recorded. Knowledge that changes nothing is a claim
about the world that the world never has to answer for.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aegis.config as cfg
from aegis.layers.discovery import experiment as experiment_module
from aegis.layers.discovery import hypothesis as hypothesis_module
from aegis.layers.discovery import symbolic
from aegis.layers.discovery.datapool import DataPool, Frame, VariableSpec
from aegis.layers.discovery.experiment import (
    CONTROLLABLE, Intervention, Preregistration, is_controllable, levels_for,
    preregister, run_observational,
)
from aegis.layers.discovery.hypothesis import Hypothesis, Scanner
from aegis.layers.discovery.ledger import OPEN, Discovery, Ledger
from aegis.layers.discovery.symbolic import Model
from aegis.telemetry.metrics import (
    DISC_EXPERIMENTS, DISC_FDR_REJECTIONS, DISC_HYPOTHESES_TESTED,
    DISC_REPLICATED, DISC_SUPPORTED,
)

logger = logging.getLogger("aegis.discovery")

__all__ = [
    "CONTROLLABLE", "DataPool", "Discovery", "DiscoveryEngine", "Frame",
    "Hypothesis", "Intervention", "Ledger", "Model", "Preregistration",
    "Scanner", "VariableSpec", "is_controllable", "levels_for", "preregister",
    "run_observational",
]

#: The metric the engine explains by default. Everything the system does is
#: ultimately meant to move this, so it is the target worth having laws about.
DEFAULT_TARGET = "aegis.reward.value"

#: Metrics pulled into the pool. Deliberately not "all of them": every extra
#: variable multiplies the number of pairs tested, and a correction over
#: thousands of tests rejects the real findings along with the noise.
WATCHED = (
    "aegis.reward.value", "aegis.bench.score", "aegis.wm.surprise",
    "aegis.wm.brier", "aegis.plan.override_rate", "aegis.plan.ev_gap",
    "aegis.policy.behaviour_delta_rate", "aegis.res.denied",
    "aegis.tick.duration_ms", "aegis.reason.pass_holdout",
)


class DiscoveryEngine:
    """The full loop, from telemetry to a registered law."""

    def __init__(self, *, directory: Path | None = None, telemetry=None,
                 world_model=None, cortex=None, target: str = DEFAULT_TARGET,
                 watched=WATCHED):
        # ``cfg.DISCOVERY_DIR``, not ``DATA_DIR / "discovery"``: the acceptance
        # harnesses isolate an arm by redirecting the per-contour constants,
        # and a store that composed its own path from ``DATA_DIR`` would ignore
        # the redirection and write into the live repository.
        root = Path(directory) if directory else Path(cfg.DISCOVERY_DIR)
        self.directory = root
        self.telemetry = telemetry
        self.world_model = world_model
        self.cortex = cortex
        self.target = str(target)
        self.watched = tuple(dict.fromkeys([self.target, *watched]))

        self.pool = DataPool(root / "datasets")
        self.scanner = Scanner()
        self.ledger = Ledger(root / "ledger.json")
        self.prereg_path = root / "prereg.jsonl"

        #: Hypotheses waiting for a model, and models waiting for an experiment.
        self.pending: list[Hypothesis] = []
        self.models: dict[str, Model] = {}
        #: Which column names each model was fitted against — a lagged predictor
        #: is a different column from the variable it came from, and reapplying
        #: the formula needs the name it was actually fitted on.
        self._predictors_for: dict[str, tuple] = {}
        self.preregs: dict[str, Preregistration] = {}
        self.intervention: Intervention | None = None

        self.scans = 0
        self.fits = 0
        self.experiments = 0
        self.last_scan_tick = 0

    # ── stage 1: the data ────────────────────────────────────────────

    def ingest(self, window: int | None = None) -> int:
        """Pull the watched telemetry series into the pool, aligned by tick."""
        if self.telemetry is None:
            return 0
        return self.pool.ingest_telemetry(self.telemetry, self.watched,
                                          name="telemetry", window=window)

    def frame(self, window: int | None = None) -> Frame:
        return self.pool.frame("telemetry", window=window)

    # ── what the action registry asks about (Appendix A) ─────────────
    #
    # The registry is the source of truth for what the system can do, and its
    # preconditions are predicates over these. They are cheap on purpose: a
    # precondition is evaluated for every candidate action on every tick, and
    # one that scanned the pool would put the cost of deciding *not* to do
    # something into the DECIDE budget.

    def data_points(self) -> int:
        """Observations available to explain anything with."""
        return self.pool.row_count("telemetry")

    def unmodelled_hypotheses(self) -> list:
        """Queued hypotheses that have no formula yet."""
        return [item for item in self.pending if item.id not in self.models]

    def active_preregistrations(self) -> list:
        """Frozen plans whose experiment has not concluded.

        A plan whose discovery has already left ``proposed`` is finished with,
        and leaving it here would make ``run_experiment`` permanently available
        and permanently a no-op.
        """
        out = []
        for identifier, record in sorted(self.preregs.items()):
            entry = self.ledger.get(identifier)
            if entry is None or entry.status in OPEN:
                out.append(record)
        return out

    # ── stage 2: hypotheses ──────────────────────────────────────────

    def scan(self, tick: int = 0, window: int | None = None) -> list[Hypothesis]:
        """Scan for associations; drop anything already refuted."""
        self.scans += 1
        self.last_scan_tick = int(tick)
        self.ingest(window=window)
        frame = self.frame(window=window)
        if len(frame) < cfg.DISC_MIN_N:
            return []

        found = self.scanner.scan(frame, self.target, tick=tick)
        found += hypothesis_module.from_world_model(
            self.world_model, frame.names, tick=tick) \
            if self.world_model is not None else []

        fresh = []
        seen = {item.id for item in self.pending}
        for item in found:
            if item.id in seen or self.ledger.is_refuted(item.id):
                continue
            seen.add(item.id)
            fresh.append(item)
        self.pending.extend(fresh)
        self._record(DISC_HYPOTHESES_TESTED, self.scanner.tested, tick)
        self._record(DISC_FDR_REJECTIONS, self.scanner.rejected, tick)
        return fresh

    def accept_formal(self, text: str, *, tick: int = 0,
                      statement: str = "") -> Hypothesis | None:
        """Take a hypothesis stated in the grammar — the cortex path (M7.4)."""
        frame = self.frame()
        item = hypothesis_module.from_formal(text, frame.names, tick=tick,
                                             statement=statement)
        if item is None or self.ledger.is_refuted(item.id):
            return None
        if all(existing.id != item.id for existing in self.pending):
            self.pending.append(item)
        return item

    # ── stage 3: the formula ─────────────────────────────────────────

    def fit_next(self, tick: int = 0, window: int | None = None) -> Model | None:
        """Fit the highest-ranked hypothesis that has no model yet."""
        pending = [item for item in self.pending if item.id not in self.models]
        if not pending:
            return None
        item = pending[0]
        frame = self.frame(window=window)

        prepared, predictors = frame, []
        for name in item.predictors:
            lag = int(item.lags.get(name, 0))
            if lag:
                prepared = prepared.lag(name, lag, as_name=f"{name}_lag{lag}")
                predictors.append(f"{name}_lag{lag}")
            else:
                predictors.append(name)

        self.fits += 1
        model = symbolic.fit(prepared, item.target, predictors)
        if model is None:
            self.pending = [other for other in self.pending if other.id != item.id]
            return None
        self.models[item.id] = model
        self._predictors_for[item.id] = tuple(predictors)
        return model

    # ── stage 4: the experiment ──────────────────────────────────────

    def preregister_next(self, tick: int = 0, *,
                         design: str = "observational_holdout",
                         variable: str | None = None,
                         levels=()) -> Preregistration | None:
        """Freeze a plan for the first modelled hypothesis without one."""
        for item in self.pending:
            model = self.models.get(item.id)
            if model is None or item.id in self.preregs:
                continue
            record = preregister(item, model, design=design, tick=tick,
                                 variable=variable, levels=levels)
            if record is None:
                continue
            self.preregs[item.id] = record
            self.ledger.propose(item, model, record, tick=tick)
            experiment_module.append_prereg(self.prereg_path, record)
            return record
        return None

    def run_observational(self, identifier: str, tick: int = 0,
                          window: int | None = None) -> dict:
        """Score a frozen plan on data recorded after it was frozen."""
        record = self.preregs.get(str(identifier))
        model = self.models.get(str(identifier))
        item = next((entry for entry in self.pending
                     if entry.id == str(identifier)), None)
        if record is None or model is None or item is None:
            return {"status": "invalid", "reason": "no frozen plan for that id"}

        frame = self.frame(window=window)
        predictors = list(self._predictors_for.get(str(identifier),
                                                   item.predictors))
        for name in item.predictors:
            lag = int(item.lags.get(name, 0))
            if lag:
                frame = frame.lag(name, lag, as_name=f"{name}_lag{lag}")

        # Where this run's window starts: after everything already counted for
        # this discovery, so a confirmation is a genuinely new window rather
        # than the first one re-read (M7.8).
        entry = self.ledger.get(str(identifier))
        since = max((int(end) for _, end in (entry.windows if entry else [])),
                    default=record.created_tick)

        self.experiments += 1
        result = run_observational(record, model, frame, predictors, item.target,
                                   since=since)
        ticks = [int(row.get("tick", 0)) for row in frame.rows()
                 if int(row.get("tick", -1)) > since]
        window_span = (min(ticks), max(ticks)) if ticks else None
        self.ledger.record_result(str(identifier), result, tick=tick,
                                  window=window_span)
        self._publish(tick)
        return result

    def start_intervention(self, identifier: str, variable: str,
                           levels, tick: int = 0, *, apply, read=None) -> bool:
        """Begin an ABAB series. Every gate of M7.6 is checked before it runs."""
        if self.intervention is not None and self.intervention.active:
            return False
        item = next((entry for entry in self.pending
                     if entry.id == str(identifier)), None)
        model = self.models.get(str(identifier))
        record = preregister(item or {"id": str(identifier)}, model,
                             design="interventional_abab", tick=tick,
                             variable=variable, levels=levels)
        if record is None:
            return False
        self.preregs[str(identifier)] = record
        if item is not None:
            self.ledger.propose(item, model, record, tick=tick)
        experiment_module.append_prereg(self.prereg_path, record)

        self.intervention = Intervention(record, apply=apply, read=read)
        started = self.intervention.start(tick)
        if not started:
            self.intervention = None
        return started

    def step_intervention(self, tick: int, reward: float, health: str = "ok",
                          kill_switch: bool = False) -> dict:
        """One tick of a running series, with the abort conditions checked first."""
        if self.intervention is None:
            return {"state": "inactive"}
        outcome = self.intervention.step(tick, reward, health=health,
                                         kill_switch=kill_switch)
        if outcome.get("state") in ("finished", "aborted"):
            self.experiments += 1
            result = self.intervention.analyse()
            self.ledger.record_result(self.intervention.prereg.hypothesis_id,
                                      result, tick=tick,
                                      window=(self.intervention.started_tick, tick))
            self.intervention = None
            self._publish(tick)
            return {**outcome, "result": result}
        return outcome

    # ── the executors the registry names ─────────────────────────────

    def fit(self, tick: int = 0) -> dict:
        """The ``fit_model`` executor. Reports rather than returns a model.

        Actions return something the tick can log; a ``Model`` object is not
        that, and the caller that wants one has :meth:`fit_next`.
        """
        model = self.fit_next(tick=tick)
        if model is None:
            return {"fitted": False, "pending": len(self.pending)}
        return {"fitted": True, "expr": model.expr,
                "r2_valid": round(model.r2_valid, 4),
                "complexity": model.complexity}

    def step_experiment(self, tick: int = 0, *, reward: float = 0.0,
                        health: str = "ok", kill_switch: bool = False) -> dict:
        """The ``run_experiment`` executor: one tick of whatever is running.

        An intervention takes priority because it is the design that is
        actively changing the system, and a tick of it is what returns the
        parameter to normal when it has to end. With no series running the
        oldest frozen observational plan is scored.
        """
        if self.intervention is not None and self.intervention.active:
            return self.step_intervention(tick, reward, health=health,
                                          kill_switch=kill_switch)
        pending = self.active_preregistrations()
        if not pending:
            return {"state": "idle"}
        return self.run_observational(pending[0].hypothesis_id, tick=tick)

    # ── stage 5: applying what was learned (M7.9) ────────────────────

    def applications(self, tick: int = 0, *, world_model=None, policy=None,
                     evolution=None, resources=None) -> list[str]:
        """Push confirmed knowledge into the systems it is about.

        Each application is recorded against the discovery, so a later
        regression can send exactly the responsible claim back for re-testing
        rather than distrusting everything at once.
        """
        applied: list[str] = []
        for record in self.ledger.by_status("law") + \
                self.ledger.by_status("replicated"):
            target = str(record.hypothesis.get("target", ""))
            predictors = list(record.hypothesis.get("predictors", []))
            if not target or not predictors:
                continue

            if world_model is not None and hasattr(world_model, "note_prior"):
                if self._apply_safely(world_model.note_prior, record.formula,
                                      float(record.effect_size)):
                    self.ledger.note_application(record.id, "world_model", tick)
                    applied.append(f"{record.id}→world_model")
            if policy is not None and hasattr(policy, "note_discovery"):
                if self._apply_safely(policy.note_discovery, record.id,
                                      record.formula, float(record.effect_size)):
                    self.ledger.note_application(record.id, "policy", tick)
                    applied.append(f"{record.id}→policy")
            if evolution is not None and hasattr(evolution, "narrow_gene"):
                for name in predictors:
                    if self._apply_safely(evolution.narrow_gene, name,
                                          float(record.effect_size)):
                        self.ledger.note_application(record.id, f"evolution:{name}",
                                                     tick)
                        applied.append(f"{record.id}→evolution:{name}")
            if resources is not None and hasattr(resources, "note_discovery"):
                if self._apply_safely(resources.note_discovery, record.id,
                                      float(record.effect_size)):
                    self.ledger.note_application(record.id, "resources", tick)
                    applied.append(f"{record.id}→resources")
        return applied

    @staticmethod
    def _apply_safely(call, *args) -> bool:
        """An application that raises must not take the tick with it."""
        try:
            return bool(call(*args))
        except Exception:
            logger.exception("Applying a discovery failed")
            return False

    def review_applications(self, metric_before: float, metric_after: float,
                            tick: int = 0) -> list[str]:
        """Send applied discoveries back for re-testing if the metric fell.

        The spec's rule (M7.9): a discovery whose application made things worse
        is not knowledge yet, whatever the experiment said.
        """
        if metric_after >= metric_before:
            return []
        sent = []
        for record in list(self.ledger.entries.values()):
            if record.applications and record.status in ("supported", "replicated",
                                                         "law"):
                if self.ledger.retest(record.id, "the metric fell after it was "
                                                 "applied", tick):
                    sent.append(record.id)
        return sent

    # ── plumbing ─────────────────────────────────────────────────────

    def _record(self, metric: str, value, tick: int) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.record(metric, float(value), tick=tick)
        except Exception:
            logger.debug("Could not record %s", metric, exc_info=True)

    def _publish(self, tick: int) -> None:
        counts = self.ledger.counts()
        self._record(DISC_SUPPORTED, counts.get("supported", 0), tick)
        self._record(DISC_REPLICATED, counts.get("replicated", 0)
                     + counts.get("law", 0), tick)
        self._record(DISC_EXPERIMENTS, self.experiments, tick)

    def publish_metrics(self, tick: int) -> None:
        """Every metric of Appendix G this contour owns, on the tick's schedule.

        Written every publication rather than only when something happens: a
        gauge that is absent between events is a gauge no series can be built
        from, and the discovery engine of all things should not be the contour
        whose own history has gaps in it.
        """
        self._record(DISC_HYPOTHESES_TESTED, self.scanner.tested, tick)
        self._record(DISC_FDR_REJECTIONS, self.scanner.rejected, tick)
        self._publish(tick)

    def save(self) -> bool:
        # Both stores are written whatever the other reports: `and` would
        # short-circuit, and a pool write that failed quietly (write_store
        # returns False rather than raising) would then skip the ledger save
        # too — losing recorded statuses and refutations on top of the rows
        # already lost. The return value is still the conjunction, because
        # "saved" means both of them.
        pool_saved = bool(self.pool.save())
        ledger_saved = bool(self.ledger.save())
        return pool_saved and ledger_saved

    def status(self) -> dict:
        counts = self.ledger.counts()
        return {
            "target": self.target,
            "scans": self.scans, "fits": self.fits,
            "experiments": self.experiments,
            "hypotheses_tested": self.scanner.tested,
            "fdr_rejections": self.scanner.rejected,
            "pending": len(self.pending), "models": len(self.models),
            "preregistered": len(self.preregs),
            "intervention": (self.intervention.status()
                             if self.intervention is not None else None),
            "discoveries": counts,
            "supported": counts.get("supported", 0),
            "replicated": counts.get("replicated", 0) + counts.get("law", 0),
            "laws": counts.get("law", 0),
            "refuted": counts.get("refuted", 0),
            "pool": self.pool.status(),
        }
