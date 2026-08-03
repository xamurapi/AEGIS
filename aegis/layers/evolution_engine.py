"""System 3: Evolution Engine v2 — a population, not a hill climb (spec M5).

    champion → 10 variants → isolated evaluation → selection → new champion
                                                            → rollback if worse

Version 1 mutated one parameter, waited for one benchmark, and kept the change
if the number went up. It is preserved underneath: ``register_champion``,
``propose_mutation``, ``judge_candidate``, ``abandon_candidate`` and ``status``
behave as they did, because the tick still calls them and the stored lineage
still has to be readable. ``propose_mutation`` now returns the first variant of
a generation rather than a hand-rolled ±10% step, which is the same contract
with a better generator behind it.

What is new is everything selection depends on:

* a genome of parameters that are actually read (Appendix C);
* ten variants per generation, scored in a pool against a fixed scenario;
* selection on ``valid`` only, with ``test`` reserved to confirm a champion —
  and ``valid_test_gap`` published, so overfitting is visible rather than
  assumed;
* a **rollback**: a promoted champion is watched, and if the live metric falls
  by more than ``EVO_ROLLBACK_DELTA`` within ``EVO_WATCH_TICKS`` it is put back.
  Without that, a genome that scored well on the benchmark and behaves badly in
  the world is permanent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.config import EVOLUTION_DIR
from aegis.layers.evolution.genome import GENE_NAMES, Genome
from aegis.layers.evolution.harness import FitnessReport, VariantEvaluator
from aegis.layers.evolution.operators import NoveltyArchive
from aegis.layers.evolution.population import Population

logger = logging.getLogger("aegis.evolution")

# Pure audit trail — never read during inference, so length costs nothing but
# disk. Kept long enough to reconstruct a full selection history.
MAX_LINEAGE = 2000
MUTATION_MAGNITUDE = 0.1  # ±10% per mutated parameter (v1 compatibility)
# A candidate must beat the champion by at least this margin to win —
# guards against benchmark noise promoting a sideways change.
FITNESS_EPSILON = cfg.EVO_EPSILON


class EvolutionEngine:
    """Owns the champion, the population and the lineage."""

    def __init__(self, store_path: Path | None = None, pool=None,
                 telemetry=None):
        self.champion: dict | None = None      # {genome, fitness, generation, created}
        self.candidate: dict | None = None     # v1 single-variant candidate
        self.generation = 0
        self.accepted = 0
        self.rejected = 0
        self.promotions = 0
        self.rollbacks = 0
        self.lineage: list[dict] = []
        self._param_idx = 0        # round-robin over genome keys (v1)
        self._direction_up = True  # alternate mutation direction (v1)
        self._store_path = store_path or (EVOLUTION_DIR / "lineage.json")

        self.telemetry = telemetry
        self.archive = NoveltyArchive()
        self.population = Population(self.archive)
        self.evaluator = VariantEvaluator(pool=pool)
        self.generation_running = False
        self.last_reports: list[FitnessReport] = []
        self.valid_test_gap: float | None = None
        #: The champion that was in place before the current one, and the tick
        #: it was replaced — everything a rollback needs.
        self.previous_champion: dict | None = None
        self.promoted_at_tick: int | None = None
        self.metric_at_promotion: float | None = None
        #: The most recent live reading, whether or not a watch is running.
        #: This is what makes a promotion's baseline the metric *before* the
        #: promotion: a baseline taken from the first reading after it lets a
        #: genome that degrades the metric immediately set its own damage as
        #: the reference and never trip the rollback.
        self._last_live_metric: float | None = None
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self):
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load evolution state from %s — starting fresh",
                           self._store_path, exc_info=True)
            return
        self.champion = self._migrate_champion(data.get("champion"))
        # A v1 candidate cannot be finished under v2 rules — its genome is the
        # old schema and the benchmark that would have judged it is gone — so it
        # is dropped rather than half-played (Appendix I). A *v2* candidate is a
        # different matter: its genome is the current schema and its benchmark
        # is still coming, and dropping it here made the restart guard in
        # `Substrate._restore_checkpoint` unreachable — the checkpointed genome
        # (the unjudged mutation) was silently adopted as the running baseline
        # on every restart while the champion record still held the old one.
        self.candidate = self._migrate_candidate(data.get("candidate"),
                                                 data.get("schema_version"))
        self.generation = data.get("generation", 0)
        self.accepted = data.get("accepted", 0)
        self.rejected = data.get("rejected", 0)
        self.promotions = data.get("promotions", 0)
        self.rollbacks = data.get("rollbacks", 0)
        self.lineage = data.get("lineage", [])
        self._param_idx = data.get("param_idx", 0)
        self._direction_up = data.get("direction_up", True)
        self.previous_champion = self._migrate_champion(data.get("previous_champion"))
        # The watch fields travel with previous_champion. Restoring the one
        # without the others meant any restart inside the watch window disabled
        # the rollback for good: `watch` saw promoted_at_tick=None and returned
        # None forever, while the badly behaved champion stayed in place.
        if self.previous_champion is not None:
            try:
                stored_tick = data.get("promoted_at_tick")
                self.promoted_at_tick = (int(stored_tick)
                                         if stored_tick is not None else None)
            except (TypeError, ValueError):
                self.promoted_at_tick = None
            try:
                stored_metric = data.get("metric_at_promotion")
                self.metric_at_promotion = (float(stored_metric)
                                            if stored_metric is not None else None)
            except (TypeError, ValueError):
                self.metric_at_promotion = None
        self.archive = NoveltyArchive.from_dict(data.get("archive"))
        self.population = Population(self.archive)

    @staticmethod
    def _migrate_champion(stored: dict | None) -> dict | None:
        """Bring a stored champion up to the current genome schema.

        A v1 champion held LoRA hyper-parameters, none of which are genes now.
        Migration keeps its fitness history and seeds the defaults — there is no
        correspondence to preserve, and inventing one would put a made-up genome
        at the top of the lineage (Appendix I).
        """
        if not isinstance(stored, dict):
            return None
        genome = Genome.from_stored({"genes": stored.get("genome")})
        try:
            fitness = float(stored.get("fitness", 0.0))
        except (TypeError, ValueError):
            fitness = 0.0
        return {"genome": genome.to_dict(), "fitness": fitness,
                "generation": int(stored.get("generation", 0) or 0),
                "created": float(stored.get("created", 0.0) or 0.0)}

    @staticmethod
    def _migrate_candidate(stored: dict | None, schema_version) -> dict | None:
        """Bring a stored pending candidate back, if it can still be judged.

        Only a v2 candidate qualifies: it was written in the current genome
        schema and its verdict — the next benchmark — is still meaningful. A v1
        candidate stays dropped, for the reason `_load` gives. Anything
        malformed is dropped too, because a candidate that cannot be read
        cannot be reverted either.
        """
        if not isinstance(stored, dict):
            return None
        try:
            version = int(schema_version or 1)
        except (TypeError, ValueError):
            version = 1
        if version < 2:
            return None
        param = stored.get("mutated_param")
        if param not in GENE_NAMES:
            return None
        return {
            "genome": Genome(stored.get("genome")).to_dict(),
            "mutated_param": str(param),
            "old_value": stored.get("old_value"),
            "new_value": stored.get("new_value"),
            "proposed_at_tick": int(stored.get("proposed_at_tick", 0) or 0),
            "created": float(stored.get("created", 0.0) or 0.0),
        }

    def save(self):
        data = {
            "schema_version": 2,
            "champion": self.champion,
            "previous_champion": self.previous_champion,
            # The watch fields are persisted with the champion they describe:
            # without them a restart inside the watch window kept the previous
            # champion but lost when and against what it was being compared,
            # and the rollback was permanently disarmed.
            "promoted_at_tick": self.promoted_at_tick,
            "metric_at_promotion": self.metric_at_promotion,
            "candidate": self.candidate,
            "generation": self.generation,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "lineage": self.lineage[-MAX_LINEAGE:],
            "param_idx": self._param_idx,
            "direction_up": self._direction_up,
            "archive": self.archive.to_dict(),
        }
        try:
            payload = json.dumps(data, ensure_ascii=False)
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to save evolution state to %s",
                           self._store_path, exc_info=True)

    # ── the champion ─────────────────────────────────────────────────

    def champion_genome(self) -> Genome:
        return Genome((self.champion or {}).get("genome"))

    def register_champion(self, genome: dict, fitness: float):
        """Set the current best-known genome. Unchanged contract from v1.

        Values outside a gene's range are clamped and unknown keys dropped, so
        a caller handing over the live parameters — which still contain the
        retired LoRA ones — gets a legal genome rather than an error.
        """
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            fitness = 0.0
        self.champion = {
            "genome": Genome(genome).to_dict(),
            "fitness": fitness,
            "generation": self.generation,
            "created": CLOCK.now(),
        }

    # ── v1 compatibility ─────────────────────────────────────────────

    def propose_mutation(self, tick: int) -> dict | None:
        """The first variant of a generation, expressed as a single change.

        Kept because the tick calls it and because ``EVO_POP_SIZE = 1`` has to
        keep working. The change reported is the gene that moved *most*, which
        is what the caller's safety pipeline needs to clamp and, if refused,
        revert.
        """
        if self.candidate is not None or not self.champion:
            return None
        parent = self.champion_genome()
        from aegis.layers.evolution.operators import coordinate_mutation

        child = coordinate_mutation(parent, self.generation, self._param_idx)
        self._param_idx += 1

        moved = self._largest_move(parent, child)
        if moved is None:
            return None
        param, old_value, new_value = moved
        self.candidate = {
            "genome": child.to_dict(),
            "mutated_param": param,
            "old_value": old_value,
            "new_value": new_value,
            "proposed_at_tick": tick,
            "created": CLOCK.now(),
        }
        return {"param": param, "old_value": old_value, "new_value": new_value}

    @staticmethod
    def _largest_move(parent: Genome, child: Genome):
        """Which gene moved furthest, in normalised terms.

        Normalised, or the answer would always be whichever gene happens to
        have the widest range.
        """
        from aegis.layers.evolution.genome import GENES_BY_NAME

        best = None
        for name in GENE_NAMES:
            gene = GENES_BY_NAME[name]
            moved = abs(gene.position(child[name]) - gene.position(parent[name]))
            if best is None or moved > best[0]:
                best = (moved, name)
        if best is None or best[0] <= 0:
            return None
        name = best[1]
        return name, parent[name], child[name]

    def judge_candidate(self, fitness: float) -> dict:
        """Selection for the single-candidate path. Unchanged contract."""
        if self.candidate is None or not self.champion:
            return {"decision": "no_candidate"}
        cand = self.candidate
        self.candidate = None
        self.generation += 1
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            fitness = 0.0
        record = {
            "generation": self.generation,
            "param": cand["mutated_param"],
            "old_value": cand["old_value"],
            "new_value": cand["new_value"],
            "champion_fitness": self.champion["fitness"],
            "candidate_fitness": fitness,
            "time": CLOCK.now(),
        }
        if fitness > self.champion["fitness"] + FITNESS_EPSILON:
            self.accepted += 1
            record["decision"] = "accepted"
            self._promote({"genome": cand["genome"], "fitness": fitness,
                           "generation": self.generation, "created": CLOCK.now()},
                          tick=cand.get("proposed_at_tick"))
            result = {"decision": "accepted", "param": cand["mutated_param"],
                      "revert_to": None}
        else:
            self.rejected += 1
            record["decision"] = "rejected"
            result = {"decision": "rejected", "param": cand["mutated_param"],
                      "revert_to": cand["old_value"]}
        self._append_lineage(record)
        self.save()
        return result

    def abandon_candidate(self) -> dict | None:
        """Drop a pending candidate without judging it."""
        if self.candidate is None:
            return None
        cand = self.candidate
        self.candidate = None
        return {"param": cand["mutated_param"], "revert_to": cand["old_value"]}

    # ── the generation (M5.6) ────────────────────────────────────────

    def run_generation(self, tick: int = 0, proposal: dict | None = None,
                       confirm: bool = True) -> dict:
        """One full generation: compose, evaluate, select, promote.

        Returns a summary rather than mutating anything outside itself, so a
        caller can decide whether to apply the champion through its own safety
        pipeline (§M5.6 requires that the application go through
        ``is_modification_safe`` and ``evaluate_action`` like any other change).
        """
        self.generation_running = True
        try:
            return self._run_generation(tick, proposal, confirm)
        finally:
            self.generation_running = False

    def _run_generation(self, tick: int, proposal: dict | None,
                        confirm: bool) -> dict:
        elites = self._elites()
        self.generation += 1
        genomes = self.population.build(elites, self.generation, proposal)
        start = self._split_start()
        reports = self.evaluator.evaluate(genomes, splits=("valid",), start=start)
        self.last_reports = reports
        self.population.record(reports)

        best_genome, best_report = self.population.best(reports)
        champion_fitness = (self.champion or {}).get("fitness")
        promoted = False
        confirmation = None

        if self.population.beats(best_report, champion_fitness):
            if confirm:
                confirmation = self.evaluator.confirm(best_genome, start=start)
                self.valid_test_gap = round(
                    best_report.score_valid - confirmation.score_valid, 6)
            self._promote({"genome": Genome(best_genome).to_dict(),
                           "fitness": best_report.fitness,
                           "generation": self.generation,
                           "created": CLOCK.now()}, tick=tick)
            self.accepted += 1
            promoted = True
        else:
            self.rejected += 1

        record = {
            "generation": self.generation,
            "population": len(genomes),
            "composition": {origin: self.population.origins.count(origin)
                            for origin in sorted(set(self.population.origins))},
            "best_fitness": best_report.fitness if best_report else None,
            "best_valid": best_report.score_valid if best_report else None,
            "champion_fitness": (self.champion or {}).get("fitness"),
            "valid_test_gap": self.valid_test_gap,
            "promoted": promoted,
            "novelty_skips": self.archive.skips,
            "time": CLOCK.now(),
        }
        self._append_lineage(record)
        self.save()
        self.publish_metrics(tick)
        return {
            "generation": self.generation,
            "promoted": promoted,
            "best": best_report.as_dict() if best_report else None,
            "test": confirmation.as_dict() if confirmation else None,
            "valid_test_gap": self.valid_test_gap,
            "composition": record["composition"],
            "reports": [report.as_dict() for report in reports],
        }

    def _elites(self) -> list[Genome]:
        """What the next generation is built from."""
        if self.last_reports and self.population.genomes:
            elites = self.population.elites_from(self.last_reports)
            if elites:
                return elites
        return [self.champion_genome()]

    def _split_start(self) -> int:
        """Rotate the generated task window every few generations (§M5.5).

        Selecting on the same tasks forever is how a population learns the
        tasks. Rotating the window means "better" has to mean better on cases
        the previous champion never saw.
        """
        rotate = max(1, int(cfg.EVO_SPLIT_ROTATE_EVERY))
        return (self.generation // rotate) * 97

    # ── promotion and rollback ───────────────────────────────────────

    def _promote(self, champion: dict, tick: int | None = None) -> None:
        self.previous_champion = self.champion
        self.champion = champion
        self.promotions += 1
        self.promoted_at_tick = int(tick or 0)
        # The baseline is the live metric BEFORE the promotion. Waiting for the
        # first reading after it would let a champion that degrades the metric
        # immediately measure the drop against its own degraded value — the
        # most common failure mode is exactly the one such a watch misses.
        # When no reading has been seen yet (a fresh engine, the unit tests),
        # the first post-promotion reading still serves as the fallback.
        self.metric_at_promotion = self._last_live_metric

    def watch(self, tick: int, live_metric: float | None) -> dict | None:
        """Check a freshly promoted champion against the live world (§M5.6).

        A genome that scores well on a benchmark can still behave badly in the
        system it was promoted into. The baseline is the last live reading
        *before* the promotion (kept by this method across calls); a fall of
        more than ``EVO_ROLLBACK_DELTA`` within ``EVO_WATCH_TICKS`` puts the
        previous champion back.

        Returns the rollback record, or None if nothing happened.
        """
        if live_metric is None:
            return None
        try:
            live_metric = float(live_metric)
        except (TypeError, ValueError):
            return None
        # Remembered even while no watch is running — this reading is the
        # baseline of whatever promotion happens next.
        self._last_live_metric = live_metric
        if self.previous_champion is None or self.promoted_at_tick is None:
            return None

        elapsed = int(tick) - int(self.promoted_at_tick)
        if elapsed < 0:
            return None
        if self.metric_at_promotion is None:
            self.metric_at_promotion = live_metric
            return None
        if elapsed > int(cfg.EVO_WATCH_TICKS):
            # The watch window closed without a fall: the promotion stands, and
            # there is nothing left to roll back to.
            self.previous_champion = None
            self.promoted_at_tick = None
            self.metric_at_promotion = None
            return None

        drop = self.metric_at_promotion - live_metric
        if drop <= float(cfg.EVO_ROLLBACK_DELTA):
            return None

        restored = self.previous_champion
        record = {
            "generation": self.generation,
            "decision": "rolled_back",
            "drop": round(drop, 6),
            "baseline": round(self.metric_at_promotion, 6),
            "at_tick": int(tick),
            "restored_fitness": restored.get("fitness"),
            "time": CLOCK.now(),
        }
        self.champion = restored
        self.previous_champion = None
        self.promoted_at_tick = None
        self.metric_at_promotion = None
        self.rollbacks += 1
        self._append_lineage(record)
        self.save()
        logger.info("Rolled back a champion after a live drop of %.4f", drop)
        return record

    def refuse_promotion(self, reason: str) -> dict | None:
        """Withdraw a promotion the caller's safety pipeline refused to apply.

        §M5.6 routes the application of a champion through
        ``is_modification_safe`` and ``evaluate_action`` like any other change.
        When those gates say no, the previous champion has to come back —
        leaving a genome that safety refused as the champion would let it seed
        every future generation while never actually running.
        """
        if self.previous_champion is None:
            return None
        restored = self.previous_champion
        record = {
            "generation": self.generation,
            "decision": "promotion_refused",
            "reason": str(reason),
            "restored_fitness": restored.get("fitness"),
            "time": CLOCK.now(),
        }
        self.champion = restored
        self.previous_champion = None
        self.promoted_at_tick = None
        self.metric_at_promotion = None
        self._append_lineage(record)
        self.save()
        logger.warning("A promoted champion was refused by the safety "
                       "pipeline: %s", reason)
        return record

    def _append_lineage(self, record: dict) -> None:
        self.lineage.append(record)
        if len(self.lineage) > MAX_LINEAGE:
            self.lineage = self.lineage[-MAX_LINEAGE:]

    # ── reporting ────────────────────────────────────────────────────

    def lineage_csv(self) -> str:
        """The lineage as CSV — the export §M10.1 asks for."""
        columns = ["generation", "decision", "best_fitness", "champion_fitness",
                   "valid_test_gap", "promoted", "time"]
        lines = [",".join(columns)]
        for row in self.lineage:
            lines.append(",".join(
                "" if row.get(column) is None else str(row.get(column, ""))
                for column in columns))
        return "\n".join(lines) + "\n"

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.EVO_GENERATION, self.generation, tick)
            self.telemetry.record(M.EVO_PROMOTIONS, self.promotions, tick)
            self.telemetry.record(M.EVO_ROLLBACKS, self.rollbacks, tick)
            self.telemetry.record(M.EVO_NOVELTY_SKIPS, self.archive.skips, tick)
            if self.champion:
                self.telemetry.record(M.EVO_CHAMPION_FITNESS,
                                      self.champion["fitness"], tick)
            self.telemetry.record(M.EVO_VALID_TEST_GAP,
                                  self.valid_test_gap or 0.0, tick)
        except Exception:
            logger.exception("Evolution metric publication failed")

    def population_report(self) -> dict:
        """The generation in full — the panel of §M10.1.

        Every variant with its fitness and origin, not just the winner. Which
        slot a genome came from is the part that says whether the composition
        rules are earning their place: a lineage where the cortex slot never
        wins is a lineage where that slot is spending tokens for nothing.
        """
        reports = [report.as_dict() for report in self.last_reports]
        for index, report in enumerate(reports):
            if index < len(self.population.origins):
                report["origin"] = self.population.origins[index]
        return {
            "generation": self.generation,
            "running": self.generation_running,
            "champion": self.champion,
            "valid_test_gap": self.valid_test_gap,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "variants": reports,
            "population": self.population.status(),
            "lineage": self.lineage[-20:],
        }

    def status(self) -> dict:
        return {
            "generation": self.generation,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "promotions": self.promotions,
            "rollbacks": self.rollbacks,
            "champion_fitness": self.champion["fitness"] if self.champion else None,
            "champion_genome": (self.champion or {}).get("genome"),
            "candidate_pending": self.candidate is not None,
            "candidate_param": self.candidate["mutated_param"] if self.candidate else None,
            "generation_running": self.generation_running,
            "valid_test_gap": self.valid_test_gap,
            "population": self.population.status(),
            "evaluator": self.evaluator.status(),
            "last_reports": [report.as_dict() for report in self.last_reports],
            "recent_lineage": self.lineage[-5:],
        }
