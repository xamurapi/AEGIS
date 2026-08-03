"""The metacognition contour (spec M11): attribution and invention.

The meta-loop over M6. Its inputs are the reasoning contour's outcomes — which
strategy won, against whom, on what weakness; its output is a change to **how
strategies are generated**: the mechanism-credit table reorders synthesis, and
the skeleton catalogue supplies candidates that are structurally far from
everything already tried. M6 changes strategies; this changes the generator.

One object, four jobs:

* **Attribution** (M11.5): for an accepted strategy, measure by ablation which
  of its edits carried the win, and record an :class:`Explanation` whose
  mechanism is computed, never narrated into existence.
* **Credit** (M11.6.3): every arena verdict feeds the mechanism-credit table,
  and :meth:`order_for` hands the synthesiser a UCB-ordered generator list —
  the measurable "behaviour changed" of this module.
* **Invention** (M11.6.2, M11.6.4): a deterministic quota of each round's
  candidates comes from the skeleton catalogue and must sit ``META_FAR`` away
  from the whole archive. Acceptance gates are M6.8's, untouched — novelty
  buys an evaluation, never a pass.
* **Memory** (M11.6.5, M11.7.8): permanent skeleton retirements per
  ``(skeleton, features)``, persisted beside the explanations and the table.

Everything here is deterministic (§3.1): no RNG, hash indexing for samples,
canonical tie-breaks for orders. The heavy step — ablation — runs through the
evaluation pool, never inside a tick (§3.4); the tick-side hook
:meth:`on_reflect` is bookkeeping only.

``META_ENABLED`` is False by default. Disabled, the contour observes nothing,
orders nothing and proposes nothing — the reasoning contour behaves exactly as
it did before M11 (acceptance criterion 11).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import aegis.config as cfg
from aegis.layers.metacognition.attribution import (
    Edit, EditAttribution, Explanation, ablation_worker,
    apply_narrative, attribute_edits, conclude, diff, revert,
)
from aegis.layers.metacognition.distance import (
    StrategyArchive, canonical_hash, canonicalize, distance,
)
from aegis.layers.metacognition.mechanism import (
    FIXED_TRANSFORM_ORDER, MECHANISMS, CreditTable, mechanism_for,
    mechanism_of_transform,
)
from aegis.layers.metacognition.skeletons import (
    SKELETON_NAMES, SkeletonCatalog, fill,
)
from aegis.layers.metacognition.store import MetaStore, evict

logger = logging.getLogger("aegis.metacognition")

__all__ = ["MetaCognition", "Explanation", "EditAttribution", "Edit",
           "StrategyArchive", "CreditTable", "SkeletonCatalog",
           "MECHANISMS", "SKELETON_NAMES", "diff", "revert", "distance",
           "canonicalize", "canonical_hash", "mechanism_for",
           "mechanism_of_transform", "fill", "ablation_worker",
           "apply_narrative", "attribute_edits", "conclude", "evict"]


def _features_of(weakness) -> tuple[str, ...]:
    """The feature labels credit is keyed on — a combo, or a parsed label."""
    combo = getattr(weakness, "combo", None)
    if combo:
        return tuple(sorted(str(part) for part in combo))
    label = getattr(weakness, "label", None) or str(weakness or "")
    return tuple(sorted(part.strip() for part in label.split(" AND ")
                        if part.strip()))


class MetaCognition:
    """Facade the substrate talks to: scan, attribute, invent, report."""

    def __init__(self, *, reasoning, pool=None, telemetry=None, cortex=None,
                 store_dir: Path | None = None, enabled: bool | None = None):
        self.reasoning = reasoning
        self.pool = pool
        self.telemetry = telemetry
        self.cortex = cortex
        self.enabled = bool(cfg.META_ENABLED if enabled is None else enabled)
        self.store = MetaStore(store_dir)

        self.credit, self.archive = self.store.load_credit()
        self.skeletons = self.store.load_retired()
        self.explanations: list[Explanation] = self.store.load_explanations()

        #: Strategy names waiting for attribution, oldest first.
        self.attribution_queue: list[str] = []
        #: How far into ``reasoning.verdicts`` the credit scan has read.
        self._verdict_cursor = 0
        #: Genome-readable knobs, held by their consumers where one exists
        #: (mechanism_c lives on the credit table); the rest live here because
        #: this object is what consumes them.
        self._far_share = float(cfg.META_FAR_SHARE)
        self._min_effect = float(cfg.META_MIN_EFFECT)
        self._ablation_n = int(cfg.META_ABLATION_N)

        self.far_issued = 0
        self.far_accepted = 0
        self.contested = 0
        self.attributions_run = 0
        self.invented = 0

        # Everything the library already holds has, by definition, been
        # evaluated; seeding the archive with it is what makes "far from
        # everything tried" include the built-ins.
        for strategy in self.reasoning.library.strategies.values():
            self.archive.add(strategy.name, strategy.steps)

        if self.enabled:
            self._attach()

    def _attach(self) -> None:
        """Give the synthesiser the credit-driven order (M11.6.3).

        Only under ``META_ENABLED``: with the hook absent the synthesiser
        walks its fixed tuple exactly as before M11.
        """
        self.reasoning.synthesiser.order_hook = self.order_for

    # ── genes (M11.7.4) ──────────────────────────────────────────────

    def set_genome(self, genome: dict) -> None:
        genome = dict(genome or {})
        try:
            self._far_share = max(0.0, min(0.5, float(
                genome.get("meta_far_share", self._far_share))))
        except (TypeError, ValueError):
            pass
        try:
            self._min_effect = max(0.01, min(0.10, float(
                genome.get("meta_min_effect", self._min_effect))))
        except (TypeError, ValueError):
            pass
        try:
            self._ablation_n = max(20, min(200, int(
                genome.get("meta_ablation_n", self._ablation_n))))
        except (TypeError, ValueError):
            pass
        try:
            self.credit.mechanism_c = max(0.0, min(2.0, float(
                genome.get("meta_mechanism_c", self.credit.mechanism_c))))
        except (TypeError, ValueError):
            pass

    # Read-backs, from the objects that consume the values — what
    # ``Substrate._genome_from_contours`` reports (M11.7.4 rule 1).

    @property
    def far_share(self) -> float:
        return self._far_share

    @property
    def min_effect(self) -> float:
        return self._min_effect

    @property
    def ablation_n(self) -> int:
        return self._ablation_n

    @property
    def mechanism_c(self) -> float:
        return self.credit.mechanism_c

    # ── the tick-side hook: bookkeeping only (M11.7.3) ───────────────

    def on_reflect(self, tick: int = 0) -> None:
        """Fold new verdicts into credit and queue unexplained strategies.

        Counter arithmetic over a bounded list — no ablation, no interpreter
        run, no disk. The ≤ 3 ms REFLECT share the spec allots is why.
        """
        if not self.enabled:
            return
        verdicts = self.reasoning.verdicts
        if self._verdict_cursor > len(verdicts):
            self._verdict_cursor = 0        # the list was trimmed under us
        for record in verdicts[self._verdict_cursor:]:
            self._credit_verdict(record)
        self._verdict_cursor = len(verdicts)
        self._queue_unexplained()

    def _credit_verdict(self, record: dict) -> None:
        transform = str(record.get("transform", ""))
        mechanism = mechanism_of_transform(transform)
        if not mechanism:
            return
        features = _features_of(str(record.get("weakness", "")))
        accepted = bool(record.get("accepted"))
        self.credit.note_attempt(mechanism, features)
        if accepted:
            self.credit.note_accepted(mechanism, features,
                                      float(record.get("holdout_gain", 0.0)))
        if transform.startswith("skeleton:"):
            name = transform.split(":", 1)[1]
            if accepted:
                self.far_accepted += 1
                self.skeletons.note_success(name, features)
            else:
                self.skeletons.note_failure(name, features)

    def _queue_unexplained(self) -> None:
        explained = {e.strategy for e in self.explanations}
        queued = set(self.attribution_queue)
        for name, strategy in sorted(self.reasoning.library.strategies.items()):
            if strategy.builtin or strategy.retired:
                continue
            if name in explained or name in queued:
                continue
            self.attribution_queue.append(name)

    def pending_attributions(self) -> list[str]:
        return list(self.attribution_queue)

    # ── attribution (M11.5): the ``attribute_strategy`` action ───────

    async def attribute(self, tick: int = 0) -> dict | None:
        """Explain the oldest accepted-but-unexplained strategy.

        The ablation itself runs through the evaluation pool — a batch of one
        picklable payload — because ``META_MAX_EDITS × META_ABLATION_N``
        interpreter passes are seconds, and seconds leave the tick (§3.4).
        """
        if not self.enabled or not self.attribution_queue:
            return None
        name = self.attribution_queue.pop(0)
        strategy = self.reasoning.library.get(name)
        if strategy is None:
            return None
        explanation = self._attribute_one(strategy, tick=tick)
        explanation = await self._narrate(explanation, strategy)
        if explanation.status == "contested":
            self.contested += 1
        self.explanations.append(explanation)
        self.explanations = evict(self.explanations,
                                  cap=int(cfg.META_MAX_EXPLANATIONS))
        self.attributions_run += 1
        if explanation.mechanism:
            best = max(explanation.confirmed_edits(),
                       key=lambda a: a.effect, default=None)
            if best is not None:
                self.credit.note_confirmed_effect(
                    explanation.mechanism,
                    _features_of(explanation.weakness), best.effect)
        if self.telemetry is not None:
            from aegis.telemetry import metrics as M
            try:
                self.telemetry.record(M.META_EXPLANATIONS, 1, tick,
                                      tags={"status": explanation.status})
                if explanation.status == "contested":
                    self.telemetry.record(M.META_CONTESTED, 1, tick)
            except Exception:
                logger.exception("Metacognition telemetry failed")
        return explanation.as_dict()

    def _attribute_one(self, strategy, *, tick: int = 0) -> Explanation:
        """The measured half: diff, ablate (in the pool), correct, conclude."""
        family = strategy.family or ""
        incumbent = (self.reasoning.library.get(strategy.incumbent)
                     or self.reasoning.library.best_for(family)
                     or self.reasoning.library.get("direct"))
        incumbent_steps = incumbent.steps if incumbent is not None else []
        edits = diff(incumbent_steps, strategy.steps)
        gain = 0.0
        for record in reversed(self.reasoning.verdicts):
            if record.get("candidate") == strategy.name:
                gain = float(record.get("holdout_gain", 0.0))
                break

        if not edits or len(edits) > int(cfg.META_MAX_EDITS):
            # Too many simultaneous edits to measure one at a time. A limit on
            # explainability, not on acceptance — the arena already accepted.
            return conclude(strategy.name,
                            getattr(incumbent, "name", ""),
                            strategy.weakness, gain, (), too_many_edits=True,
                            tick=tick)

        payload = {
            "candidate": canonicalize(strategy.steps),
            "digest": canonical_hash(strategy.steps),
            "family": family,
            "n": self._ablation_n,
            "budget": self.reasoning._budget(),
            "genome": dict(self.reasoning.genome),
            "edits": [{"signature": edit.signature(),
                       "reverted": revert(strategy.steps, edit)}
                      for edit in edits],
        }
        measured = self._run_ablation(payload)
        attributions = attribute_edits(edits, measured,
                                       min_effect=self._min_effect)
        return conclude(strategy.name, getattr(incumbent, "name", ""),
                        strategy.weakness, gain, attributions, tick=tick)

    def _run_ablation(self, payload: dict) -> dict:
        """Hand the measurement to the pool; inline only when there is none.

        The pool itself may still degrade to in-process execution when a lease
        is refused — that is the pool's honest fallback (M9.1), and the
        distinction the budget test pins is that *the tick path never scores
        tasks*: scoring happens inside ``map``, under the pool's accounting.
        """
        if self.pool is not None:
            results = self.pool.map(ablation_worker, [payload],
                                    purpose="meta_ablation")
            outcome = results[0].value if results and results[0].ok else None
        else:
            outcome = ablation_worker(payload)
        rows = (outcome or {}).get("rows") or []
        return {str(row.get("signature", "")): row for row in rows
                if isinstance(row, dict)}

    async def _narrate(self, explanation: Explanation, strategy) -> Explanation:
        """Step 6: the cortex adds a story and a guess; it moves no number."""
        if self.cortex is None or not hasattr(self.cortex, "structured"):
            return explanation
        try:
            if not self.cortex.role_available("deep"):
                return explanation
            reply = await self.cortex.structured(
                "deep",
                [{"role": "user",
                  "content": self._narrative_prompt(explanation, strategy)}],
                "meta_explanation")
        except Exception:
            logger.exception("The cortex narrative path failed")
            return explanation
        if not isinstance(reply, dict):
            return explanation
        return apply_narrative(explanation,
                               str(reply.get("narrative", "") or ""),
                               str(reply.get("mechanism", "") or ""))

    @staticmethod
    def _narrative_prompt(explanation: Explanation, strategy) -> str:
        rows = "\n".join(
            f"  - {a.edit.kind} {a.edit.op} at {a.edit.position}: "
            f"effect {a.effect:+.3f} (n={a.n}, p={a.p_value:.4f}, "
            f"confirmed={a.confirmed})"
            for a in explanation.edits)
        return (
            "A reasoning strategy was accepted and its win has already been "
            "attributed by ablation. Explain, for a human, why it won.\n\n"
            f"Strategy: {explanation.strategy}\n"
            f"Incumbent: {explanation.incumbent}\n"
            f"Weakness: {explanation.weakness}\n"
            f"Held-out gain: {explanation.gain:+.3f}\n"
            f"Measured edit effects:\n{rows or '  (none)'}\n\n"
            "Answer with a short narrative and, in `mechanism`, the one label "
            f"from this closed list you believe applies: {list(MECHANISMS)}. "
            "The measured numbers are final; your label is a hypothesis that "
            "will be compared against the computed one."
        )

    # ── ordering (M11.6.3): what the synthesiser consults ────────────

    def order_for(self, weakness) -> tuple[str, ...]:
        """Transformation order for this weakness, best-credited first.

        Only measured credit speaks here: an ``unsupported`` explanation put
        nothing into the table, so it cannot move this order — the invariant
        of M11.4, enforced by construction rather than by a check.
        """
        return self.credit.order(_features_of(weakness),
                                 names=FIXED_TRANSFORM_ORDER)

    def order_delta(self) -> float:
        """Share of current weaknesses whose order differs from the fixed one.

        Zero means the table is decorative — the exact defect class the fifth
        audit round found in genes nobody read (M11.6.3).
        """
        found = getattr(self.reasoning.detector, "last", []) or []
        if not found:
            return 0.0
        moved = sum(
            1 for weakness in found
            if self.credit.order_differs(_features_of(weakness),
                                         FIXED_TRANSFORM_ORDER))
        return moved / len(found)

    # ── invention (M11.6.2-M11.6.5): the ``invent_strategy`` action ──

    def far_quota(self) -> int:
        """⌈K × META_FAR_SHARE⌉ of one round's candidates (M11.6.4)."""
        k = int(getattr(self.reasoning.synthesiser, "max_candidates", 6))
        return max(0, math.ceil(k * self._far_share))

    def quota_open(self) -> bool:
        """Whether the far quota still has room among pending candidates."""
        pending_far = sum(
            1 for candidate in self.reasoning.candidates
            if str(getattr(candidate, "transform", "")).startswith("skeleton:"))
        return pending_far < self.far_quota()

    def invent(self, tick: int = 0) -> list[dict]:
        """Propose structurally different candidates for the found weaknesses.

        Skeletons in credit order, filled deterministically, each required to
        sit ``META_FAR`` from the entire archive. The walk starts at the
        rank-1 weakness and moves down only when a rank is exhausted — its
        skeleton pairs retired, or its shapes already archived — because a
        pair retired *for those features* means "look elsewhere", not "stop
        inventing" (M11.6.5). Admission and acceptance are M6's, unchanged:
        these candidates join the same queue and face the same arena. Novelty
        buys the evaluation, nothing else.
        """
        if not self.enabled:
            return []
        if not getattr(self.reasoning, "found", None):
            self.reasoning.scan_weakness()
        weaknesses = list(self.reasoning.found or [])
        if not weaknesses:
            return []
        quota = self.far_quota()
        if quota <= 0:
            return []

        from aegis.layers.reasoning.synthesis import Candidate

        library_digests = {s.digest
                           for s in self.reasoning.library.strategies.values()}
        pending_digests = {c.digest for c in self.reasoning.candidates}
        issued: list[Candidate] = []
        for weakness in weaknesses:
            if len(issued) >= quota:
                break
            features = _features_of(weakness)
            label = getattr(weakness, "label", str(weakness))
            names = [f"skeleton:{name}"
                     for name in self.skeletons.available(features)]
            for ranked in self.credit.order(features, names=names):
                if len(issued) >= quota:
                    break
                name = ranked.split(":", 1)[1]
                try:
                    steps = fill(name, features, self.credit)
                except ValueError:
                    logger.exception("Skeleton %r failed to fill", name)
                    continue
                if self.archive.min_distance(steps) < float(cfg.META_FAR):
                    self.archive.note_skip()
                    continue
                candidate = Candidate(
                    name=f"far-{name}-{canonical_hash(steps)[:6]}",
                    steps=steps, parent="", weakness=label, origin="synth",
                    transform=f"skeleton:{name}", created_tick=int(tick))
                if candidate.digest in library_digests | pending_digests:
                    continue
                pending_digests.add(candidate.digest)
                self.archive.add(candidate.name, steps)
                self.reasoning.candidates.append(candidate)
                issued.append(candidate)

        self.far_issued += len(issued)
        self.invented += len(issued)
        return [candidate.as_dict() for candidate in issued]

    # ── reporting, persistence, integration (M11.7) ──────────────────

    def explanation_for(self, strategy: str) -> dict | None:
        for explanation in reversed(self.explanations):
            if explanation.strategy == str(strategy):
                return explanation.as_dict()
        return None

    def explanations_report(self) -> list[dict]:
        return [explanation.as_dict() for explanation in self.explanations]

    def _status_counts(self) -> dict[str, int]:
        counts = {"supported": 0, "unsupported": 0, "contested": 0}
        for explanation in self.explanations:
            counts[explanation.status] = counts.get(explanation.status, 0) + 1
        return counts

    def status(self) -> dict:
        counts = self._status_counts()
        return {
            "enabled": self.enabled,
            "explanations": len(self.explanations),
            "supported": counts["supported"],
            "unsupported": counts["unsupported"],
            "contested": counts["contested"],
            "pending": len(self.attribution_queue),
            "attributions_run": self.attributions_run,
            "order_delta": round(self.order_delta(), 4),
            "far_issued": self.far_issued,
            "far_accepted": self.far_accepted,
            "invented": self.invented,
            "far_quota": self.far_quota(),
            "top_mechanisms": self.credit.win_rates(),
            "retired_skeletons": self.skeletons.retired_report(),
            "archive": self.archive.status(),
            "credit_rows": len(self.credit.rows),
        }

    def snapshot(self) -> dict:
        """What the determinism digest must see (§M11.7.1, M9.4)."""
        return {
            "enabled": self.enabled,
            "explanations": [e.as_dict() for e in self.explanations],
            "credit": self.credit.to_dict(),
            "retired": self.skeletons.to_dict(),
            "archive_hashes": sorted(entry["hash"]
                                     for entry in self.archive.entries),
            "queue": list(self.attribution_queue),
        }

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            counts = self._status_counts()
            for status, value in sorted(counts.items()):
                self.telemetry.record(M.META_EXPLANATIONS, value, tick,
                                      tags={"status": status})
            confirmed = sum(len(e.confirmed_edits()) for e in self.explanations)
            self.telemetry.record(M.META_CONFIRMED_EDITS, confirmed, tick)
            self.telemetry.record(M.META_CONTESTED, self.contested, tick)
            self.telemetry.record(M.META_ORDER_DELTA, self.order_delta(), tick)
            self.telemetry.record(M.META_FAR_SHARE, self._far_share, tick)
            self.telemetry.record(M.META_FAR_ACCEPTED, self.far_accepted, tick)
            self.telemetry.record(M.META_RETIRED_SKELETONS,
                                  len(self.skeletons.retired_report()), tick)
            attempts = sum(row.attempts for row in self.credit.rows.values())
            accepted = sum(row.accepted for row in self.credit.rows.values())
            self.telemetry.record(M.META_CANDIDATES_TO_ACCEPT,
                                  attempts / accepted if accepted else 0.0,
                                  tick)
            rates = self.credit.win_rates()
            for mechanism in MECHANISMS:
                entry = rates.get(mechanism) or {"win_rate": 0.0}
                self.telemetry.record(M.META_MECHANISM_WIN_RATE,
                                      entry["win_rate"], tick,
                                      tags={"mechanism": mechanism})
        except Exception:
            logger.exception("Metacognition metric publication failed")

    def save(self) -> None:
        self.store.save_explanations(self.explanations)
        self.store.save_credit(self.credit, self.archive)
        self.store.save_retired(self.skeletons)
