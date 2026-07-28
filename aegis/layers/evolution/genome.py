"""The genome: parameters that provably change measured behaviour (spec M5.3).

The old genome was ``learning_rate``, ``dropout``, ``attention_heads`` — LoRA
hyper-parameters for a model the benchmark never touches. Evolution mutated them
faithfully for thousands of ticks and the measured score could not respond,
because nothing downstream read them. That is the failure this file exists to
correct: **a gene is a parameter some contour actually consults.**

Every gene here names its reader (Appendix C). A gene whose contour has not
landed yet is declared with the stage that will deliver it and excluded from the
sensitivity gate until then — the same rule the action registry uses, and for the
same reason: a contour under construction must be visible and inert rather than
silently absent.

Three protections are structural rather than conventional:

* ``safety_critical`` genes do not exist. Anything in ``IMMUTABLE_PARAMS``
  cannot be *named* here, and a schema that tried would fail its own test.
* Every value is clamped to its declared range on the way in, so a mutation, a
  crossover, a cortex proposal and a migrated file all arrive bounded.
* The resource shares are renormalised to sum to one, because four independent
  mutations of a simplex do not stay on the simplex.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aegis.safety import immutable
from aegis.util.stats import clamp

logger = logging.getLogger("aegis.evolution")

#: Bumped when the set of genes changes. A stored genome from an older schema is
#: migrated by taking what still exists and defaulting the rest (M5.3).
GENOME_SCHEMA_VERSION = 2

#: The last development stage whose contour reads its genes. Genes above it are
#: declared, defaulted, persisted — and left out of the sensitivity gate, which
#: would otherwise fail for a contour that does not exist yet.
DELIVERED_STAGE = 6


@dataclass(frozen=True)
class GeneSpec:
    """One evolvable parameter and the range it lives in."""

    name: str
    kind: str                       # "float" | "int" | "enum"
    default: object
    low: float = 0.0
    high: float = 1.0
    choices: tuple = ()
    reader: str = ""                # who consults it — Appendix C
    stage: int = 4                  # the stage that delivered its reader
    #: Shares that must sum to one across their group.
    simplex: str = ""

    def clamp(self, value):
        """Bring a proposed value inside the range, whatever it arrived as."""
        if self.kind == "enum":
            return value if value in self.choices else self.default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.default
        if number != number:                     # NaN
            return self.default
        number = clamp(number, self.low, self.high)
        return int(round(number)) if self.kind == "int" else number

    def span(self) -> float:
        """The width of the range, for normalised distances and step sizes."""
        if self.kind == "enum":
            return float(max(1, len(self.choices) - 1))
        return float(self.high - self.low) or 1.0

    def position(self, value) -> float:
        """Where a value sits in its range, on 0..1.

        Distances between genomes are computed here rather than on raw values,
        or a gene ranging over 100..5000 would drown out every gene ranging over
        0..1 and the novelty archive would only ever notice one of them.
        """
        if self.kind == "enum":
            try:
                return self.choices.index(value) / self.span()
            except ValueError:
                return 0.0
        try:
            return (float(value) - self.low) / self.span()
        except (TypeError, ValueError):
            return 0.0

    def at(self, position: float):
        """The value at a position on 0..1 — the inverse of :meth:`position`."""
        position = clamp(float(position), 0.0, 1.0)
        if self.kind == "enum":
            index = min(len(self.choices) - 1,
                        int(round(position * self.span())))
            return self.choices[index]
        return self.clamp(self.low + position * self.span())


SOLVER_ORDERS = ("by_success", "by_length", "by_recency")


def simplex_floor(count: int) -> float:
    """The guaranteed minimum for one member of a share group.

    Read from configuration rather than fixed here, and capped at an equal
    split: floors that together exceed the whole would leave nothing to
    distribute. This is the same definition ``ROITracker.floor`` uses, so the
    genome and the resource manager cannot disagree about what a legal
    allocation is.
    """
    import aegis.config as cfg

    return max(0.0, min(1.0 / max(1, count), float(cfg.RESOURCE_MIN_SHARE)))

#: Appendix C, in full. `reader` is not decoration: it is the claim that this
#: parameter changes behaviour, and `test_genome_sensitivity` checks it.
GENES: tuple[GeneSpec, ...] = (
    # ── the planner (M2) ─────────────────────────────────────────────
    GeneSpec("plan_beam", "int", 5, 1, 16, reader="Planner.plan", stage=4),
    GeneSpec("plan_depth", "int", 3, 1, 5, reader="Planner.plan", stage=4),
    GeneSpec("plan_discount", "float", 0.9, 0.5, 0.99,
             reader="Simulator.rollout", stage=4),
    GeneSpec("w_ev", "float", 1.0, 0.0, 2.0, reader="Planner._score", stage=4),
    GeneSpec("w_val", "float", 0.6, 0.0, 2.0, reader="Planner._score", stage=4),
    GeneSpec("w_exp", "float", 0.3, 0.0, 2.0, reader="Planner._score", stage=4),
    GeneSpec("w_cost", "float", 0.4, 0.0, 2.0, reader="Planner._score", stage=4),
    GeneSpec("w_risk", "float", 0.5, 0.0, 2.0, reader="Planner._score", stage=4),
    # ── the behaviour policy (M3) ────────────────────────────────────
    GeneSpec("policy_weight", "float", 0.3, 0.0, 1.0,
             reader="PolicyStore.delta", stage=5),
    GeneSpec("policy_min_support", "int", 20, 5, 100,
             reader="RuleMiner.mine", stage=5),
    # ── the world model (M1) ─────────────────────────────────────────
    GeneSpec("explore_bonus", "float", 0.15, 0.0, 0.5,
             reader="PredictiveWorldModel.knows", stage=4),
    GeneSpec("wm_smoothing", "float", 1.0, 0.1, 5.0,
             reader="TransitionModel", stage=4),
    GeneSpec("wm_half_life", "int", 500, 100, 5000,
             reader="TransitionModel forgetting", stage=4),
    # ── the capability layer (M9) ────────────────────────────────────
    GeneSpec("solver_timeout", "float", 3.0, 0.5, 10.0,
             reader="MultiAgentSolver.timeout", stage=6),
    GeneSpec("solver_order", "enum", "by_success", choices=SOLVER_ORDERS,
             reader="SkillLibrary.for_kind ordering", stage=6),
    GeneSpec("synth_attempts", "int", 2, 1, 4,
             reader="Substrate._skill_synthesis", stage=6),
    # ── reasoning (M6) — declared, delivered by stages 8-9 ───────────
    GeneSpec("reason_budget", "int", 6, 1, 8, reader="interpreter.run", stage=8),
    GeneSpec("reason_vote_n", "int", 1, 1, 5, reader="VOTE", stage=8),
    GeneSpec("reason_ucb_c", "float", 1.4, 0.2, 3.0,
             reader="strategy selection", stage=9),
    # ── memory ───────────────────────────────────────────────────────
    GeneSpec("mem_retention_bias", "float", 1.5, 0.5, 3.0,
             reader="WorldModel._retention_score", stage=4),
    # ── priority (M4) ────────────────────────────────────────────────
    GeneSpec("priority_w_value", "float", 1.0, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    GeneSpec("priority_w_urgency", "float", 0.7, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    GeneSpec("priority_w_drive", "float", 0.5, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    GeneSpec("priority_w_aging", "float", 0.3, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    GeneSpec("priority_w_plan", "float", 0.8, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    GeneSpec("priority_w_cost", "float", 0.4, 0.0, 2.0,
             reader="PriorityScheduler.priority", stage=4),
    # ── resource shares (M4) — a simplex ─────────────────────────────
    GeneSpec("res_share_competence", "float", 0.35, 0.05, 0.7,
             reader="ResourceManager split", stage=4, simplex="drive"),
    GeneSpec("res_share_knowledge", "float", 0.30, 0.05, 0.7,
             reader="ResourceManager split", stage=4, simplex="drive"),
    GeneSpec("res_share_coherence", "float", 0.20, 0.05, 0.7,
             reader="ResourceManager split", stage=4, simplex="drive"),
    GeneSpec("res_share_stability", "float", 0.15, 0.05, 0.7,
             reader="ResourceManager split", stage=4, simplex="drive"),
)

GENES_BY_NAME: dict[str, GeneSpec] = {gene.name: gene for gene in GENES}

#: Gene names in a fixed order. Everything that walks the genome — mutation,
#: crossover, distance, the digest — walks this, so two runs agree.
GENE_NAMES: tuple[str, ...] = tuple(sorted(GENES_BY_NAME))

#: LoRA parameters that used to be genes. They stay in
#: ``SelfModification.parameters`` and remain reachable by parametric
#: self-modification; they are simply no longer selected on, because nothing the
#: benchmark measures reads them (§M5.3).
RETIRED_GENES: frozenset[str] = frozenset({
    "learning_rate", "dropout", "attention_heads", "temperature",
    "curiosity_weight", "memory_decay",
})


def defaults() -> dict:
    return {gene.name: gene.default for gene in GENES}


def observable_genes() -> tuple[str, ...]:
    """Genes whose reader exists in the current build.

    The sensitivity gate runs over these. A gene declared for a later stage is
    honestly inert, and demanding that it move the fitness would mean either
    deleting it or faking a reader for it.
    """
    return tuple(name for name in GENE_NAMES
                 if GENES_BY_NAME[name].stage <= DELIVERED_STAGE)


class Genome(dict):
    """A validated set of gene values. Always in range, always complete."""

    def __init__(self, values: dict | None = None):
        super().__init__(defaults())
        if values:
            self.update_values(values)
        else:
            # Normalise even the defaults, so every genome is in the same
            # canonical form. Otherwise `Genome()` and a genome that arrived
            # through `update_values` differ in the last decimal of the shares
            # and compare unequal for no reason anybody can see.
            self.normalise_simplexes()

    # ── construction ─────────────────────────────────────────────────

    def update_values(self, values: dict) -> None:
        """Take what is usable and clamp it. Unknown keys are dropped.

        Dropping rather than raising: values arrive from disk, from a migration
        and from a model, and a genome that refused to load because one key was
        strange would take the whole lineage with it.
        """
        for name, value in (values or {}).items():
            gene = GENES_BY_NAME.get(str(name))
            if gene is None:
                if str(name) not in RETIRED_GENES:
                    logger.debug("Dropping unknown gene %r", name)
                continue
            self[gene.name] = gene.clamp(value)
        self.normalise_simplexes()

    def normalise_simplexes(self) -> None:
        """Renormalise each share group to sum to one, above the floor.

        Four independent mutations of a simplex do not stay on the simplex, and
        a resource split summing to 1.4 would hand out more budget than exists.

        The floor is handed out first and only the remainder is split
        proportionally — the same rule ``ROITracker._normalized`` uses, and
        deliberately the same one. Two different normalisations for the same
        four numbers meant a share written as 0.05 came back as 0.09, so a
        genome could not be reverted to exactly what it was, and a rejected
        variant left a trace behind every generation.
        """
        groups: dict[str, list[str]] = {}
        for gene in GENES:
            if gene.simplex:
                groups.setdefault(gene.simplex, []).append(gene.name)
        for names in groups.values():
            names = sorted(names)
            floor = simplex_floor(len(names))
            remainder = 1.0 - floor * len(names)
            weights = {name: max(0.0, float(self[name]) - floor) for name in names}
            total = sum(weights.values())
            if total <= 0:
                for name in names:
                    self[name] = 1.0 / len(names)
                continue
            for name in names:
                self[name] = floor + remainder * (weights[name] / total)

    # ── comparison ───────────────────────────────────────────────────

    def position_vector(self) -> list[float]:
        """The genome as a point in the unit cube, in gene-name order."""
        return [GENES_BY_NAME[name].position(self[name]) for name in GENE_NAMES]

    def distance(self, other: "Genome") -> float:
        """Normalised Euclidean distance, on 0..1.

        Normalised by the number of genes so the threshold in
        ``EVO_MIN_DISTANCE`` means the same thing however many genes there are.
        """
        mine, theirs = self.position_vector(), other.position_vector()
        if not mine:
            return 0.0
        squared = sum((a - b) ** 2 for a, b in zip(mine, theirs))
        return (squared / len(mine)) ** 0.5

    def digest(self) -> str:
        """Stable identity, so the same genome is recognised as already seen."""
        from aegis.util.canonical import digest_of

        return digest_of({name: self[name] for name in GENE_NAMES})[:16]

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {name: self[name] for name in GENE_NAMES}

    def stored(self) -> dict:
        return {"schema_version": GENOME_SCHEMA_VERSION, "genes": self.to_dict()}

    @classmethod
    def from_stored(cls, data: dict | None) -> "Genome":
        """Rebuild from disk, migrating an older schema forward.

        A v1 file is the *old* genome — LoRA hyper-parameters that are no longer
        genes. Migration therefore keeps nothing and seeds the defaults, which
        is the honest outcome: there is no correspondence to preserve.
        """
        data = data or {}
        if not isinstance(data, dict):
            return cls()
        genes = data.get("genes")
        if not isinstance(genes, dict):
            # A bare mapping of values, or the v1 shape.
            genes = data if all(isinstance(k, str) for k in data) else {}
        return cls(genes)


def assert_no_immutable_genes() -> None:
    """No gene may name a protected parameter (Appendix B).

    Called at import so a schema that grew a forbidden gene fails loudly at the
    first import rather than quietly at the first mutation.
    """
    offenders = sorted(name for name in GENE_NAMES if immutable.is_immutable(name))
    if offenders:
        raise immutable.ImmutableParameterError(
            f"genes name protected parameters: {offenders}")


assert_no_immutable_genes()
