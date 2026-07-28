"""A generation: ten variants, evaluated, selected, archived (spec M5.6).

The composition is fixed by §M5.6 and each slot has a job:

* **2 elites**, unchanged. Without them a generation can be worse than the one
  before it, and the champion would drift downward on a run of bad luck.
* **4 coordinate mutations** — the ordinary search.
* **2 crossovers** — recombination of what the two elites each got right.
* **1 big step** — the only thing that leaves a converged neighbourhood.
* **1 cortex proposal** — a model reading the lineage. Rejected proposals fall
  back to a mutation, and the fallback is recorded, so the lineage says whether
  the cortex actually contributed.

Selection reads ``valid`` and only ``valid``. ``test`` confirms a champion once,
after selection is over. Keeping them apart is the entire protection against
evolution learning the validation set instead of the task, and
``valid_test_gap`` is what makes that visible rather than assumed.
"""
from __future__ import annotations

import logging

import aegis.config as cfg
from aegis.layers.evolution.genome import Genome
from aegis.layers.evolution.operators import (
    NoveltyArchive, coordinate_mutation, crossover, diversify, big_step,
    from_proposal,
)

logger = logging.getLogger("aegis.evolution")

#: What each slot of a generation is for. The counts sum to ``EVO_POP_SIZE``;
#: a different population size scales the mutations, because they are the part
#: there can sensibly be more or fewer of.
SLOT_ELITES = 2
SLOT_CROSSOVERS = 2
SLOT_BIG_STEPS = 1
SLOT_CORTEX = 1


def compose(elites: list[Genome], generation: int, size: int | None = None,
            proposal: dict | None = None) -> tuple[list[Genome], list[str]]:
    """Build one generation. Returns the genomes and what each slot was.

    The origins are returned alongside because "where did this champion come
    from" is a question the lineage has to be able to answer — an evolution
    whose wins all came from the big step is a different system from one whose
    wins came from crossover, and only the record can tell them apart.
    """
    size = int(cfg.EVO_POP_SIZE if size is None else size)
    size = max(1, size)
    elites = [Genome(elite) for elite in (elites or [])] or [Genome()]
    while len(elites) < SLOT_ELITES:
        elites.append(Genome(elites[0]))

    genomes: list[Genome] = []
    origins: list[str] = []

    for index in range(min(SLOT_ELITES, size)):
        genomes.append(Genome(elites[index]))
        origins.append("elite")

    remaining = size - len(genomes)
    crossovers = min(SLOT_CROSSOVERS, max(0, remaining - SLOT_BIG_STEPS - SLOT_CORTEX))
    big_steps = min(SLOT_BIG_STEPS, max(0, remaining - crossovers - SLOT_CORTEX))
    cortex_slots = min(SLOT_CORTEX, max(0, remaining - crossovers - big_steps))
    mutations = max(0, remaining - crossovers - big_steps - cortex_slots)

    for index in range(mutations):
        genomes.append(coordinate_mutation(elites[index % len(elites)],
                                           generation, index))
        origins.append("mutation")
    for index in range(crossovers):
        genomes.append(crossover(elites[0], elites[1], generation, index))
        origins.append("crossover")
    for index in range(big_steps):
        genomes.append(big_step(elites[0], generation, 100 + index))
        origins.append("big_step")
    for index in range(cortex_slots):
        child, accepted = from_proposal(
            proposal, coordinate_mutation(elites[0], generation, 200 + index))
        genomes.append(child)
        origins.append("cortex" if accepted else "cortex_rejected")

    return genomes, origins


class Population:
    """One generation, from composition through selection."""

    def __init__(self, archive: NoveltyArchive | None = None,
                 size: int | None = None, epsilon: float | None = None):
        self.archive = archive if archive is not None else NoveltyArchive()
        self.size = int(cfg.EVO_POP_SIZE if size is None else size)
        self.epsilon = float(cfg.EVO_EPSILON if epsilon is None else epsilon)
        self.genomes: list[Genome] = []
        self.origins: list[str] = []
        self.reports: list = []

    # ── building ─────────────────────────────────────────────────────

    def build(self, elites: list[Genome], generation: int,
              proposal: dict | None = None) -> list[Genome]:
        genomes, origins = compose(elites, generation, self.size, proposal)
        # Elites are exempt from the novelty filter: they are *meant* to be
        # what was already evaluated, and redrawing them would throw away the
        # one thing keeping the generation from regressing.
        elite_count = origins.count("elite")
        fresh = diversify(genomes[elite_count:], self.archive, generation,
                          Genome(elites[0]) if elites else Genome())
        self.genomes = genomes[:elite_count] + fresh
        self.origins = origins
        return self.genomes

    # ── selection ────────────────────────────────────────────────────

    def rank(self, reports) -> list[tuple[int, object]]:
        """Variants best first, ties broken by genome id.

        A total order matters more than it looks: two variants with identical
        fitness are common early on, and "whichever the sort happened to put
        first" would make the champion depend on list order.
        """
        self.reports = list(reports)
        indexed = list(enumerate(self.reports))
        return sorted(indexed,
                      key=lambda pair: (-pair[1].fitness, pair[1].genome_id))

    def elites_from(self, reports, count: int = SLOT_ELITES) -> list[Genome]:
        """The genomes that carry forward."""
        ranked = self.rank(reports)
        return [self.genomes[index] for index, _ in ranked[:count]
                if index < len(self.genomes)]

    def best(self, reports):
        ranked = self.rank(reports)
        if not ranked:
            return None, None
        index, report = ranked[0]
        genome = self.genomes[index] if index < len(self.genomes) else Genome()
        return genome, report

    def beats(self, challenger, champion_fitness: float | None) -> bool:
        """Whether a challenger is enough better to be worth promoting.

        The margin is what stops benchmark noise from promoting a sideways
        change every generation and calling it progress.
        """
        if challenger is None:
            return False
        if champion_fitness is None:
            return True
        return challenger.fitness > float(champion_fitness) + self.epsilon

    def record(self, reports) -> None:
        """Put everything evaluated into the archive — including the losers.

        The losers are data (§M5.6): they are what the discovery engine reads to
        learn which genes matter, and they are what stops the next generation
        proposing the same failures again.
        """
        for genome, report in zip(self.genomes, reports):
            self.archive.add(genome, report.fitness)

    def status(self) -> dict:
        return {
            "size": self.size,
            "epsilon": self.epsilon,
            "composition": {origin: self.origins.count(origin)
                            for origin in sorted(set(self.origins))},
            "archive": self.archive.status(),
        }
