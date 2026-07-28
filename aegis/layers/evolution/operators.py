"""Making the next generation, without a random number generator (spec M5.4).

Evolution needs variation, and variation normally means randomness — which §3.1
forbids, because two runs of the same experiment have to produce the same
answer. So variation comes from low-discrepancy sequences and hashes instead:

* **Halton** for coordinate mutation. A Halton point spreads more evenly over
  the search space than a random one, which is exactly what a small population
  needs — ten random samples in thirty dimensions cluster, ten Halton points do
  not.
* **Hashes** for the discrete choices: which genes a crossover takes from which
  parent, which gene a big step moves. Deterministic in the generation number,
  so generation seven is always the same generation seven.

Every operator returns a *clamped* genome, and every operator is a pure function
of its inputs. There is nowhere for a hidden state to accumulate.
"""
from __future__ import annotations

import logging

import aegis.config as cfg
from aegis.layers.evolution.genome import (
    GENE_NAMES, GENES_BY_NAME, Genome,
)
from aegis.util.quasirandom import halton, hash_index, hash_unit, signed

logger = logging.getLogger("aegis.evolution")

#: A big step is this many times the ordinary amplitude (§M5.4). Large enough to
#: leave the neighbourhood the population is sitting in, which is the only way a
#: converged population ever finds anything else.
BIG_STEP_FACTOR = 4.0


def coordinate_mutation(parent: Genome, generation: int, index: int,
                        sigma: float | None = None) -> Genome:
    """Move every gene by a Halton-driven step of amplitude ``sigma``.

    Amplitude is in *fractions of each gene's range*, so a gene spanning
    100..5000 and a gene spanning 0..1 are perturbed comparably. Mutating raw
    values would leave the narrow genes effectively frozen.
    """
    sigma = float(cfg.EVO_SIGMA if sigma is None else sigma)
    point = halton(_sequence_index(generation, index), len(GENE_NAMES))
    child = Genome(parent)
    for offset, name in enumerate(GENE_NAMES):
        gene = GENES_BY_NAME[name]
        step = signed(point[offset]) * sigma
        child[name] = gene.at(gene.position(parent[name]) + step)
    child.normalise_simplexes()
    return child


def big_step(parent: Genome, generation: int, index: int,
             sigma: float | None = None) -> Genome:
    """One variant per generation that jumps rather than steps (§M5.4)."""
    sigma = float(cfg.EVO_SIGMA if sigma is None else sigma)
    return coordinate_mutation(parent, generation, index,
                               sigma=sigma * BIG_STEP_FACTOR)


def crossover(first: Genome, second: Genome, generation: int,
              index: int) -> Genome:
    """Uniform crossover: each gene comes from one parent or the other.

    Which one is decided by a hash of (generation, index, gene name) — so the
    mix is arbitrary but reproducible, and two different children of the same
    pair in the same generation are genuinely different.
    """
    child = Genome()
    for name in GENE_NAMES:
        take_first = hash_unit("evo_crossover", generation, index, name) < 0.5
        child[name] = first[name] if take_first else second[name]
    child.normalise_simplexes()
    return child


def from_proposal(proposal: dict | None, fallback: Genome) -> tuple[Genome, bool]:
    """Take a genome a model proposed, or fall back.

    Returns ``(genome, accepted)``. A proposal is accepted when it is a mapping
    naming at least one real gene; everything unusable in it is dropped and
    everything out of range is clamped, so an accepted proposal is still a legal
    genome. §M5.4 requires the fallback, and it has to be visible: silently
    substituting a mutation for a rejected proposal would make the cortex look
    useful in the lineage whether or not it was.
    """
    if not isinstance(proposal, dict):
        return Genome(fallback), False
    usable = {name: value for name, value in proposal.items()
              if name in GENES_BY_NAME}
    if not usable:
        return Genome(fallback), False
    child = Genome(fallback)
    child.update_values(usable)
    return child, True


def _sequence_index(generation: int, index: int) -> int:
    """Where in the Halton sequence this variant draws from.

    Offset by the generation so successive generations do not re-draw the same
    points — a population that mutated by the same offsets every generation
    would explore one direction forever.
    """
    return max(1, int(generation) * 16 + int(index) + 1)


class NoveltyArchive:
    """Remembers what has been evaluated, so nothing is evaluated twice.

    Evaluation is the expensive step — seconds to minutes per variant — and a
    converged population proposes near-duplicates constantly. Skipping them is
    not an optimisation but the difference between a generation that explores
    ten things and one that explores the same thing ten times.
    """

    def __init__(self, min_distance: float | None = None, capacity: int = 500):
        self.min_distance = float(
            cfg.EVO_MIN_DISTANCE if min_distance is None else min_distance)
        self.capacity = int(capacity)
        self.entries: list[dict] = []
        self.skips = 0

    def is_novel(self, genome: Genome) -> bool:
        """Whether this genome is far enough from everything already seen."""
        for entry in self.entries:
            if genome.distance(Genome(entry["genes"])) < self.min_distance:
                return False
        return True

    def add(self, genome: Genome, fitness: float | None = None) -> None:
        self.entries.append({"digest": genome.digest(),
                             "genes": genome.to_dict(),
                             "fitness": fitness})
        if len(self.entries) > self.capacity:
            # Drop the oldest: the archive is about where the search has been
            # recently, and an ancient point no longer describes the frontier.
            self.entries = self.entries[-self.capacity:]

    def note_skip(self) -> None:
        self.skips += 1

    def to_dict(self) -> dict:
        return {"min_distance": self.min_distance, "skips": self.skips,
                "entries": list(self.entries)}

    @classmethod
    def from_dict(cls, data: dict | None) -> "NoveltyArchive":
        data = data or {}
        archive = cls(min_distance=data.get("min_distance"))
        try:
            archive.skips = max(0, int(data.get("skips", 0)))
        except (TypeError, ValueError):
            archive.skips = 0
        for entry in data.get("entries") or []:
            if isinstance(entry, dict) and isinstance(entry.get("genes"), dict):
                archive.entries.append({
                    "digest": str(entry.get("digest", "")),
                    "genes": dict(entry["genes"]),
                    "fitness": entry.get("fitness"),
                })
        return archive

    def status(self) -> dict:
        return {"size": len(self.entries), "skips": self.skips,
                "min_distance": self.min_distance}


def diversify(candidates: list[Genome], archive: NoveltyArchive,
              generation: int, elite: Genome) -> list[Genome]:
    """Replace near-duplicates with fresh draws, keeping the count.

    A generation that shrank because half of it was too similar would make the
    population size depend on how converged the search happened to be, and every
    comparison between generations would be against a different sample size.
    """
    out: list[Genome] = []
    for index, candidate in enumerate(candidates):
        attempt, replacement = 0, candidate
        while not archive.is_novel(replacement) and attempt < 8:
            archive.note_skip()
            attempt += 1
            replacement = coordinate_mutation(
                elite, generation, hash_index(1024, "evo_redraw", generation,
                                              index, attempt))
        out.append(replacement)
    return out
