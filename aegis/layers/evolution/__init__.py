"""Population evolution (spec M5).

    создать 10 вариантов улучшения → проверить → оставить лучший

The previous engine mutated one parameter, waited for one benchmark, and kept
the change if the number went up. Two things were wrong with it, and only the
second is obvious.

The obvious one: one variant per benchmark is not evolution, it is hill
climbing with a very long step time.

The one that mattered more: **the genome could not affect the measurement.**
It held LoRA hyper-parameters — learning rate, dropout, attention heads — and
nothing the benchmark exercises reads any of them. The engine ran faithfully for
thousands of ticks over a search space with no gradient in it. So the genome was
replaced wholesale (Appendix C), and every gene now names the contour that
consults it.
"""
from aegis.layers.evolution.genome import (
    GENE_NAMES, GENES, GENES_BY_NAME, GENOME_SCHEMA_VERSION, Genome, GeneSpec,
    RETIRED_GENES, defaults, observable_genes,
)
from aegis.layers.evolution.harness import (
    FitnessReport, VariantEvaluator, cost_of, evaluate_variant,
    make_variant_request,
)
from aegis.layers.evolution.operators import (
    NoveltyArchive, big_step, coordinate_mutation, crossover, diversify,
    from_proposal,
)
from aegis.layers.evolution.population import Population, compose

__all__ = [
    "GENE_NAMES", "GENES", "GENES_BY_NAME", "GENOME_SCHEMA_VERSION",
    "Genome", "GeneSpec", "RETIRED_GENES", "defaults", "observable_genes",
    "FitnessReport", "VariantEvaluator", "cost_of", "evaluate_variant",
    "make_variant_request",
    "NoveltyArchive", "big_step", "coordinate_mutation", "crossover",
    "diversify", "from_proposal",
    "Population", "compose",
]
