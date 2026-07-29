"""pytest-bdd step definitions for tests/features/evolution.feature.

Executable Gherkin over the real genome, population and operators (M5). The
scenarios are about the three properties that make a generation evidence rather
than churn: it is composed by rule, selection is a total order, and the genome
cannot reach anything the safety contract forbids.
"""
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aegis.layers.evolution.genome import GENES, Genome
from aegis.layers.evolution.population import SLOT_ELITES, Population
from aegis.safety.immutable import IMMUTABLE_PARAMS, normalize

scenarios("features/evolution.feature")


class _Report:
    """The shape :meth:`Population.rank` reads."""

    def __init__(self, genome_id: str, fitness: float):
        self.genome_id = genome_id
        self.fitness = fitness


@given("a population of ten", target_fixture="ctx")
def _population():
    return {"population": Population(size=10),
            "elites": [Genome({"plan_beam": 4}), Genome({"plan_beam": 7})]}


@given("the genome schema", target_fixture="ctx")
def _schema():
    return {"genes": GENES}


# ── composition ──────────────────────────────────────────────────────

@when("a generation is composed from two elites")
def _compose(ctx):
    ctx["genomes"] = ctx["population"].build(ctx["elites"], generation=3)


@when("an identical population composes the same generation")
def _compose_again(ctx):
    ctx["again"] = Population(size=10).build(
        [Genome({"plan_beam": 4}), Genome({"plan_beam": 7})], generation=3)


@then(parsers.parse("it should hold {count:d} genomes"))
def _size(ctx, count):
    assert len(ctx["genomes"]) == count


@then("the first two genomes should be the elites themselves")
def _elites_first(ctx):
    """Elites carry forward unchanged and are exempt from the novelty filter:
    they are *meant* to be what was already evaluated, and redrawing them is
    the one thing that would let a generation regress."""
    for index in range(SLOT_ELITES):
        assert dict(ctx["genomes"][index]) == dict(ctx["elites"][index])


@then("the two generations should be identical")
def _deterministic(ctx):
    """§3.1. A generation that differed between runs would make every fitness
    comparison across runs meaningless."""
    assert [dict(genome) for genome in ctx["genomes"]] == \
        [dict(genome) for genome in ctx["again"]]


# ── selection ────────────────────────────────────────────────────────

@when(parsers.parse("four variants are judged with fitnesses {a:f}, {b:f}, "
                    "{c:f} and {d:f}"))
def _judge(ctx, a, b, c, d):
    ctx["reports"] = [_Report(f"g{index}", fitness)
                      for index, fitness in enumerate((a, b, c, d))]
    ctx["ranked"] = ctx["population"].rank(ctx["reports"])


@when("two variants are judged with the same fitness")
def _tie(ctx):
    ctx["forward"] = Population(size=10).rank(
        [_Report("alpha", 0.5), _Report("beta", 0.5)])
    ctx["backward"] = Population(size=10).rank(
        [_Report("beta", 0.5), _Report("alpha", 0.5)])


@then(parsers.parse("the best should be the one scoring {best:f}"))
def _best(ctx, best):
    assert ctx["ranked"][0][1].fitness == pytest.approx(best)


@then("the ranking should be the same whichever order they arrived in")
def _stable_tie(ctx):
    """Two variants with identical fitness are common early on, and
    "whichever the sort happened to put first" would make the champion depend
    on list order."""
    assert [report.genome_id for _, report in ctx["forward"]] == \
        [report.genome_id for _, report in ctx["backward"]]


# ── the safety contract ──────────────────────────────────────────────

@then("no gene should name a parameter from the immutable set")
def _no_immutable_gene(ctx):
    """The negative test the spec asks for by name (M5.9). Evolution reaching
    an ethical threshold would not be a bug, it would be the end of the
    guarantee the whole safety contract rests on."""
    immutable = {normalize(name) for name in IMMUTABLE_PARAMS}
    for gene in ctx["genes"]:
        assert normalize(gene.name) not in immutable, gene.name


@when("a genome is pushed past every bound")
def _push(ctx):
    ctx["pushed"] = Genome({gene.name: (gene.high * 100 if gene.kind != "enum"
                                        else "nonsense")
                            for gene in ctx["genes"]})


@then("every gene should be clamped back into its range")
def _clamped(ctx):
    for gene in ctx["genes"]:
        value = ctx["pushed"][gene.name]
        if gene.kind == "enum":
            assert value in gene.choices
        else:
            assert gene.low <= value <= gene.high, gene.name


@when("a genome carrying an undeclared gene is loaded")
def _undeclared(ctx):
    ctx["loaded"] = Genome({"plan_beam": 5, "a_gene_nobody_declared": 99})


@then("the undeclared gene should not survive")
def _dropped(ctx):
    """Dropped rather than refused: values arrive from disk, from a migration
    and from a model, and a genome that refused to load because one key was
    strange would take the whole lineage with it."""
    assert "a_gene_nobody_declared" not in ctx["loaded"]
    assert ctx["loaded"]["plan_beam"] == 5
