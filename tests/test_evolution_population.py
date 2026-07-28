"""Population evolution (spec M5): genome, operators, selection, rollback.

The failure this stage exists to correct is worth restating, because it is the
one a test suite is least likely to notice on its own: the old genome held LoRA
hyper-parameters that **nothing the benchmark measures reads**. Evolution
mutated them faithfully for thousands of ticks over a search space with no
gradient in it, and every test passed the whole time.

So the first thing tested here is not that mutation works. It is that a gene is
a parameter some contour actually consults, and that changing it changes what
the system does.
"""
import pytest

import aegis.config as cfg
from aegis.layers.evolution import (
    GENES, GENES_BY_NAME, GENE_NAMES, Genome, NoveltyArchive, Population,
    RETIRED_GENES, big_step, compose, coordinate_mutation, cost_of, crossover,
    diversify, from_proposal, observable_genes,
)
from aegis.layers.evolution.genome import DELIVERED_STAGE, simplex_floor
from aegis.layers.evolution.harness import FitnessReport
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.safety import immutable


def _engine(tmp_path, **kw):
    return EvolutionEngine(store_path=tmp_path / "lineage.json", **kw)


def _report(fitness, genome_id="g", valid=None):
    return FitnessReport(genome_id=genome_id, fitness=fitness,
                         score_valid=valid if valid is not None else fitness)


# ── the genome is real ───────────────────────────────────────────────

def test_every_gene_names_the_contour_that_reads_it():
    """A gene with no reader is what the old genome was made of."""
    for gene in GENES:
        assert gene.reader, f"{gene.name} names no reader"


def test_the_lora_parameters_are_no_longer_genes():
    assert not (RETIRED_GENES & set(GENE_NAMES))


def test_no_gene_names_a_protected_parameter():
    """Appendix B: evolution may not reach the ethics thresholds, the kill
    switch, the sandbox gate or the control plane."""
    for name in GENE_NAMES:
        assert not immutable.is_immutable(name), name


def test_a_schema_naming_a_protected_parameter_fails_at_import():
    import aegis.layers.evolution.genome as module

    original = module.GENE_NAMES
    module.GENE_NAMES = original + ("API_TOKEN",)
    try:
        with pytest.raises(immutable.ImmutableParameterError):
            module.assert_no_immutable_genes()
    finally:
        module.GENE_NAMES = original


def test_a_genome_is_always_complete():
    assert set(Genome()) == set(GENE_NAMES)
    assert set(Genome({"w_ev": 1.2})) == set(GENE_NAMES)


def test_values_are_clamped_to_their_range():
    genome = Genome({"plan_beam": 999, "plan_depth": -4, "w_ev": 17.0})
    assert genome["plan_beam"] == 16
    assert genome["plan_depth"] == 1
    assert genome["w_ev"] == 2.0


def test_an_unusable_value_falls_back_to_the_default():
    genome = Genome({"w_ev": "lots", "plan_beam": None,
                     "solver_order": "sideways"})
    assert genome["w_ev"] == GENES_BY_NAME["w_ev"].default
    assert genome["plan_beam"] == GENES_BY_NAME["plan_beam"].default
    assert genome["solver_order"] == "by_success"


def test_a_not_a_number_is_refused():
    assert Genome({"w_ev": float("nan")})["w_ev"] == GENES_BY_NAME["w_ev"].default


def test_an_unknown_key_is_dropped_not_kept():
    genome = Genome({"learning_rate": 0.01, "nonsense": 5})
    assert "learning_rate" not in genome and "nonsense" not in genome


def test_integer_genes_stay_integers():
    genome = Genome({"plan_beam": 7.6})
    assert genome["plan_beam"] == 8 and isinstance(genome["plan_beam"], int)


# ── the simplex ──────────────────────────────────────────────────────

def test_the_resource_shares_sum_to_one():
    genome = Genome({"res_share_competence": 0.7, "res_share_knowledge": 0.7,
                     "res_share_coherence": 0.7, "res_share_stability": 0.7})
    total = sum(value for name, value in genome.items()
                if name.startswith("res_share_"))
    assert total == pytest.approx(1.0)


def test_no_share_falls_below_the_floor():
    genome = Genome({"res_share_competence": 0.7, "res_share_knowledge": 0.05,
                     "res_share_coherence": 0.05, "res_share_stability": 0.05})
    floor = simplex_floor(4)
    for name, value in genome.items():
        if name.startswith("res_share_"):
            assert value >= floor - 1e-9


def test_normalising_twice_changes_nothing():
    """Idempotence is what makes a revert exact: applying a genome and reading
    it back has to give the same numbers, or every rejected variant leaves a
    little of itself behind."""
    genome = Genome({"res_share_competence": 0.6, "res_share_knowledge": 0.2,
                     "res_share_coherence": 0.1, "res_share_stability": 0.1})
    before = dict(genome)
    genome.normalise_simplexes()
    assert genome == before


def test_shares_that_are_all_zero_become_an_equal_split():
    genome = Genome()
    for name in genome:
        if name.startswith("res_share_"):
            dict.__setitem__(genome, name, 0.0)
    genome.normalise_simplexes()
    shares = [v for k, v in genome.items() if k.startswith("res_share_")]
    assert all(value == pytest.approx(0.25) for value in shares)


def test_the_genome_and_the_resource_manager_agree_on_a_legal_split(tmp_path):
    """One definition, used in both places. Two different normalisations meant
    a share written as 0.05 came back as 0.09."""
    from aegis.layers.motivation.roi import ROITracker

    roi = ROITracker(store_path=tmp_path / "roi.json")
    genome = Genome({"res_share_competence": 0.05, "res_share_knowledge": 0.6,
                     "res_share_coherence": 0.2, "res_share_stability": 0.15})
    roi.set_genome(genome.to_dict())
    for drive, share in roi.shares.items():
        assert share == pytest.approx(genome[f"res_share_{drive}"])


# ── distance and identity ────────────────────────────────────────────

def test_distance_is_normalised_by_range():
    """Raw distances would let a gene spanning 100..5000 drown out every gene
    spanning 0..1, and the novelty archive would only ever notice one of them.

    Both genes are swung from one end of their own range to the other, so a
    normalised distance has to report the same number for each.
    """
    wide = Genome({"wm_half_life": 100}).distance(Genome({"wm_half_life": 5000}))
    narrow = Genome({"w_ev": 0.0}).distance(Genome({"w_ev": 2.0}))
    assert wide == pytest.approx(narrow, abs=1e-9)
    assert wide > 0


def test_a_genome_is_at_no_distance_from_itself():
    genome = Genome({"w_ev": 1.7})
    assert genome.distance(Genome(genome)) == 0.0


def test_the_digest_identifies_the_values_not_the_object():
    assert Genome({"w_ev": 1.5}).digest() == Genome({"w_ev": 1.5}).digest()
    assert Genome({"w_ev": 1.5}).digest() != Genome({"w_ev": 1.6}).digest()


# ── persistence and migration ────────────────────────────────────────

def test_a_genome_round_trips_through_storage():
    genome = Genome({"w_ev": 1.7, "solver_order": "by_length", "plan_beam": 9})
    assert Genome.from_stored(genome.stored()) == genome


def test_a_v1_file_migrates_to_defaults():
    """A v1 genome held LoRA hyper-parameters, none of which are genes now.
    There is no correspondence to preserve, and inventing one would put a
    made-up genome at the top of the lineage."""
    migrated = Genome.from_stored({"learning_rate": 0.01, "dropout": 0.2})
    assert migrated == Genome()


@pytest.mark.parametrize("stored", [None, {}, "not a mapping", []])
def test_unreadable_storage_gives_a_default_genome(stored):
    assert Genome.from_stored(stored) == Genome()


# ── operators are deterministic ──────────────────────────────────────

def test_a_mutation_is_a_pure_function_of_its_inputs():
    first = coordinate_mutation(Genome(), generation=3, index=1)
    second = coordinate_mutation(Genome(), generation=3, index=1)
    assert first == second


def test_different_slots_of_a_generation_differ():
    first = coordinate_mutation(Genome(), generation=3, index=1)
    second = coordinate_mutation(Genome(), generation=3, index=2)
    assert first != second


def test_successive_generations_do_not_redraw_the_same_points():
    """A population that mutated by the same offsets every generation would
    explore one direction forever."""
    first = coordinate_mutation(Genome(), generation=1, index=0)
    second = coordinate_mutation(Genome(), generation=2, index=0)
    assert first != second


def test_a_mutation_stays_in_range():
    """Every gene lands inside its declared range — except the share genes.

    A share's declared range is the range a *proposal* is clamped to; what
    comes out is an *allocation*, and normalising four shares to sum to one can
    legitimately push the largest above its own ceiling (with a floor of 0.05,
    the most one share can hold is 1 − 3·0.05 = 0.85). The invariant that
    matters for those is the simplex, and it is asserted instead.
    """
    floor = simplex_floor(4)
    for index in range(20):
        child = coordinate_mutation(Genome(), generation=index, index=index,
                                    sigma=0.9)
        shares = []
        for name, value in child.items():
            gene = GENES_BY_NAME[name]
            if gene.kind == "enum":
                assert value in gene.choices
            elif gene.simplex:
                shares.append(float(value))
                assert float(value) >= floor - 1e-9
            else:
                assert gene.low - 1e-9 <= float(value) <= gene.high + 1e-9
        assert sum(shares) == pytest.approx(1.0)


def test_a_big_step_moves_further_than_an_ordinary_one():
    """The only thing that leaves a converged neighbourhood."""
    parent = Genome()
    ordinary = coordinate_mutation(parent, 1, 0, sigma=0.05)
    jump = big_step(parent, 1, 0, sigma=0.05)
    assert parent.distance(jump) > parent.distance(ordinary)


def test_a_crossover_takes_every_gene_from_one_parent_or_the_other():
    first = Genome({name: GENES_BY_NAME[name].high
                    for name in GENE_NAMES
                    if GENES_BY_NAME[name].kind != "enum"})
    second = Genome()
    child = crossover(first, second, generation=2, index=0)
    for name in GENE_NAMES:
        if GENES_BY_NAME[name].simplex:
            continue                    # renormalised, so not a bare copy
        assert child[name] in (first[name], second[name]), name


def test_two_crossovers_of_the_same_pair_differ():
    # Parents that differ in every gene, or two draws could legitimately pick
    # the same handful of identical values and the test would prove nothing.
    first = Genome({name: GENES_BY_NAME[name].low
                    for name in GENE_NAMES
                    if GENES_BY_NAME[name].kind != "enum"})
    second = Genome({name: GENES_BY_NAME[name].high
                     for name in GENE_NAMES
                     if GENES_BY_NAME[name].kind != "enum"})
    assert crossover(first, second, 1, 0) != crossover(first, second, 1, 1)


# ── the cortex slot ──────────────────────────────────────────────────

def test_a_usable_proposal_is_accepted_and_clamped():
    child, accepted = from_proposal({"w_ev": 99.0, "plan_beam": 12}, Genome())
    assert accepted
    assert child["w_ev"] == 2.0 and child["plan_beam"] == 12


@pytest.mark.parametrize("proposal", [None, {}, "a genome", [1, 2],
                                      {"nonsense": 1}])
def test_an_unusable_proposal_falls_back_visibly(proposal):
    """The fallback has to be visible: silently substituting a mutation would
    make the cortex look useful in the lineage whether or not it was."""
    fallback = Genome({"w_ev": 1.9})
    child, accepted = from_proposal(proposal, fallback)
    assert not accepted
    assert child == fallback


# ── the novelty archive ──────────────────────────────────────────────

def test_a_seen_genome_is_not_novel():
    archive = NoveltyArchive(min_distance=0.01)
    genome = Genome({"w_ev": 1.5})
    assert archive.is_novel(genome)
    archive.add(genome, 0.5)
    assert not archive.is_novel(Genome({"w_ev": 1.5}))


def test_a_distant_genome_is_novel():
    archive = NoveltyArchive(min_distance=0.001)
    archive.add(Genome({"w_ev": 0.0}), 0.5)
    assert archive.is_novel(Genome({"w_ev": 2.0}))


def test_the_archive_is_bounded():
    archive = NoveltyArchive(min_distance=0.0, capacity=5)
    for index in range(20):
        archive.add(Genome({"w_ev": index / 20}), index)
    assert len(archive.entries) == 5


def test_diversify_keeps_the_generation_size():
    """A generation that shrank because half of it was too similar would make
    the population size depend on how converged the search happened to be."""
    archive = NoveltyArchive(min_distance=0.5)      # almost nothing is novel
    archive.add(Genome(), 0.5)
    candidates = [Genome() for _ in range(4)]
    out = diversify(candidates, archive, generation=1, elite=Genome())
    assert len(out) == 4
    assert archive.skips > 0


def test_the_archive_survives_a_round_trip():
    archive = NoveltyArchive(min_distance=0.02)
    archive.add(Genome({"w_ev": 1.1}), 0.4)
    archive.note_skip()
    restored = NoveltyArchive.from_dict(archive.to_dict())
    assert restored.skips == 1
    assert len(restored.entries) == 1
    assert not restored.is_novel(Genome({"w_ev": 1.1}))


def test_a_torn_archive_row_is_skipped():
    restored = NoveltyArchive.from_dict(
        {"entries": ["nonsense", {"no": "genes"},
                     {"digest": "d", "genes": {"w_ev": 1.0}}],
         "skips": "many"})
    assert len(restored.entries) == 1 and restored.skips == 0


# ── the generation ───────────────────────────────────────────────────

def test_a_generation_has_the_composition_the_spec_asks_for():
    genomes, origins = compose([Genome(), Genome({"w_ev": 1.2})],
                               generation=1, size=10)
    assert len(genomes) == 10
    counts = {origin: origins.count(origin) for origin in set(origins)}
    assert counts["elite"] == 2
    assert counts["crossover"] == 2
    assert counts["big_step"] == 1
    assert counts.get("cortex", 0) + counts.get("cortex_rejected", 0) == 1
    assert counts["mutation"] == 4


def test_a_population_of_one_is_still_legal():
    """`EVO_POP_SIZE = 1` has to keep working — it is the v1 behaviour."""
    genomes, origins = compose([Genome()], generation=1, size=1)
    assert len(genomes) == 1 and origins == ["elite"]


def test_a_generation_with_no_elites_starts_from_the_defaults():
    genomes, _ = compose([], generation=1, size=4)
    assert len(genomes) == 4


def test_the_elites_carry_forward_unchanged():
    population = Population(NoveltyArchive(min_distance=0.0), size=6)
    elite = Genome({"w_ev": 1.9})
    genomes = population.build([elite, Genome()], generation=1)
    assert genomes[0] == elite


def test_selection_is_total():
    """Two variants with identical fitness are common early on, and whichever
    the sort happened to put first must not decide the champion."""
    population = Population(NoveltyArchive(min_distance=0.0), size=3)
    population.genomes = [Genome({"w_ev": v}) for v in (0.1, 0.2, 0.3)]
    reports = [_report(0.5, "b"), _report(0.5, "a"), _report(0.5, "c")]
    assert [index for index, _ in population.rank(reports)] == [1, 0, 2]


def test_the_best_variant_is_the_fittest():
    population = Population(NoveltyArchive(min_distance=0.0), size=3)
    population.genomes = [Genome({"w_ev": v}) for v in (0.1, 0.2, 0.3)]
    genome, report = population.best([_report(0.1), _report(0.9), _report(0.4)])
    assert report.fitness == 0.9
    assert genome["w_ev"] == 0.2


def test_an_empty_generation_has_no_best():
    population = Population(NoveltyArchive(), size=3)
    assert population.best([]) == (None, None)


def test_a_challenger_must_clear_the_margin():
    """Without a margin, benchmark noise promotes a sideways change every
    generation and calls it progress."""
    population = Population(NoveltyArchive(), epsilon=0.05)
    assert population.beats(_report(0.60), 0.50)
    assert not population.beats(_report(0.52), 0.50)
    assert not population.beats(None, 0.50)
    assert population.beats(_report(0.01), None)     # nothing to beat yet


def test_every_variant_reaches_the_archive_including_the_losers():
    """The losers are data (§M5.6): they are what stops the next generation
    proposing the same failures again."""
    archive = NoveltyArchive(min_distance=0.0)
    population = Population(archive, size=4)
    population.genomes = [Genome({"w_ev": v}) for v in (0.1, 0.2, 0.3, 0.4)]
    population.record([_report(v) for v in (0.1, 0.2, 0.3, 0.4)])
    assert len(archive.entries) == 4


# ── fitness ──────────────────────────────────────────────────────────

def test_an_expensive_configuration_costs_more():
    """Unpriced, evolution reliably discovers that a longer timeout and a wider
    beam score better, and converges on something correct and unaffordable."""
    cheap = Genome({"solver_timeout": 0.5, "plan_beam": 1, "plan_depth": 1})
    dear = Genome({"solver_timeout": 10.0, "plan_beam": 16, "plan_depth": 5})
    assert cost_of(cheap) == pytest.approx(0.0)
    assert cost_of(dear) == pytest.approx(1.0)
    assert cost_of(Genome()) > 0.0


def test_a_fitness_report_round_trips():
    report = FitnessReport(genome_id="g", score_valid=0.8, fitness=0.75,
                           subscores={"calc": 1.0}, failures=["late"])
    restored = FitnessReport.from_dict(report.as_dict())
    assert restored.genome_id == "g" and restored.fitness == pytest.approx(0.75)
    assert restored.failures == ["late"] and not restored.ok


def test_a_report_with_no_failures_is_ok():
    assert FitnessReport(genome_id="g").ok


# ── the engine ───────────────────────────────────────────────────────

def test_a_generation_promotes_the_best_variant(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.0)

    scores = iter([0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05])

    def _fake(genomes, splits=("valid",), start=0):
        return [_report(next(scores), f"v{index}")
                for index in range(len(genomes))]

    monkeypatch.setattr(engine.evaluator, "evaluate", _fake)
    monkeypatch.setattr(engine.evaluator, "confirm",
                        lambda genome, start=0: _report(0.88, "test"))

    summary = engine.run_generation(tick=10)
    assert summary["promoted"] is True
    assert engine.champion["fitness"] == pytest.approx(0.9)
    assert engine.promotions == 1
    assert summary["valid_test_gap"] == pytest.approx(0.02, abs=1e-6)


def test_a_generation_that_beats_nothing_promotes_nothing(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.95)
    monkeypatch.setattr(engine.evaluator, "evaluate",
                        lambda genomes, splits=("valid",), start=0:
                        [_report(0.1, f"v{i}") for i in range(len(genomes))])
    summary = engine.run_generation(tick=10)
    assert summary["promoted"] is False
    assert engine.champion["fitness"] == pytest.approx(0.95)
    assert engine.rejected == 1


def test_the_split_window_rotates(tmp_path):
    """Selecting on the same tasks forever is how a population learns the
    tasks rather than the job."""
    engine = _engine(tmp_path)
    engine.generation = 0
    first = engine._split_start()
    engine.generation = cfg.EVO_SPLIT_ROTATE_EVERY
    assert engine._split_start() != first


def test_a_generation_is_flagged_while_it_runs(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.0)
    seen = {}

    def _fake(genomes, splits=("valid",), start=0):
        seen["running"] = engine.generation_running
        return [_report(0.1, f"v{i}") for i in range(len(genomes))]

    monkeypatch.setattr(engine.evaluator, "evaluate", _fake)
    engine.run_generation(tick=1)
    assert seen["running"] is True
    assert engine.generation_running is False


# ── rollback ─────────────────────────────────────────────────────────

def test_a_champion_that_hurts_the_live_metric_is_rolled_back(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    engine._promote({"genome": Genome({"w_ev": 2.0}).to_dict(), "fitness": 0.9,
                     "generation": 1, "created": 0.0}, tick=100)

    assert engine.watch(tick=101, live_metric=0.80) is None    # the baseline
    record = engine.watch(tick=150, live_metric=0.80 - cfg.EVO_ROLLBACK_DELTA - 0.01)
    assert record is not None and record["decision"] == "rolled_back"
    assert engine.rollbacks == 1
    assert engine.champion["fitness"] == pytest.approx(0.5)


def test_a_champion_that_holds_up_is_kept(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    engine._promote({"genome": Genome().to_dict(), "fitness": 0.9,
                     "generation": 1, "created": 0.0}, tick=100)
    engine.watch(tick=101, live_metric=0.80)
    assert engine.watch(tick=150, live_metric=0.82) is None
    assert engine.champion["fitness"] == pytest.approx(0.9)
    assert engine.rollbacks == 0


def test_the_watch_window_closes(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    engine._promote({"genome": Genome().to_dict(), "fitness": 0.9,
                     "generation": 1, "created": 0.0}, tick=100)
    engine.watch(tick=101, live_metric=0.80)
    assert engine.watch(tick=100 + cfg.EVO_WATCH_TICKS + 1, live_metric=0.1) is None
    assert engine.rollbacks == 0
    assert engine.previous_champion is None      # nothing left to go back to


@pytest.mark.parametrize("metric", [None, "not a number"])
def test_an_unreadable_live_metric_decides_nothing(tmp_path, metric):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    engine._promote({"genome": Genome().to_dict(), "fitness": 0.9,
                     "generation": 1, "created": 0.0}, tick=100)
    assert engine.watch(tick=150, live_metric=metric) is None


def test_nothing_to_roll_back_to_is_not_a_rollback(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    assert engine.watch(tick=150, live_metric=0.0) is None


# ── persistence ──────────────────────────────────────────────────────

def test_the_engine_survives_a_restart(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome({"w_ev": 1.7}).to_dict(), fitness=0.6)
    engine.archive.add(Genome({"w_ev": 1.7}), 0.6)
    engine.promotions = 3
    engine.save()

    restored = _engine(tmp_path)
    assert restored.champion_genome()["w_ev"] == pytest.approx(1.7)
    assert restored.promotions == 3
    assert not restored.archive.is_novel(Genome({"w_ev": 1.7}))


def test_a_v1_champion_migrates_without_inventing_a_genome(tmp_path):
    import json

    (tmp_path / "lineage.json").write_text(json.dumps({
        "champion": {"genome": {"learning_rate": 0.01, "dropout": 0.2},
                     "fitness": 0.42, "generation": 7},
        "candidate": {"mutated_param": "learning_rate", "old_value": 0.01},
        "generation": 7, "accepted": 2,
    }), encoding="utf-8")

    engine = _engine(tmp_path)
    assert engine.champion["fitness"] == pytest.approx(0.42)
    assert engine.champion_genome() == Genome()
    assert engine.generation == 7 and engine.accepted == 2
    # A v1 candidate cannot be finished under v2 rules, so it is dropped rather
    # than half-played (Appendix I).
    assert engine.candidate is None


def test_an_unreadable_store_starts_fresh(tmp_path):
    (tmp_path / "lineage.json").write_text("{not json", encoding="utf-8")
    engine = _engine(tmp_path)
    assert engine.champion is None and engine.generation == 0


def test_the_lineage_exports_as_csv(tmp_path):
    engine = _engine(tmp_path)
    engine.lineage = [{"generation": 1, "decision": "accepted",
                       "best_fitness": 0.7, "promoted": True}]
    csv = engine.lineage_csv()
    assert csv.splitlines()[0].startswith("generation,decision")
    assert "accepted" in csv


# ── metrics ──────────────────────────────────────────────────────────

def test_every_required_evolution_metric_is_published(tmp_path):
    from aegis.telemetry import metrics as M

    recorded = []
    engine = _engine(tmp_path, telemetry=type("T", (), {
        "record": staticmethod(lambda name, value, tick, tags=None:
                               recorded.append(name))})())
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    engine.publish_metrics(1)
    assert set(recorded) == {M.EVO_GENERATION, M.EVO_PROMOTIONS,
                             M.EVO_ROLLBACKS, M.EVO_NOVELTY_SKIPS,
                             M.EVO_CHAMPION_FITNESS, M.EVO_VALID_TEST_GAP}


def test_a_failing_telemetry_sink_does_not_stop_a_generation(tmp_path):
    def _explode(*a, **k):
        raise RuntimeError("sink down")

    engine = _engine(tmp_path, telemetry=type("T", (), {
        "record": staticmethod(_explode)})())
    engine.publish_metrics(1)          # swallowed, logged


def test_status_describes_the_population(tmp_path):
    engine = _engine(tmp_path)
    engine.register_champion(Genome().to_dict(), fitness=0.5)
    status = engine.status()
    assert status["champion_fitness"] == 0.5
    assert set(status["champion_genome"]) == set(GENE_NAMES)
    assert "population" in status and "evaluator" in status


# ── the sensitivity claim (Appendix C) ───────────────────────────────

def test_only_delivered_genes_are_held_to_the_sensitivity_gate():
    """A gene declared for a later stage is honestly inert. Demanding that it
    move the fitness would mean either deleting it or faking a reader."""
    observable = set(observable_genes())
    assert observable
    for name in GENE_NAMES:
        gene = GENES_BY_NAME[name]
        assert (name in observable) == (gene.stage <= DELIVERED_STAGE)


def test_every_delivered_gene_reaches_a_live_contour(isolated_state):
    """The claim each `reader` makes, checked against the running system.

    This is the test that would have caught the old genome: set a gene to each
    end of its range, apply it, and require the system's own configuration to
    come back different. A gene that cannot move anything is a gene evolution
    would search over for nothing.
    """
    from aegis.layers.substrate import Substrate

    substrate = Substrate()
    for name in observable_genes():
        gene = GENES_BY_NAME[name]
        low = Genome({name: gene.low if gene.kind != "enum" else gene.choices[0]})
        high = Genome({name: gene.high if gene.kind != "enum" else gene.choices[-1]})
        if low[name] == high[name]:
            continue                      # a one-value gene cannot be moved

        substrate.apply_genome(low)
        after_low = substrate.current_genome()[name]
        substrate.apply_genome(high)
        after_high = substrate.current_genome()[name]
        assert after_low != after_high, (
            f"{name} does not reach any live contour — its declared reader "
            f"{gene.reader!r} does not read it")


# ── the switch that keeps a generation deliberate ────────────────────

def test_a_generation_is_never_started_by_accident(isolated_state, monkeypatch):
    """`EVO_ENABLED` gates the action, not the engine.

    A generation evaluates ten variants, each building a fresh system in
    another process. Any long-running test that ticked past the interval would
    otherwise start one: measured at 137 python processes and a suite that
    stopped finishing. The engine itself is always callable — this only decides
    whether a *tick* may start one.
    """
    from aegis.layers.actions import evaluate_precondition
    from aegis.layers.substrate import Substrate

    substrate = Substrate()
    monkeypatch.setattr(cfg, "EVO_ENABLED", False)
    assert not evaluate_precondition("evolution_allowed", substrate)

    monkeypatch.setattr(cfg, "EVO_ENABLED", True)
    assert evaluate_precondition("evolution_allowed", substrate)


def test_evolution_is_off_in_the_suite():
    """The autouse fixture in conftest, asserted rather than assumed."""
    assert cfg.EVO_ENABLED is False


def test_regulation_can_still_withhold_evolution(isolated_state, monkeypatch):
    from aegis.layers.actions import evaluate_precondition
    from aegis.layers.substrate import Substrate

    substrate = Substrate()
    monkeypatch.setattr(cfg, "EVO_ENABLED", True)
    substrate._regulation_directives = {"skip_learning": True}
    assert not evaluate_precondition("evolution_allowed", substrate)
