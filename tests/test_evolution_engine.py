"""Unit tests for System 3: EvolutionEngine (mutate -> benchmark -> select)."""
from aegis.layers.evolution_engine import EvolutionEngine


def _ev(tmp_path):
    return EvolutionEngine(store_path=tmp_path / "ev.json")


def test_register_champion(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01, "temp": 0.7}, fitness=0.5)
    assert ev.champion["fitness"] == 0.5
    assert ev.status()["champion_fitness"] == 0.5


def test_propose_mutation_changes_one_param(tmp_path):
    """The single-variant contract, on the v2 genome.

    `register_champion` no longer takes arbitrary keys: the genome is the fixed
    schema of Appendix C, and the LoRA parameters that used to be genes were
    removed because nothing the benchmark measures reads them (§M5.3). Unknown
    keys are dropped rather than refused, so a caller handing over the live
    `self_mod.parameters` still gets a legal genome.
    """
    from aegis.layers.evolution.genome import GENE_NAMES

    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01, "temp": 0.7}, fitness=0.5)
    assert "lr" not in ev.champion["genome"]

    m = ev.propose_mutation(tick=100)
    assert m["param"] in GENE_NAMES
    assert m["new_value"] != m["old_value"]
    assert ev.candidate is not None


def test_only_one_candidate_at_a_time(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    assert ev.propose_mutation(tick=1) is not None
    assert ev.propose_mutation(tick=2) is None  # candidate still pending


def test_accept_when_fitness_improves(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    ev.propose_mutation(tick=1)
    verdict = ev.judge_candidate(0.7)  # clearly better
    assert verdict["decision"] == "accepted"
    assert verdict["revert_to"] is None
    assert ev.champion["fitness"] == 0.7
    assert ev.accepted == 1


def test_reject_when_fitness_drops(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    m = ev.propose_mutation(tick=1)
    verdict = ev.judge_candidate(0.4)  # worse
    assert verdict["decision"] == "rejected"
    assert verdict["revert_to"] == m["old_value"]
    assert ev.rejected == 1
    # champion fitness unchanged
    assert ev.champion["fitness"] == 0.5


def test_reject_on_marginal_gain_below_epsilon(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    ev.propose_mutation(tick=1)
    # gain smaller than FITNESS_EPSILON -> not a real improvement
    verdict = ev.judge_candidate(0.5001)
    assert verdict["decision"] == "rejected"


def test_generation_increments_on_judge(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    ev.propose_mutation(tick=1)
    ev.judge_candidate(0.6)
    assert ev.generation == 1


def test_abandon_candidate(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    m = ev.propose_mutation(tick=1)
    revert = ev.abandon_candidate()
    assert revert["revert_to"] == m["old_value"]
    assert ev.candidate is None
    # can propose again after abandon
    assert ev.propose_mutation(tick=2) is not None


def test_a_mutation_moves_the_genome_and_stays_in_range(tmp_path):
    """v2 replaces the round-robin ±10% step with a Halton coordinate move
    (§M5.4): every gene shifts by a fraction of *its own* range, so a gene
    spanning 100..5000 and one spanning 0..1 are perturbed comparably. The
    exact ±10% of v1 is therefore no longer the contract; staying in range and
    actually moving are.
    """
    from aegis.layers.evolution.genome import GENES_BY_NAME, Genome

    ev = _ev(tmp_path)
    ev.register_champion(Genome().to_dict(), fitness=0.5)
    parent = ev.champion_genome()

    m = ev.propose_mutation(tick=1)
    child = Genome(ev.candidate["genome"])
    assert child != parent
    for name, value in child.items():
        gene = GENES_BY_NAME[name]
        if gene.kind == "enum":
            assert value in gene.choices
        else:
            assert gene.low <= float(value) <= gene.high

    # The reported change is the gene that moved furthest in normalised terms —
    # what the caller's safety pipeline has to clamp and, if refused, revert.
    gene = GENES_BY_NAME[m["param"]]
    moved = abs(gene.position(m["new_value"]) - gene.position(m["old_value"]))
    assert moved == max(
        abs(GENES_BY_NAME[name].position(child[name])
            - GENES_BY_NAME[name].position(parent[name]))
        for name in child)


def test_successive_mutations_explore_different_points(tmp_path):
    """Two proposals in a row must not be the same proposal.

    v1 alternated direction to guarantee this; v2 walks the Halton sequence,
    which spreads more evenly than alternating ever did. What has to hold
    either way is that the search moves.
    """
    from aegis.layers.evolution.genome import Genome

    ev = _ev(tmp_path)
    ev.register_champion(Genome().to_dict(), fitness=0.5)
    ev.propose_mutation(tick=1)
    first = Genome(ev.candidate["genome"])
    ev.abandon_candidate()
    ev.propose_mutation(tick=2)
    second = Genome(ev.candidate["genome"])
    assert first != second
    assert first.distance(second) > 0


def test_lineage_is_bounded(tmp_path, monkeypatch):
    # Kills the `len(lineage) > MAX_LINEAGE` boundary mutant.
    import aegis.layers.evolution_engine as evmod
    monkeypatch.setattr(evmod, "MAX_LINEAGE", 3)
    ev = _ev(tmp_path)
    ev.register_champion({"a": 1.0}, fitness=0.5)
    for _ in range(10):
        ev.propose_mutation(tick=1)
        ev.judge_candidate(0.1)  # each judged cycle appends one lineage record
    assert len(ev.lineage) <= 3


def test_direction_default_is_up(tmp_path):
    # A fresh engine (no persisted direction) must start mutating upward.
    ev = _ev(tmp_path)
    assert ev._direction_up is True


def test_direction_defaults_to_up_when_absent_from_store(tmp_path):
    # Kills the `data.get("direction_up", True)` load-default mutant: a stored
    # state file that omits the key must load as True, not False.
    import json
    p = tmp_path / "ev.json"
    p.write_text(json.dumps({"generation": 2, "accepted": 1}), encoding="utf-8")
    ev = EvolutionEngine(store_path=p)  # loads a file without "direction_up"
    assert ev.generation == 2
    assert ev._direction_up is True


def test_judge_without_candidate(tmp_path):
    ev = _ev(tmp_path)
    ev.register_champion({"a": 1.0}, fitness=0.5)
    assert ev.judge_candidate(0.9)["decision"] == "no_candidate"


def test_persistence_round_trip(tmp_path):
    p = tmp_path / "ev.json"
    ev = EvolutionEngine(store_path=p)
    ev.register_champion({"lr": 0.01}, fitness=0.5)
    ev.propose_mutation(tick=1)
    ev.judge_candidate(0.8)  # accepted -> saves
    ev2 = EvolutionEngine(store_path=p)
    assert ev2.generation == 1
    assert ev2.accepted == 1
    assert ev2.champion["fitness"] == 0.8


def test_default_store_path_is_used(tmp_path, monkeypatch):
    # Exercises the `store_path or (EVOLUTION_DIR / "lineage.json")` default
    # branch (kills the Path-division mutant there).
    import aegis.layers.evolution_engine as evmod
    monkeypatch.setattr(evmod, "EVOLUTION_DIR", tmp_path)
    ev = EvolutionEngine()  # no store_path -> default branch
    assert ev._store_path == tmp_path / "lineage.json"
