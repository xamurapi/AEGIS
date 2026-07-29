"""The system comes up on real v1 files (spec Appendix I, §VII.9).

``tests/test_migrations.py`` covers the migration *machinery* — version
detection, the step registry, the refusal to go backwards. This file covers the
thing the machinery exists for: a directory of files written by the previous
build, loaded by this one, with the system booting on them and losing nothing
it was supposed to keep.

The fixture in ``tests/fixtures/legacy_state/`` is a v1 snapshot: no
``schema_version`` anywhere, the pre-M5.3 LoRA genome where the genome of
Appendix C now goes, and an experience log that ends mid-line — because a
snapshot taken from a killed process does.

Two things must survive and one must not:

* the **champion** and the **benchmark history** are history, and history is
  not something a version bump is allowed to discard;
* the **v1 genome** must not survive. Its genes do not influence the measured
  benchmark at all — that is precisely why the genome was replaced — so
  carrying their values into the new one would carry the old problem with them.
"""
import asyncio
import json
import shutil
from pathlib import Path

import pytest

from aegis.store.migrations import read_store, version_of

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_state"

#: The genes of the genome that was replaced. None of them may appear in a
#: migrated champion.
RETIRED_GENES = ("learning_rate", "dropout", "attention_heads", "temperature",
                 "curiosity_weight", "memory_decay")


@pytest.fixture
def legacy(tmp_path):
    """A private copy of the snapshot, so a test can migrate it in place."""
    root = tmp_path / "state"
    shutil.copytree(FIXTURE, root)
    return root


# ── the snapshot is what it claims to be ─────────────────────────────

def test_the_snapshot_exists_and_is_version_one():
    """A fixture that had quietly been stamped v2 would make every migration
    test below pass by migrating nothing."""
    assert FIXTURE.is_dir()
    stores = sorted(FIXTURE.rglob("*.json"))
    assert len(stores) >= 6, [p.name for p in stores]
    for path in stores:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" not in payload, path.name
        assert version_of(payload) == 1, path.name


def test_the_snapshot_carries_the_genome_that_was_replaced():
    champion = json.loads(
        (FIXTURE / "evolution" / "lineage.json").read_text(encoding="utf-8")
    )["champion"]
    assert set(RETIRED_GENES) <= set(champion["genome"])


# ── what the migration keeps ─────────────────────────────────────────

def test_the_champion_survives_with_its_fitness_and_lineage(legacy):
    """The champion is the result of eighty-eight generations of evidence.
    A version bump that dropped it would throw that away silently."""
    migrated = read_store(legacy / "evolution" / "lineage.json", store="evolution")

    assert migrated["champion"]["fitness"] == pytest.approx(0.4815)
    assert migrated["champion"]["generation"] == 87
    assert migrated["generation"] == 88
    assert migrated["accepted"] == 5 and migrated["rejected"] == 83
    assert len(migrated["lineage"]) == 88


def test_the_benchmark_history_survives_intact(legacy):
    """Two hundred points of measured capability. The discovery engine fits
    models over exactly this kind of window."""
    migrated = read_store(legacy / "eval" / "eval_history.json",
                          store="eval_history")

    assert migrated["total_runs"] == 418
    assert migrated["last_score"] == pytest.approx(0.5312)
    assert len(migrated["history"]) == 200
    assert migrated["history"][0]["score"] == pytest.approx(0.3105)


def test_the_causal_links_survive_verbatim(legacy):
    """The predictive tables cannot be reconstructed from cause→effect strings,
    so they start empty — but the strings themselves are evidence and stay."""
    migrated = read_store(legacy / "world_model" / "model.json",
                          store="world_model")

    assert migrated["total_observations"] == 150
    assert migrated["links"]["decision:run_benchmark"]["score_up"]["successes"] == 61
    assert len(migrated["chains"]) == 1


@pytest.mark.parametrize("store,path,key,expected", [
    ("goal_intelligence", "goal_intelligence/values.json", "total_reward", 812.4),
    ("cognitive_graph", "cognitive_graph/graph.json", "ingested_episodic", 220),
    ("skills", "eval/skills.json", "skills", None),
    ("checkpoint", "checkpoints/latest.json", "tick_count", 41_207),
])
def test_a_passthrough_store_keeps_its_data(legacy, store, path, key, expected):
    migrated = read_store(legacy / path, store=store)
    assert key in migrated
    if expected is not None:
        assert migrated[key] == pytest.approx(expected) \
            if isinstance(expected, float) else migrated[key] == expected


def test_every_store_comes_back_stamped_at_the_current_version(legacy):
    for path in sorted(legacy.rglob("*.json")):
        migrated = read_store(path)
        assert migrated["schema_version"] == 2, path.name


# ── what the migration drops, deliberately ───────────────────────────

def test_the_replaced_genome_does_not_survive(legacy):
    """§M5.3: the old genes do not move the measured benchmark, which is why
    the genome was replaced. Carrying their values across would carry the
    reason for replacing it across too."""
    champion = read_store(legacy / "evolution" / "lineage.json",
                          store="evolution")["champion"]

    for gene in RETIRED_GENES:
        assert gene not in champion["genome"], gene
    assert champion["migrated_from"] == "genome_v1"


def test_the_migrated_champion_carries_the_current_genome(legacy):
    from aegis.layers.evolution.genome import GENES_BY_NAME

    champion = read_store(legacy / "evolution" / "lineage.json",
                          store="evolution")["champion"]
    assert set(champion["genome"]) == set(GENES_BY_NAME)


def test_a_pending_candidate_is_dropped_rather_than_half_judged(legacy):
    """There is no way to finish judging a v1 candidate correctly: its genome
    is in the old space and its comparison would be against a champion in the
    new one. Dropping it loses one generation; keeping it would put a
    meaningless number into the lineage."""
    migrated = read_store(legacy / "evolution" / "lineage.json",
                          store="evolution")
    assert migrated["candidate"] is None


def test_the_predictive_tables_start_empty_rather_than_invented(legacy):
    """Transitions and outcomes are keyed by encoded system state, which v1
    never recorded. Seeding them from cause→effect strings would put fiction
    into the model and every calibration number after that would be wrong."""
    migrated = read_store(legacy / "world_model" / "model.json",
                          store="world_model")
    assert "transitions" not in migrated or not migrated.get("transitions")
    assert "outcomes" not in migrated or not migrated.get("outcomes")


# ── the torn log ─────────────────────────────────────────────────────

def test_the_experience_log_loses_only_its_torn_line(legacy):
    """A snapshot from a killed process ends mid-line. Sixty complete
    experiences must not be lost for the sake of the sixty-first."""
    rows = []
    for line in (legacy / "feedback" / "experiences.jsonl").read_text(
            encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert len(rows) == 60
    assert rows[0]["id"] == "exp_0000" and rows[-1]["id"] == "exp_0059"


# ── the system boots on it and ticks ─────────────────────────────────

def test_the_system_comes_up_on_a_v1_directory_and_ticks(legacy, monkeypatch):
    """Appendix I's own criterion: it starts, it runs fifty ticks, and neither
    the champion nor the benchmark history is gone afterwards."""
    import importlib

    from aegis.clock import frozen
    from tests.conftest import _STATE_DIRS

    for module_name, constant, subdir in _STATE_DIRS:
        module = importlib.import_module(module_name)
        if not hasattr(module, constant):
            continue
        target = legacy / subdir
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, constant, target, raising=False)

    import aegis.layers.substrate as substrate_mod
    from aegis.layers.state_backup import StateBackup
    monkeypatch.setattr(substrate_mod, "StateBackup",
                        lambda *a, **k: StateBackup(backup_dir=legacy / "backups"),
                        raising=False)

    from aegis.layers.substrate import Substrate

    with frozen() as clock:
        substrate = Substrate()

        async def _no_agents():
            return []

        async def _no_learning(*args, **kwargs):
            return {"success": False}

        substrate.agent_system.run_due_agents = _no_agents
        substrate.external_learning.learn_from_source = _no_learning
        substrate.llm.enabled = False
        substrate.sensors.read_all = lambda: {"pinned": True}
        substrate.evaluator.run = lambda *a, **k: {"score": 0.5, "per_kind": {}}
        substrate.environment.step = lambda *a, **k: {
            "reward": 0.25, "solved": True, "task": "canned", "kind": "calc"}
        substrate.evolution.run_generation = lambda *a, **k: {"generation": 1}

        # It came up carrying the history it was given.
        assert substrate.evolution.champion is not None
        assert substrate.evolution.champion["fitness"] == pytest.approx(0.4815)
        assert substrate.evolution.generation == 88

        async def _drive():
            for _ in range(50):
                await substrate.tick()
                clock.advance(3.0)
            await substrate.cancel_background_tasks()

        asyncio.run(_drive())

        # It resumed from the checkpoint's own tick rather than from zero —
        # which is the point of a checkpoint, and one more thing the migration
        # had to carry across.
        assert substrate.tick_count == 41_207 + 50
        assert substrate.evolution.champion is not None
        assert substrate.evolution.champion["fitness"] == pytest.approx(0.4815)
        assert len(substrate.evolution.lineage) >= 88
        assert substrate.world_model.total_observations >= 150


def test_the_migration_is_idempotent(legacy):
    """Reading a store twice must not migrate it twice. A migration that
    re-ran would apply its transformation to already-transformed data, and
    `_v1_to_v2_evolution` would reseed a champion that had already been
    reseeded — losing whatever it had learned since."""
    path = legacy / "evolution" / "lineage.json"
    once = read_store(path, store="evolution")

    from aegis.store.migrations import write_store
    write_store(path, once)

    twice = read_store(path, store="evolution")
    assert twice["champion"]["fitness"] == once["champion"]["fitness"]
    assert twice["champion"]["genome"] == once["champion"]["genome"]
    assert twice["generation"] == once["generation"]
