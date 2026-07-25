"""Unit tests for aegis.layers.dataset_builder.

Isolation: WEIGHT_DATASETS_DIR is redirected into pytest's tmp_path so no test
ever writes into the real data/ tree. No network involved in this module.
"""
import json
import types

import pytest

import aegis.layers.dataset_builder as db
from aegis.layers.dataset_builder import DatasetBuilder


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setattr(db, "WEIGHT_DATASETS_DIR", d)
    return d


def make_memory(semantic=None, episodic=None, procedural=None):
    return types.SimpleNamespace(
        semantic=semantic or {},
        episodic=episodic or [],
        procedural=procedural or [],
    )


def make_agent_system(items):
    return types.SimpleNamespace(collected_knowledge=list(items))


def _sem(relations):
    return {"relations": relations}


# ── build_from_memory: source coverage ────────────────────────────────────

def test_build_covers_all_sample_sources(datasets_dir):
    semantic = {
        "transformers": _sem({"type": "external_learning",
                              "summary": "Attention based sequence models that scale well."}),
        "too_short": _sem({"type": "external_learning", "summary": "tiny"}),  # <20 -> skipped
        "empty_sum": _sem({"type": "external_learning"}),  # no summary -> skipped
        "concept_x": _sem({"type": "learned_concept",
                           "definition": "A precise definition of concept x here."}),
        "curio": _sem({"type": "curiosity_exploration",
                       # NOTE: a summary is required to pass the length guard before
                       # the curiosity branch is reached (see BUG note in report).
                       "summary": "Curiosity entry that also carries a summary field.",
                       "question": "Why does the sky appear blue in daytime?",
                       "connection": "Rayleigh scattering of shorter wavelengths."}),
        "unmatched": _sem({"type": "some_unknown_type",
                           "summary": "This summary is long enough but matches no branch."}),
    }
    episodic = [
        {"event": "Reflection: I noticed that persistent effort compounds over long stretches.",
         "importance": 0.8},
        {"event": "Reflection: short", "importance": 0.9},  # <=30 chars after strip -> skipped
        {"event": "LLM Insight: my exploration rate should decrease as confidence grows.",
         "importance": 0.65},
        {"event": "LLM Insight: tiny", "importance": 0.7},  # <=20 -> skipped
        {"event": "Reflection: ignored low importance one here", "importance": 0.3},  # low imp
    ]
    procedural = [
        {"name": "solve maze", "procedure": {"steps": "turn left, then right, repeat"}},
        {"name": "no steps", "procedure": {}},  # skipped
    ]
    agent = make_agent_system([
        {"source": "arxiv", "data": {"title": "P1", "summary": "A" * 40}},
        {"source": "wikipedia", "data": {"title": "W1", "summary": "B" * 40}},
        {"source": "github", "data": {"title": "G1", "summary": "C" * 40}},
        {"source": "news", "data": {"title": "N1", "summary": "D" * 40}},
        {"source": "arxiv", "data": {"title": "short", "summary": "tooshort"}},  # <=30 skipped
        {"source": "unknown_src", "data": {"title": "X", "summary": "E" * 40}},  # no branch
    ])

    b = DatasetBuilder()
    res = b.build_from_memory(make_memory(semantic, episodic, procedural), agent)

    assert res["success"] is True
    sources = res["source_distribution"]
    assert "external_learning" in sources
    assert "self_reflection" in sources
    assert "curiosity" in sources
    assert "episodic_reflection" in sources
    assert "llm_insight" in sources
    assert "agent_arxiv" in sources
    assert "agent_wikipedia" in sources
    assert "agent_github" in sources
    assert "agent_news" in sources
    assert "procedural" in sources

    # Files exist and are valid jsonl.
    dataset_dir = datasets_dir / f"dataset_{res['timestamp']}"
    train = (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val = (dataset_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train) == res["train_size"]
    assert len(val) == res["val_size"]
    for line in train + val:
        obj = json.loads(line)
        assert {"instruction", "output", "source"} <= set(obj)

    # Bookkeeping updated.
    assert b.builds_total == 1
    assert b.last_dataset_size == res["total_size"]
    assert b.last_dataset_path == str(dataset_dir)


def test_build_deduplicates_by_output(datasets_dir):
    # Two concepts with identical output -> only one sample survives dedup.
    semantic = {
        "a": _sem({"type": "external_learning", "summary": "Identical output text repeated here."}),
        "b": _sem({"type": "external_learning", "summary": "Identical output text repeated here."}),
    }
    b = DatasetBuilder()
    res = b.build_from_memory(make_memory(semantic))
    assert res["total_size"] == 1


def test_build_no_samples_returns_failure(datasets_dir):
    b = DatasetBuilder()
    res = b.build_from_memory(make_memory())
    assert res["success"] is False
    assert res["size"] == 0
    # Nothing written.
    assert list(datasets_dir.glob("dataset_*")) == []


def test_build_without_agent_system(datasets_dir):
    semantic = {
        "topic": _sem({"type": "external_learning",
                       "summary": "A sufficiently long summary about a topic here."}),
    }
    b = DatasetBuilder()
    res = b.build_from_memory(make_memory(semantic), agent_system=None)
    assert res["success"] is True
    assert res["total_size"] == 1


def test_build_learned_concept_summary_fallback(datasets_dir):
    # learned_concept with no explicit definition -> uses summary as definition.
    semantic = {
        "c": _sem({"type": "learned_concept",
                   "summary": "Summary doubling as the definition text here."}),
    }
    b = DatasetBuilder()
    res = b.build_from_memory(make_memory(semantic))
    assert res["success"] is True
    assert res["source_distribution"].get("self_reflection") == 1


# ── get_latest_dataset ────────────────────────────────────────────────────

def test_get_latest_dataset_none_when_empty(datasets_dir):
    b = DatasetBuilder()
    assert b.get_latest_dataset() is None


def test_get_latest_dataset_skips_dirs_without_train(datasets_dir):
    (datasets_dir / "dataset_100").mkdir()  # no train.jsonl
    good = datasets_dir / "dataset_200"
    good.mkdir()
    (good / "train.jsonl").write_text("{}\n", encoding="utf-8")
    b = DatasetBuilder()
    assert b.get_latest_dataset() == good


# ── cleanup_old_datasets ──────────────────────────────────────────────────

def test_cleanup_keeps_n_recent(datasets_dir):
    for ts in (100, 200, 300, 400):
        d = datasets_dir / f"dataset_{ts}"
        d.mkdir()
        (d / "train.jsonl").write_text("{}\n", encoding="utf-8")
    b = DatasetBuilder()
    removed = b.cleanup_old_datasets(keep=2)
    assert removed == 2
    remaining = sorted(p.name for p in datasets_dir.glob("dataset_*"))
    assert remaining == ["dataset_300", "dataset_400"]


def test_cleanup_handles_errors(datasets_dir):
    # dataset_100 is a FILE (not a dir) and, being the oldest, lands in the
    # removable slice -> iterdir raises -> caught, not counted.
    keep_dir = datasets_dir / "dataset_500"
    keep_dir.mkdir()
    (keep_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (datasets_dir / "dataset_100").write_text("x", encoding="utf-8")
    b = DatasetBuilder()
    removed = b.cleanup_old_datasets(keep=1)
    # The file entry could not be removed as a dir -> not counted.
    assert removed == 0


# ── status ────────────────────────────────────────────────────────────────

def test_status_report(datasets_dir):
    semantic = {
        "topic": _sem({"type": "external_learning",
                       "summary": "A sufficiently long summary about the topic here."}),
    }
    b = DatasetBuilder()
    b.build_from_memory(make_memory(semantic))
    st = b.status()
    assert st["builds_total"] == 1
    assert st["datasets_on_disk"] == 1
    assert st["latest_dataset"] is not None
    assert st["last_dataset_size"] == 1
