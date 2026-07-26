"""Tests for MemorySystem hardening (MEDIUM defect fixes):

- old-format episodic entries (missing keys) don't raise KeyError;
- episodic/semantic have soft RAM caps;
- forgotten_total round-trips through save/_load;
- add_semantic preserves the original `created` timestamp on update.

Isolation: MEMORY_DIR is redirected to tmp_path so no real state file is
touched and tests never leak persisted memory into one another.
"""
import json
import time
import pytest
import aegis.layers.memory as memory_mod
from aegis.layers.memory import (
    MemorySystem, MAX_EPISODIC_RAM, MAX_SEMANTIC_CONCEPTS,
)


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """A MemorySystem whose persistence lives entirely under tmp_path."""
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path)
    return MemorySystem()


# ── (a) old-format entries: no KeyError ──────────────────────────

def test_recall_episodic_tolerates_missing_keys(isolated_memory):
    m = isolated_memory
    # Simulate an old persisted format lacking access_count/last_access.
    m.episodic = [{"event": "old style", "timestamp": time.time()}]
    res = m.recall_episodic("old", limit=5)
    assert len(res) == 1
    assert res[0]["access_count"] == 1  # defaulted then incremented


def test_recall_episodic_tolerates_missing_event(isolated_memory):
    m = isolated_memory
    m.episodic = [{"timestamp": time.time()}]  # no "event" at all
    # Empty query returns everything without crashing.
    res = m.recall_episodic("", limit=5)
    assert len(res) == 1


def test_apply_forgetting_tolerates_missing_keys(isolated_memory):
    m = isolated_memory
    m.episodic = [
        {"event": "bare"},  # no timestamp/importance/access_count
        {"event": "recent", "timestamp": time.time(), "importance": 0.9, "access_count": 3},
    ]
    forgotten = m.apply_forgetting()  # must not raise
    assert isinstance(forgotten, int)
    events = [e["event"] for e in m.episodic]
    assert "recent" in events


def test_status_tolerates_missing_keys(isolated_memory):
    m = isolated_memory
    m.episodic = [{"foo": "bar"}]  # totally malformed
    st = m.status()  # must not raise
    assert st["episodic_count"] == 1
    assert st["recent_episodic"][0]["event"] == ""


# ── (b) soft RAM caps ────────────────────────────────────────────

def test_episodic_soft_cap(isolated_memory):
    m = isolated_memory
    for i in range(MAX_EPISODIC_RAM + 250):
        m.add_episodic(f"ev {i}")
    assert len(m.episodic) <= MAX_EPISODIC_RAM
    # Most-recent entries retained.
    assert m.episodic[-1]["event"] == f"ev {MAX_EPISODIC_RAM + 249}"


def test_semantic_soft_cap_prunes_least_recently_updated(isolated_memory):
    m = isolated_memory
    # First concept updated long ago; it should be pruned when we overflow.
    m.add_semantic("stale", {"summary": "old"})
    m.semantic["stale"]["updated"] = 1.0  # ancient
    for i in range(MAX_SEMANTIC_CONCEPTS + 50):
        m.add_semantic(f"c{i}", {"summary": str(i)})
    assert len(m.semantic) <= MAX_SEMANTIC_CONCEPTS
    assert "stale" not in m.semantic  # least-recently-updated pruned


# ── (c) forgotten_total round-trips ──────────────────────────────

def test_forgotten_total_persists_across_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path)
    m1 = MemorySystem()
    m1.forgotten_total = 42
    m1.save()
    # A raw check that it was written.
    data = json.loads((tmp_path / "memory_state.json").read_text(encoding="utf-8"))
    assert data["forgotten_total"] == 42
    # Fresh instance loads it back.
    m2 = MemorySystem()
    assert m2.forgotten_total == 42


# ── (d) add_semantic preserves created ───────────────────────────

def test_add_semantic_preserves_created_on_update(isolated_memory):
    m = isolated_memory
    m.add_semantic("topic", {"summary": "v1"})
    created0 = m.semantic["topic"]["created"]
    time.sleep(0.01)
    m.add_semantic("topic", {"summary": "v2"})
    assert m.semantic["topic"]["created"] == created0        # unchanged
    assert m.semantic["topic"]["updated"] >= created0        # bumped
    assert m.semantic["topic"]["relations"]["summary"] == "v2"
