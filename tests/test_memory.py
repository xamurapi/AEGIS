"""Tests for the memory system."""
import time
from aegis.layers.memory import MemorySystem


def test_semantic_payload_is_nested_under_relations():
    m = MemorySystem()
    m.add_semantic("transformers", {"summary": "attention is all you need", "confidence": 0.6})
    entry = m.semantic["transformers"]
    # The audit fix relies on this nesting being stable.
    assert entry["relations"]["summary"] == "attention is all you need"


def test_episodic_recall_filters_by_query():
    m = MemorySystem()
    m.add_episodic("learned about cats", importance=0.5)
    m.add_episodic("learned about dogs", importance=0.5)
    res = m.recall_episodic("cats", limit=10)
    assert len(res) == 1
    assert "cats" in res[0]["event"]


def test_recall_increments_access_count():
    m = MemorySystem()
    m.add_episodic("event", importance=0.5)
    m.recall_episodic("event")
    assert m.episodic[-1]["access_count"] == 1


def test_forgetting_keeps_important_recent():
    m = MemorySystem()
    # Old, unimportant memory should be forgotten.
    m.add_episodic("trivial old", importance=0.1)
    m.episodic[-1]["timestamp"] = time.time() - 3600 * 500  # very old
    m.add_episodic("fresh", importance=0.9)
    m.apply_forgetting()
    events = [e["event"] for e in m.episodic]
    assert "fresh" in events
    assert "trivial old" not in events


def test_working_memory_is_capped():
    m = MemorySystem()
    for i in range(200):
        m.add_working({"i": i})
    from aegis.config import MAX_WORKING_MEMORY
    assert len(m.working) <= MAX_WORKING_MEMORY


def test_semantic_summary_helper_reads_nested():
    from aegis.api.server import _semantic_summary
    m = MemorySystem()
    m.add_semantic("topic", {"summary": "the summary"})
    assert _semantic_summary(m.semantic["topic"]) == "the summary"


def test_rag_retrieve_ranks_relevant_concept_first():
    m = MemorySystem()
    m.add_semantic("reinforcement learning", {"summary": "agents learn from reward signals"})
    m.add_semantic("photosynthesis", {"summary": "plants convert light to energy"})
    m.add_semantic("neural networks", {"summary": "layers of weighted connections"})
    results = m.retrieve("how do agents learn from reward", k=3)
    assert results
    assert results[0]["concept"] == "reinforcement learning"


def test_rag_retrieve_empty_query_returns_nothing():
    m = MemorySystem()
    m.add_semantic("x", {"summary": "y"})
    assert m.retrieve("", k=3) == []
