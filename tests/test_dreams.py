"""Tests for the Dream Engine."""
from aegis.layers.dreams import (
    DreamEngine, _deterministic_pick, MOOD_MOTIF_WEIGHTS, MOTIFS, THEMES,
)


def test_deterministic_pick_empty():
    assert _deterministic_pick({}, 0) == "encounter"


def test_deterministic_pick_returns_key():
    weights = {"a": 1, "b": 2, "c": 3}
    for seed in range(20):
        assert _deterministic_pick(weights, seed) in weights


def test_deterministic_pick_is_deterministic():
    weights = {"a": 1, "b": 2, "c": 3}
    assert _deterministic_pick(weights, 42) == _deterministic_pick(weights, 42)


def test_generate_dream_basic():
    e = DreamEngine()
    d = e.generate_dream("joy", ["saw a cat"], ["felines"])
    assert d["id"] == 1
    assert d["motif"] in MOTIFS
    assert d["theme"] in THEMES
    assert d["mood_source"] == "joy"
    assert "saw a cat" in d["fragments"]
    assert "felines" in d["fragments"]
    assert "Echoes of" in d["narrative"]
    assert e.dream_count == 1


def test_generate_dream_no_fragments():
    e = DreamEngine()
    d = e.generate_dream("neutral", [], [])
    assert d["fragments"] == []
    assert "Echoes of" not in d["narrative"]


def test_generate_dream_only_events():
    e = DreamEngine()
    d = e.generate_dream("neutral", ["event"], [])
    assert d["fragments"] == ["event"]


def test_generate_dream_unknown_mood_uses_neutral_weights():
    e = DreamEngine()
    d = e.generate_dream("some_unknown_mood", [], [])
    # motif still valid, picked from neutral weights
    assert d["motif"] in MOOD_MOTIF_WEIGHTS["neutral"]


def test_symbols_capped_at_two():
    e = DreamEngine()
    d = e.generate_dream("fear", [], [])
    assert len(d["symbols"]) == 2


def test_dreams_capped_at_50():
    e = DreamEngine()
    for _ in range(55):
        e.generate_dream("neutral", [], [])
    assert len(e.dreams) == 50
    assert e.dream_count == 55


def test_interpret_tension():
    e = DreamEngine()
    assert "tension" in e._interpret("pursuit", "fear", [])
    assert "tension" in e._interpret("descent", "anxious", [])


def test_interpret_knowledge():
    e = DreamEngine()
    assert "knowledge" in e._interpret("discovery", "curiosity", [])
    assert "knowledge" in e._interpret("ascent", "inspired", [])


def test_interpret_transformation():
    e = DreamEngine()
    assert "restructuring" in e._interpret("transformation", "neutral", [])


def test_interpret_default_consolidation():
    e = DreamEngine()
    assert "consolidation" in e._interpret("encounter", "neutral", [])


def test_status():
    e = DreamEngine()
    e.generate_dream("joy", ["x"], ["y"])
    st = e.status()
    assert st["total_dreams"] == 1
    assert len(st["recent_dreams"]) == 1
    assert st["recent_dreams"][0]["id"] == 1
