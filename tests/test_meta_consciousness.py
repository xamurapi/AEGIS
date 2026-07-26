"""Tests for MetaConsciousness."""
from aegis.layers.meta_consciousness import MetaConsciousness


class _Archetype:
    def __init__(self, success_score):
        self.success_score = success_score


def test_initial_state():
    mc = MetaConsciousness()
    assert mc.fragmentation_score == 0.0
    assert mc.coherence_score == 1.0
    assert mc.recommendations == []


def test_no_conflicts_well_integrated():
    mc = MetaConsciousness()
    res = mc.evaluate("heuristic", None, "neutral", 0.8, None)
    assert res["conflicts"] == []
    assert res["coherence"] == 1.0
    assert res["fragmentation"] == 0.0
    # coherence > 0.9 -> well-integrated recommendation
    assert any("well-integrated" in r for r in res["recommendations"])


def test_survival_explorer_conflict():
    mc = MetaConsciousness()
    res = mc.evaluate("survival", "Explorer", "neutral", 0.8, None)
    assert any("Explorer" in c for c in res["conflicts"])
    assert res["alignment_score"] < 1.0


def test_low_energy_reflective_conflict():
    mc = MetaConsciousness()
    res = mc.evaluate("reflective", None, "neutral", 0.1, None)
    assert any("Reflective" in c for c in res["conflicts"])


def test_negative_mood_optimization_conflict():
    mc = MetaConsciousness()
    res = mc.evaluate("heuristic", None, "anxious", 0.8, "optimize memory")
    assert any("optimization" in c for c in res["conflicts"])


def test_negative_mood_without_optim_goal_no_conflict():
    mc = MetaConsciousness()
    res = mc.evaluate("heuristic", None, "fear", 0.8, "explore world")
    assert res["conflicts"] == []


def test_archetype_spread_conflict():
    mc = MetaConsciousness()
    archs = [_Archetype(0.1), _Archetype(0.9)]
    res = mc.evaluate("heuristic", None, "neutral", 0.8, None, archetypes=archs)
    assert any("fragmentation" in c for c in res["conflicts"])


def test_archetype_low_spread_no_conflict():
    mc = MetaConsciousness()
    archs = [_Archetype(0.5), _Archetype(0.6)]
    res = mc.evaluate("heuristic", None, "neutral", 0.8, None, archetypes=archs)
    assert res["conflicts"] == []


def test_single_archetype_ignored():
    mc = MetaConsciousness()
    res = mc.evaluate("heuristic", None, "neutral", 0.8, None, archetypes=[_Archetype(0.1)])
    assert res["conflicts"] == []


def test_fragmentation_recommendation_and_conflict_count():
    mc = MetaConsciousness()
    # Two big conflicts push fragmentation above 0.4
    archs = [_Archetype(0.1), _Archetype(0.9)]
    res = mc.evaluate("survival", "Explorer", "neutral", 0.8, None, archetypes=archs)
    assert mc.fragmentation_score > 0.4
    assert any("fragmentation" in r for r in res["recommendations"])
    assert any("conflict" in r for r in res["recommendations"])
    # recommendations capped at 3
    assert len(res["recommendations"]) <= 3


def test_very_low_energy_recommendation():
    mc = MetaConsciousness()
    res = mc.evaluate("heuristic", None, "neutral", 0.1, None)
    assert any("instinctive" in r for r in res["recommendations"])


def test_evaluations_recorded():
    mc = MetaConsciousness()
    mc.evaluate("heuristic", None, "neutral", 0.8, None)
    mc.evaluate("heuristic", None, "neutral", 0.8, None)
    assert len(mc.evaluations) == 2


def test_log_integration_event():
    mc = MetaConsciousness()
    mc.log_integration_event("merge", "two archetypes merged")
    assert len(mc.integration_events) == 1
    assert mc.integration_events[0]["type"] == "merge"


def test_status_empty():
    mc = MetaConsciousness()
    st = mc.status()
    assert st["last_evaluation"] is None
    assert st["evaluations_count"] == 0
    assert st["integration_events"] == 0


def test_status_after_evaluation():
    mc = MetaConsciousness()
    mc.evaluate("heuristic", None, "neutral", 0.8, None)
    mc.log_integration_event("resolve", "conflict resolved")
    st = mc.status()
    assert st["last_evaluation"] is not None
    assert st["evaluations_count"] == 1
    assert st["integration_events"] == 1
