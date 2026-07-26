"""Tests for the Worldview / ValueSystem."""
from aegis.layers.worldview import Value, ValueSystem, Worldview


def test_value_reinforce_positive_raises_priority():
    v = Value("x", "def", priority=0.5)
    v.reinforce("ctx", alignment=1.0)
    assert v.priority > 0.5
    assert len(v.reinforcements) == 1


def test_value_reinforce_negative_lowers_priority():
    v = Value("x", "def", priority=0.5)
    v.reinforce("ctx", alignment=0.0)
    assert v.priority < 0.5


def test_value_priority_clamped():
    v = Value("x", "def", priority=0.99)
    for _ in range(20):
        v.reinforce("ctx", alignment=1.0)
    assert v.priority <= 1.0
    v2 = Value("y", "def", priority=0.01)
    for _ in range(20):
        v2.reinforce("ctx", alignment=0.0)
    assert v2.priority >= 0.0


def test_value_reinforcements_truncate_at_100():
    v = Value("x", "def")
    for _ in range(150):
        v.reinforce("ctx", alignment=0.5)
    assert len(v.reinforcements) <= 100


def test_value_to_dict():
    v = Value("understanding", "definition text", priority=0.6)
    d = v.to_dict()
    assert d["name"] == "understanding"
    assert d["definition"] == "definition text"
    assert d["priority"] == 0.6
    assert d["reinforcement_count"] == 0


def test_value_system_has_five_values():
    vs = ValueSystem()
    names = {v.name for v in vs.values}
    assert names == {"understanding", "harmony", "growth", "preservation", "curiosity"}


def test_evaluate_action_understanding():
    vs = ValueSystem()
    vs.evaluate_action(mood="neutral", reward=1.0, goal="analyze")
    understanding = next(v for v in vs.values if v.name == "understanding")
    # alignment 0.5 + 1.0*0.4 = 0.9 -> priority rises
    assert understanding.priority > 0.5


def test_evaluate_action_harmony():
    vs = ValueSystem()
    vs.evaluate_action(mood="contentment", reward=0.0, goal="idle")
    harmony = next(v for v in vs.values if v.name == "harmony")
    assert harmony.priority > 0.5


def test_evaluate_action_growth():
    vs = ValueSystem()
    vs.evaluate_action(mood="inspired", reward=0.0, goal="idle")
    growth = next(v for v in vs.values if v.name == "growth")
    assert growth.priority > 0.5


def test_evaluate_action_preservation():
    vs = ValueSystem()
    vs.evaluate_action(mood="anxious", reward=0.0, goal="idle")
    preservation = next(v for v in vs.values if v.name == "preservation")
    assert preservation.priority > 0.5


def test_evaluate_action_curiosity():
    vs = ValueSystem()
    vs.evaluate_action(mood="neutral", reward=1.0, goal="explore_topic")
    curiosity = next(v for v in vs.values if v.name == "curiosity")
    # 0.5 + 1.0*0.3 = 0.8
    assert curiosity.priority > 0.5


def test_evaluate_action_neutral_defaults():
    vs = ValueSystem()
    # No branch matches -> everyone gets alignment 0.5 (no priority change)
    before = [v.priority for v in vs.values]
    vs.evaluate_action(mood="disgust", reward=0.5, goal="unknown_goal")
    after = [v.priority for v in vs.values]
    assert before == after
    # But reinforcement was still recorded.
    assert all(len(v.reinforcements) == 1 for v in vs.values)


def test_value_system_status():
    vs = ValueSystem()
    s = vs.status()
    assert len(s["values"]) == 5
    assert all("name" in v for v in s["values"])


def test_worldview_status():
    w = Worldview()
    s = w.status()
    assert len(s["axioms"]) == 4
    assert len(s["metaphors"]) == 3
    assert "learning" in s["beliefs"]
    assert s["beliefs"]["error"] == "a point of growth, not failure"
