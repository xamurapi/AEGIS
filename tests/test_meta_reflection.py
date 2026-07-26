"""Tests for MetaReflection."""
from aegis.layers.meta_reflection import MetaReflection


def _reflect_series(r, energies, mood_valence=0.5, error_rate=0.0,
                    goals_completed=0, goals_total=1, mode="heuristic",
                    events=None):
    out = None
    for i, e in enumerate(energies):
        out = r.reflect(i, e, "neutral", mood_valence, error_rate,
                        goals_completed, goals_total, mode, events or [])
    return out


def test_initial_state():
    r = MetaReflection()
    assert r.reflection_count == 0
    assert r.trends["energy"] == []


def test_reflection_count_and_trends_tracked():
    r = MetaReflection()
    r.reflect(0, 0.5, "neutral", 0.5, 0.0, 1, 2, "heuristic", [])
    assert r.reflection_count == 1
    assert r.trends["energy"] == [0.5]
    assert r.trends["goal_completion"] == [0.5]


def test_goal_completion_division_guard():
    r = MetaReflection()
    r.reflect(0, 0.5, "neutral", 0.5, 0.0, 0, 0, "heuristic", [])
    assert r.trends["goal_completion"] == [0.0]


def test_energy_declining_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.9, 0.8, 0.7, 0.6, 0.5])
    assert any("declining" in i for i in out["insights"])
    assert out["trends_summary"]["energy_trend"] == "falling"


def test_energy_rising_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert any("rising" in i for i in out["insights"])
    assert out["trends_summary"]["energy_trend"] == "rising"


def test_negative_mood_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.5] * 5, mood_valence=0.2)
    assert any("negative" in i for i in out["insights"])


def test_positive_mood_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.5] * 5, mood_valence=0.8)
    assert any("thriving" in i for i in out["insights"])


def test_high_error_rate_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.5, 0.5, 0.5], error_rate=0.5)
    assert any("error rate" in i for i in out["insights"])


def test_low_goal_completion_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.5] * 5, goals_completed=0, goals_total=10)
    assert any("too ambitious" in i for i in out["insights"])


def test_high_goal_completion_insight():
    r = MetaReflection()
    out = _reflect_series(r, [0.5] * 5, goals_completed=9, goals_total=10)
    assert any("challenging" in i for i in out["insights"])


def test_instinctive_mode_insight():
    r = MetaReflection()
    out = r.reflect(0, 0.7, "neutral", 0.5, 0.0, 1, 2, "instinctive", [])
    assert any("instinctive" in i for i in out["insights"])


def test_event_error_pattern_insight():
    r = MetaReflection()
    events = ["error occurred", "task failed", "normal event"]
    out = r.reflect(0, 0.5, "neutral", 0.5, 0.0, 1, 2, "heuristic", events)
    assert any("systemic issue" in i for i in out["insights"])


def test_insights_recorded_in_deque():
    r = MetaReflection()
    _reflect_series(r, [0.9, 0.8, 0.7, 0.6, 0.5])
    assert len(r.insights) >= 1


def test_trend_direction_insufficient():
    assert MetaReflection._trend_direction([0.1]) == "insufficient_data"


def test_trend_direction_rising_falling_stable():
    assert MetaReflection._trend_direction([0.1, 0.2, 0.3]) == "rising"
    assert MetaReflection._trend_direction([0.3, 0.2, 0.1]) == "falling"
    assert MetaReflection._trend_direction([0.2, 0.3, 0.2]) == "stable"


def test_trends_capped_at_50():
    r = MetaReflection()
    _reflect_series(r, [0.5] * 60)
    assert len(r.trends["energy"]) == 50


def test_status():
    r = MetaReflection()
    _reflect_series(r, [0.9, 0.8, 0.7, 0.6, 0.5])
    st = r.status()
    assert st["reflection_count"] == 5
    assert "energy" in st["trends"]
    assert st["self_reports_count"] == 5
