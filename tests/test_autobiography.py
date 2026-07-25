"""Tests for the Autobiographer."""
from aegis.layers.autobiography import Autobiographer


def test_log_event_records_and_accumulates_impact():
    a = Autobiographer()
    a.log_event("learning", "learned X", impact=0.5)
    assert len(a.events) == 1
    assert a.total_impact == 0.5
    assert a.events[0]["category"] == "learning"


def test_log_event_clamps_impact():
    a = Autobiographer()
    a.log_event("x", "over", impact=5.0)
    a.log_event("y", "under", impact=-3.0)
    assert a.events[0]["impact"] == 1.0
    assert a.events[1]["impact"] == 0.0


def test_events_truncate_at_500():
    a = Autobiographer()
    for i in range(600):
        a.log_event("c", f"event {i}", impact=0.1)
    assert len(a.events) <= 500


def test_generate_narrative_empty():
    a = Autobiographer()
    assert a.generate_narrative() == "No significant events yet."


def test_generate_narrative_with_events():
    a = Autobiographer()
    a.log_event("learning", "learned something", impact=0.8)
    narrative = a.generate_narrative()
    assert "Life narrative:" in narrative
    assert "learned something" in narrative
    assert "[learning]" in narrative


def test_generate_narrative_last_n():
    a = Autobiographer()
    for i in range(20):
        a.log_event("c", f"event {i}", impact=0.1)
    narrative = a.generate_narrative(last_n=3)
    # only last 3 events + the header line
    assert narrative.count("[c]") == 3
    assert "event 19" in narrative
    assert "event 0" not in narrative


def test_get_milestones_default_threshold():
    a = Autobiographer()
    a.log_event("c", "minor", impact=0.5)
    a.log_event("c", "major", impact=0.9)
    milestones = a.get_milestones()
    assert len(milestones) == 1
    assert milestones[0]["summary"] == "major"


def test_get_milestones_custom_threshold():
    a = Autobiographer()
    a.log_event("c", "a", impact=0.4)
    a.log_event("c", "b", impact=0.6)
    assert len(a.get_milestones(threshold=0.3)) == 2


def test_status_shape():
    a = Autobiographer()
    a.log_event("learning", "x", impact=0.9)
    a.log_event("event", "y", impact=0.2)
    s = a.status()
    assert s["total_events"] == 2
    assert s["total_impact"] == round(0.9 + 0.2, 2)
    assert s["milestones"] == 1
    assert len(s["recent"]) == 2
    assert s["recent"][0]["category"] == "learning"


def test_status_recent_truncates_summary():
    a = Autobiographer()
    a.log_event("c", "z" * 200, impact=0.5)
    s = a.status()
    assert len(s["recent"][0]["summary"]) == 60
