"""Tests for archetypes and their geopolitics."""
from aegis.layers.archetypes import (
    Archetype,
    create_default_archetypes,
    ArchetypeGeopolitics,
)


def _make_archetype(name="Test", moods=None, energy_range=(0.0, 0.3)):
    return Archetype(
        name=name,
        activation_moods=moods or ["anxious"],
        energy_range=energy_range,
        strategies={"heuristic": "watching", "reflective": "thinking"},
        tone="Calm",
    )


def test_should_activate_on_mood_match():
    a = _make_archetype(moods=["anxious"])
    assert a.should_activate("anxious", energy=0.9) is True


def test_should_activate_on_low_energy_without_mood():
    a = _make_archetype(moods=["anxious"], energy_range=(0.0, 0.3))
    # mood not matching but energy in range and < 0.3
    assert a.should_activate("joy", energy=0.2) is True


def test_should_not_activate_energy_out_of_range():
    a = _make_archetype(moods=["anxious"], energy_range=(0.0, 0.3))
    assert a.should_activate("joy", energy=0.8) is False


def test_should_not_activate_energy_in_range_but_not_below_threshold():
    a = _make_archetype(moods=["anxious"], energy_range=(0.0, 0.5))
    # energy in range but not < 0.3 and mood does not match
    assert a.should_activate("joy", energy=0.4) is False


def test_act_uses_strategy_for_mode():
    a = _make_archetype()
    out = a.act("heuristic", "explore")
    assert "watching" in out
    assert "explore" in out
    assert a.name in out


def test_act_falls_back_for_unknown_mode():
    a = _make_archetype()
    out = a.act("survival", "explore")
    assert "operating in default mode" in out


def test_log_experience_updates_score():
    a = _make_archetype()
    a.log_experience(tick=1, mood="joy", reward=1.0, action="do")
    assert a.steps_active == 1
    assert len(a.experience) == 1
    # 0.9*0.5 + 0.1*1.0 = 0.55
    assert a.success_score > 0.5


def test_experience_truncates_at_200():
    a = _make_archetype()
    for i in range(250):
        a.log_experience(tick=i, mood="joy", reward=0.5, action="do")
    assert len(a.experience) <= 200


def test_to_dict():
    a = _make_archetype(name="Sentinel")
    d = a.to_dict()
    assert d["name"] == "Sentinel"
    assert d["tone"] == "Calm"
    assert d["steps_active"] == 0
    assert d["experience_count"] == 0


def test_create_default_archetypes():
    arcs = create_default_archetypes()
    names = {a.name for a in arcs}
    assert names == {"Sentinel", "Explorer", "Caretaker"}


def test_geopolitics_initial_state():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    assert set(geo.influence.keys()) == {"Sentinel", "Explorer", "Caretaker"}
    assert geo.dominant is None


def test_update_influence_sets_dominant():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    geo.update_influence()
    assert geo.dominant in geo.influence
    assert geo.get_dominant() is not None


def test_update_influence_positive_mood_boost():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    explorer = geo.archetypes["Explorer"]
    explorer.success_score = 0.8
    explorer.log_experience(tick=1, mood="inspired", reward=0.8, action="x")
    geo.update_influence()
    # inspired mood multiplies base by 1.2
    assert geo.influence["Explorer"] > 0


def test_update_influence_negative_mood_penalty():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    sentinel = geo.archetypes["Sentinel"]
    sentinel.success_score = 0.8
    sentinel.log_experience(tick=1, mood="fear", reward=0.2, action="x")
    geo.update_influence()
    # fear multiplies base by 0.7; floor of 0.1 enforced
    assert geo.influence["Sentinel"] >= 0.1


def test_update_influence_relationship_floor():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    geo.update_influence()
    assert all(v >= 0.1 for v in geo.influence.values())


def test_detect_conflict_insufficient_data():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    assert geo.detect_conflict() is False


def test_detect_conflict_true():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    # Give two archetypes >5 experiences with widely differing success scores.
    high = geo.archetypes["Explorer"]
    low = geo.archetypes["Sentinel"]
    for i in range(6):
        high.log_experience(tick=i, mood="joy", reward=1.0, action="x")
        low.log_experience(tick=i, mood="sadness", reward=0.0, action="x")
    high.success_score = 0.95
    low.success_score = 0.1
    assert geo.detect_conflict() is True


def test_detect_conflict_no_conflict_when_close():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    for name in ("Explorer", "Sentinel"):
        a = geo.archetypes[name]
        for i in range(6):
            a.log_experience(tick=i, mood="joy", reward=0.5, action="x")
        a.success_score = 0.5
    assert geo.detect_conflict() is False


def test_status_shape():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    geo.update_influence()
    s = geo.status()
    assert "dominant" in s
    assert "influence" in s
    assert "conflict_detected" in s
    assert set(s["archetypes"].keys()) == {"Sentinel", "Explorer", "Caretaker"}


def test_update_influence_neutral_last_mood():
    arcs = create_default_archetypes()
    geo = ArchetypeGeopolitics(arcs)
    a = geo.archetypes["Caretaker"]
    a.log_experience(tick=1, mood="contentment", reward=0.5, action="x")
    geo.update_influence()  # neutral-ish mood, no ×1.2 or ×0.7
    assert geo.influence["Caretaker"] >= 0.1
