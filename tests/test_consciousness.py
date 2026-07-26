"""Tests for the consciousness mode-switching state machine."""
from aegis.layers.consciousness import ConsciousnessState


def test_initial_mode_is_heuristic():
    c = ConsciousnessState()
    assert c.mode == "heuristic"


def test_survival_mode_on_critical_energy():
    c = ConsciousnessState()
    assert c.update_mode("joy", energy=0.1) == "survival"


def test_survival_takes_priority_over_instinctive_moods():
    c = ConsciousnessState()
    # Even a scary mood yields survival when energy is critically low.
    assert c.update_mode("fear", energy=0.05) == "survival"


def test_instinctive_on_negative_mood():
    c = ConsciousnessState()
    assert c.update_mode("anxious", energy=0.8) == "instinctive"


def test_instinctive_on_low_energy():
    c = ConsciousnessState()
    assert c.update_mode("joy", energy=0.2) == "instinctive"


def test_reflective_mode():
    c = ConsciousnessState()
    assert c.update_mode("curiosity", energy=0.8, arousal=0.5) == "reflective"


def test_reflective_blocked_by_high_arousal():
    c = ConsciousnessState()
    # curiosity + high energy but arousal too high -> falls to heuristic
    assert c.update_mode("curiosity", energy=0.8, arousal=0.9) == "heuristic"


def test_heuristic_default():
    c = ConsciousnessState()
    assert c.update_mode("neutral", energy=0.8) == "heuristic"


def test_switch_log_records_transition():
    c = ConsciousnessState()
    c.update_mode("anxious", energy=0.8)  # heuristic -> instinctive
    assert len(c.switch_log) == 1
    entry = c.switch_log[0]
    assert entry["from"] == "heuristic"
    assert entry["to"] == "instinctive"
    assert entry["trigger_mood"] == "anxious"


def test_no_log_when_mode_unchanged():
    c = ConsciousnessState()
    c.update_mode("neutral", energy=0.8)  # stays heuristic
    assert len(c.switch_log) == 0


def test_mode_durations_accumulate():
    c = ConsciousnessState()
    c.update_mode("anxious", energy=0.8)
    c.update_mode("neutral", energy=0.8)
    # heuristic duration should have accumulated a non-negative value
    assert c.mode_durations["heuristic"] >= 0
    assert c.mode_durations["instinctive"] >= 0


def test_switch_log_truncates_at_200():
    c = ConsciousnessState()
    # Alternate modes to force many switches.
    for i in range(260):
        if i % 2 == 0:
            c.update_mode("anxious", energy=0.8)   # instinctive
        else:
            c.update_mode("neutral", energy=0.8)   # heuristic
    assert len(c.switch_log) <= 200


def test_status_shape():
    c = ConsciousnessState()
    c.update_mode("anxious", energy=0.8)
    s = c.status()
    assert s["mode"] == "instinctive"
    assert set(s["mode_durations"].keys()) == {"survival", "instinctive", "heuristic", "reflective"}
    assert isinstance(s["recent_switches"], list)
    assert s["total_switches"] == len(c.switch_log)
