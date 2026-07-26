"""Tests for the emotional (VAD) system."""
from aegis.layers.emotions import EmotionalSystem, EMOTION_MAP, MOOD_MODIFIERS


def test_initial_state():
    es = EmotionalSystem()
    assert es.mood == "neutral"
    assert es.energy == 1.0
    assert es.valence == 0.5


def test_update_no_context_decays_arousal_toward_baseline():
    es = EmotionalSystem()
    es.arousal = 0.9
    es.update(0.5)
    # 0.95 * 0.9 + 0.05 * 0.5 = 0.88
    assert 0.87 < es.arousal < 0.89
    # energy decays by 0.005 each tick
    assert es.energy < 1.0


def test_update_context_flags_raise_arousal():
    es = EmotionalSystem()
    es.arousal = 0.5
    es.update(0.8, {"unexpected": True, "error": True, "new_knowledge": True})
    assert es.arousal > 0.5


def test_update_repetitive_lowers_arousal():
    es = EmotionalSystem()
    es.arousal = 0.5
    es.update(0.5, {"repetitive": True})
    assert es.arousal < 0.5


def test_certainty_computed_from_reward_variance():
    es = EmotionalSystem()
    # Fill history with consistent rewards -> low variance -> high certainty
    for _ in range(6):
        es.update(0.5)
    assert es.certainty > 0.5


def test_emotional_memory_recorded_on_mood_change_with_context():
    es = EmotionalSystem()
    # Drive strongly toward a happy mood so mood changes from neutral.
    for _ in range(5):
        es.update(1.0, {"unexpected": True})
    assert es.mood != "neutral"
    assert len(es.emotional_memories) >= 1


def test_regulate_anxious_recovers():
    es = EmotionalSystem()
    es.mood = "anxious"
    es.success_rate = 0.5
    es.valence, es.arousal = 0.3, 0.7
    es._regulate()
    assert es.mood == "recovering"


def test_regulate_prolonged_mood_pulls_to_baseline():
    es = EmotionalSystem()
    es.mood_duration = 25
    es.mood = "neutral"
    es.arousal = 0.9
    es.valence = 0.9
    es._regulate()
    assert es.arousal < 0.9
    assert es.valence < 0.9


def test_regulate_high_arousal_dampened():
    es = EmotionalSystem()
    es.mood = "neutral"
    es.arousal = 0.95
    es._regulate()
    assert es.arousal < 0.95


def test_regulate_low_arousal_raised():
    es = EmotionalSystem()
    es.mood = "neutral"
    es.arousal = 0.05
    es._regulate()
    assert es.arousal > 0.05


def test_regulate_low_energy_caps_arousal_and_dampens_dominance():
    es = EmotionalSystem()
    es.mood = "neutral"
    es.energy = 0.1
    es.arousal = 0.9
    es.dominance = 0.8
    es._regulate()
    assert es.arousal <= 0.6
    assert es.dominance < 0.8


def test_energy_drops_below_threshold_over_many_ticks():
    es = EmotionalSystem()
    for _ in range(200):
        es.update(0.5)
    assert es.energy < 0.2


def test_determine_mood_matches_known_emotion():
    es = EmotionalSystem()
    params = EMOTION_MAP["joy"]
    es.valence, es.arousal, es.dominance = params["valence"], params["arousal"], params["dominance"]
    assert es._determine_mood() == "joy"


def test_determine_mixed_returns_close_emotions():
    es = EmotionalSystem()
    es.valence, es.arousal, es.dominance = 0.5, 0.5, 0.5
    mixed = es._determine_mixed()
    assert mixed
    assert mixed[0][0] == "neutral"
    assert len(mixed) <= 3


def test_emotional_modifier_uses_mood():
    es = EmotionalSystem()
    es.mood = "joy"
    assert es.emotional_modifier() > 0
    assert MOOD_MODIFIERS["joy"] == 1.2


def test_emotional_modifier_unknown_mood_defaults():
    es = EmotionalSystem()
    es.mood = "not_a_real_mood"
    es.energy = 1.0
    es.certainty = 1.0
    # base defaults to 1.0
    assert abs(es.emotional_modifier() - 1.0) < 1e-9


def test_get_color_valid_hex():
    es = EmotionalSystem()
    color = es.get_color()
    assert color.startswith("#")
    assert len(color) == 7
    int(color[1:], 16)  # parses as hex


def test_get_color_clamps_out_of_range_values():
    es = EmotionalSystem()
    es.valence = -0.5  # would produce negative green channel
    es.dominance = 2.0
    color = es.get_color()
    assert len(color) == 7
    int(color[1:], 16)


def test_recharge_caps_at_one():
    es = EmotionalSystem()
    es.energy = 0.9
    es.recharge(0.5)
    assert es.energy == 1.0


def test_status_shape():
    es = EmotionalSystem()
    es.update(0.5)
    s = es.status()
    for key in ("mood", "energy", "valence", "arousal", "dominance", "certainty",
                "mood_duration", "mixed_emotions", "color", "modifier", "success_rate"):
        assert key in s
    assert s["color"].startswith("#")


def test_history_capped():
    es = EmotionalSystem()
    for _ in range(150):
        es.update(0.5)
    assert len(es.history) <= 100


def test_mood_duration_increments_when_stable():
    es = EmotionalSystem()
    es.update(0.5)
    first = es.mood
    es.update(0.5)
    if es.mood == first:
        assert es.mood_duration >= 1
