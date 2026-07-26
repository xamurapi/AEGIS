"""Exact-value tests for the deterministic behaviour modules.

`emotions`, `health_monitor` and `meta_regulation` are pure functions of their
inputs (the project's "zero randomness" guarantee), yet mutation testing showed
their suites asserted only coarse properties: half the arithmetic could be
inverted (`+`→`-`, `*`→`/`) and every test still passed.

These tests pin the CONSTANTS and the STATE MACHINE, so a change to any blend
factor, threshold or transition is a test failure rather than a silent drift in
how the system feels and regulates itself.
"""
import pytest

from aegis.layers.emotions import EmotionalSystem
from aegis.layers.health_monitor import HealthMonitor
from aegis.layers.meta_regulation import MetaRegulator


# ══════════════════════ emotions: exact VAD arithmetic ═══════════════

def test_success_rate_is_a_90_10_blend_of_reward():
    em = EmotionalSystem()                      # success_rate starts at 0.5
    em.update(reward=1.0)
    assert em.success_rate == pytest.approx(0.9 * 0.5 + 0.1 * 1.0)
    em.update(reward=0.0)
    assert em.success_rate == pytest.approx(0.9 * 0.55 + 0.1 * 0.0)


def test_every_update_costs_a_fixed_amount_of_energy():
    em = EmotionalSystem()                      # energy starts at 1.0
    em.update(reward=0.5)
    assert em.energy == pytest.approx(0.995)
    em.update(reward=0.5)
    assert em.energy == pytest.approx(0.990)


def test_energy_never_goes_negative():
    em = EmotionalSystem()
    em.energy = 0.002
    em.update(reward=0.5)
    assert em.energy == 0.0


def test_valence_is_a_70_30_blend_toward_success_rate():
    em = EmotionalSystem()
    em.update(reward=1.0)
    expected_sr = 0.9 * 0.5 + 0.1 * 1.0
    assert em.valence == pytest.approx(0.7 * 0.5 + 0.3 * expected_sr)


def test_dominance_blends_reward_scaled_by_energy():
    em = EmotionalSystem()
    em.update(reward=1.0)
    # energy is decremented BEFORE dominance is updated.
    assert em.dominance == pytest.approx(0.9 * 0.5 + 0.1 * (1.0 * 0.995))


@pytest.mark.parametrize("flag,delta", [
    ("unexpected", +0.15),
    ("repetitive", -0.08),
    ("error", +0.10),
    ("new_knowledge", +0.05),
])
def test_each_context_flag_moves_arousal_by_its_own_step(flag, delta):
    em = EmotionalSystem()
    before = em.arousal
    em.update(reward=0.5, context={flag: True})
    # _regulate() only kicks in outside [0.1, 0.9]; these steps stay inside.
    assert em.arousal == pytest.approx(before + delta)


def test_arousal_decays_toward_baseline_without_context():
    em = EmotionalSystem()
    em.arousal = 0.8
    em.update(reward=0.5)
    assert em.arousal == pytest.approx(0.95 * 0.8 + 0.05 * 0.5)


def test_context_flags_replace_decay_rather_than_add_to_it():
    """With a context present the baseline decay must NOT also run."""
    em = EmotionalSystem()
    em.arousal = 0.8
    em.update(reward=0.5, context={"error": True})
    assert em.arousal == pytest.approx(0.9)


def test_certainty_falls_as_reward_variance_rises():
    steady, noisy = EmotionalSystem(), EmotionalSystem()
    for _ in range(6):
        steady.update(reward=0.5)
    for r in (0.0, 1.0, 0.0, 1.0, 0.0, 1.0):
        noisy.update(reward=r)
    assert steady.certainty == pytest.approx(1.0)
    assert noisy.certainty < steady.certainty, "variance did not reduce certainty"


def test_certainty_equals_one_minus_twice_the_reward_variance():
    """Exact value — the history used is the one BEFORE the current update."""
    em = EmotionalSystem()
    for r in (0.0, 1.0, 0.0):
        em.update(reward=r)
    em.update(reward=0.5)          # certainty computed over [0.0, 1.0, 0.0]
    mean = 1 / 3
    variance = ((0 - mean) ** 2 + (1 - mean) ** 2 + (0 - mean) ** 2) / 3
    assert em.certainty == pytest.approx(1.0 - min(variance * 2, 1.0))


def test_certainty_needs_more_than_two_samples():
    em = EmotionalSystem()
    for r in (0.0, 1.0):
        em.update(reward=r)
    assert em.certainty == 0.5, "certainty moved before enough history existed"


def test_mood_duration_counts_consecutive_identical_moods():
    em = EmotionalSystem()
    em.update(reward=0.5)
    first, first_duration = em.mood, em.mood_duration
    em.update(reward=0.5)
    if em.mood == first:
        assert em.mood_duration == first_duration + 1, "duration must increment by one"
    em.mood = "definitely-different"
    em.update(reward=0.5)
    assert em.mood_duration == 0, "duration must reset when the mood changes"


def test_emotional_memory_is_recorded_only_on_a_transition_with_context():
    em = EmotionalSystem()
    em.mood = "sadness"
    em.update(reward=1.0, context={"unexpected": True})
    assert em.emotional_memories, "a mood transition with context was not recorded"
    assert all(k.startswith("sadness->") for k in em.emotional_memories)


def test_no_emotional_memory_without_a_transition():
    em = EmotionalSystem()
    em.update(reward=0.5, context={"unexpected": True})
    em.emotional_memories.clear()
    steady = em.mood
    em.mood = steady
    em.update(reward=0.5, context={"repetitive": True})
    if em.mood == steady:
        assert em.emotional_memories == {}


def test_emotional_modifier_scales_with_energy_and_certainty():
    em = EmotionalSystem()
    em.mood, em.energy, em.certainty = "neutral", 1.0, 1.0
    assert em.emotional_modifier() == pytest.approx(1.0 * (0.5 + 0.5) * (0.8 + 0.2))
    em.energy, em.certainty = 0.0, 0.0
    assert em.emotional_modifier() == pytest.approx(1.0 * 0.5 * 0.8)


def test_emotional_modifier_uses_the_mood_weight():
    em = EmotionalSystem()
    em.energy, em.certainty = 1.0, 1.0
    em.mood = "joy"
    assert em.emotional_modifier() == pytest.approx(1.2)
    em.mood = "shame"
    assert em.emotional_modifier() == pytest.approx(0.7)


def test_prolonged_mood_pulls_state_back_toward_baseline():
    em = EmotionalSystem()
    em.mood_duration = 25
    em.arousal, em.valence = 0.8, 0.8
    em._regulate()
    assert em.arousal == pytest.approx(0.97 * 0.8 + 0.03 * 0.5)
    assert em.valence == pytest.approx(0.97 * 0.8 + 0.03 * 0.5)


def test_extreme_arousal_is_damped_and_floor_is_lifted():
    em = EmotionalSystem()
    em.mood, em.arousal = "neutral", 0.95
    em._regulate()
    assert em.arousal == pytest.approx(0.95 * 0.95)

    em2 = EmotionalSystem()
    em2.mood, em2.arousal = "neutral", 0.05
    em2._regulate()
    assert em2.arousal == pytest.approx(0.10)


def test_low_energy_caps_arousal_and_damps_dominance():
    em = EmotionalSystem()
    em.mood, em.energy, em.arousal, em.dominance = "neutral", 0.1, 0.8, 0.8
    em._regulate()
    assert em.arousal == 0.6
    assert em.dominance == pytest.approx(0.8 * 0.95)


def test_anxious_state_recovers_when_success_rate_is_decent():
    em = EmotionalSystem()
    em.mood, em.success_rate, em.valence, em.arousal = "anxious", 0.5, 0.3, 0.7
    em._regulate()
    assert em.mood == "recovering"
    assert em.valence == pytest.approx(0.35)
    assert em.arousal == pytest.approx(0.675)


def test_mixed_emotions_are_ranked_by_closeness():
    em = EmotionalSystem()
    em.valence, em.arousal, em.dominance = 0.5, 0.5, 0.5   # exactly "neutral"
    mixed = em._determine_mixed()
    assert mixed[0][0] == "neutral"
    assert mixed[0][1] == 1.0, "an exact match must score 1.0"
    assert len(mixed) <= 3
    assert [m[1] for m in mixed] == sorted([m[1] for m in mixed], reverse=True)


def test_colour_channels_follow_valence_dominance_and_arousal():
    em = EmotionalSystem()
    em.valence, em.dominance, em.arousal = 1.0, 1.0, 1.0
    # r = 255*(1-1)=0, g = 255*1=255, b = 255*(0.5+0.5)=255, brightness = 1.0
    assert em.get_color().lower() == "#00ffff"

    em.valence, em.dominance, em.arousal = 0.0, 0.0, 0.0
    # r = 255, g = 0, b = 127; brightness = 0.3
    assert em.get_color().lower() == "#4c0026"


def test_colour_brightness_scales_a_non_zero_green_channel():
    """Brightness must MULTIPLY each channel — with valence 0.5 the green
    channel is non-zero, so a divide is observable (it is not at g = 0)."""
    em = EmotionalSystem()
    em.valence, em.dominance, em.arousal = 0.5, 0.5, 0.5
    brightness = 0.3 + 0.7 * 0.5
    expected_g = int(int(255 * 0.5) * brightness)
    assert em.get_color()[3:5].lower() == f"{expected_g:02x}"


def test_mixed_emotion_score_falls_off_linearly_with_distance():
    """An exact match scores 1.0 either way, so pin a state at a known
    non-zero distance from "neutral"."""
    em = EmotionalSystem()
    em.valence, em.arousal, em.dominance = 0.55, 0.5, 0.5   # distance 0.05
    scores = dict(em._determine_mixed())
    assert scores["neutral"] == pytest.approx(round(1.0 - 0.05 / 0.3, 2))


def test_colour_is_clamped_for_out_of_range_state():
    em = EmotionalSystem()
    em.valence, em.dominance, em.arousal = 1.5, -1.0, 1.0
    colour = em.get_color()
    assert colour.startswith("#") and len(colour) == 7
    assert all(c in "0123456789abcdefABCDEF" for c in colour[1:])


# ══════════════════════ health_monitor ═══════════════════════════════

def test_success_rate_is_a_percentage_of_all_ticks():
    hm = HealthMonitor()
    for _ in range(3):
        hm.record_tick(10.0, success=True)
    hm.record_tick(10.0, success=False)
    assert hm.status()["success_rate"] == pytest.approx(75.0)


def test_success_rate_is_zero_safe_with_no_ticks():
    assert HealthMonitor().status()["success_rate"] == pytest.approx(0.0)


def test_consecutive_errors_reset_on_success():
    hm = HealthMonitor()
    hm.record_tick(5.0, success=False)
    hm.record_tick(5.0, success=False)
    assert hm.consecutive_errors == 2
    hm.record_tick(5.0, success=True)
    assert hm.consecutive_errors == 0


@pytest.mark.parametrize("failures,expected", [
    (0, "healthy"),
    (1, "warning"),
    (4, "warning"),
    (5, "critical"),
])
def test_health_status_thresholds(failures, expected):
    hm = HealthMonitor()
    for _ in range(failures):
        hm.record_tick(5.0, success=False)
    assert hm.status()["health_status"] == expected


def test_averages_are_means_not_products():
    hm = HealthMonitor()
    hm.cpu_history.extend([10.0, 20.0, 60.0])
    hm.mem_history.extend([40.0, 60.0])
    st = hm.status()
    assert st["cpu_avg"] == pytest.approx(30.0)
    assert st["mem_avg"] == pytest.approx(50.0)


def test_averages_are_zero_without_samples():
    st = HealthMonitor().status()
    assert st["cpu_avg"] == 0 and st["mem_avg"] == 0


def test_total_ticks_is_the_sum_of_both_outcomes():
    hm = HealthMonitor()
    hm.record_tick(1.0, success=True)
    hm.record_tick(1.0, success=False)
    hm.record_tick(1.0, success=False)
    assert hm.status()["total_ticks"] == 3
    assert hm.error_count == 2


def test_uptime_is_the_elapsed_time_since_start():
    hm = HealthMonitor()
    hm.start_time -= 60.0
    assert hm.status()["uptime_seconds"] == pytest.approx(60.0, abs=5.0)


def test_record_tick_counts_as_a_success_by_default():
    hm = HealthMonitor()
    hm.record_tick(10.0)                      # no `success=` argument
    assert hm.successful_ticks == 1 and hm.failed_ticks == 0
    assert hm.consecutive_errors == 0


def test_average_tick_duration_is_a_mean():
    hm = HealthMonitor()
    for ms in (100.0, 200.0, 300.0):
        hm.record_tick(ms)
    assert hm.check()["metrics"]["avg_tick_ms"] == pytest.approx(200.0)


def test_psutil_flag_matches_whether_psutil_is_actually_importable():
    """Compare against the environment, not against the module's own constant —
    the latter is a tautology that any flipped flag would satisfy."""
    import importlib.util
    from aegis.layers import health_monitor as hmod

    available = importlib.util.find_spec("psutil") is not None
    assert hmod.HAS_PSUTIL is available
    assert HealthMonitor().status()["has_psutil"] is available


def test_psutil_flag_is_false_when_the_import_fails():
    """Exercise the no-psutil fallback itself.

    On a machine WITH psutil the `except ImportError` branch never runs, so the
    degraded path this system relies on when deployed in a slim container was
    never executed by any test. Re-import the module with psutil blocked.
    """
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil blocked for this test")
        return real_import(name, *args, **kwargs)

    saved_psutil = sys.modules.pop("psutil", None)
    builtins.__import__ = blocking_import
    try:
        degraded = importlib.reload(sys.modules["aegis.layers.health_monitor"])
        assert degraded.HAS_PSUTIL is False
        # The monitor must still work, just without resource metrics.
        report = degraded.HealthMonitor().check()
        assert report["status"] == "healthy"
        assert "cpu" not in report["metrics"]
    finally:
        builtins.__import__ = real_import
        if saved_psutil is not None:
            sys.modules["psutil"] = saved_psutil
        importlib.reload(sys.modules["aegis.layers.health_monitor"])


def test_available_memory_is_reported_in_megabytes():
    from aegis.layers import health_monitor as hmod

    if not hmod.HAS_PSUTIL:
        pytest.skip("psutil not installed")
    mb = HealthMonitor().check()["metrics"]["memory_available_mb"]
    # Bytes converted to MB — a machine with more than 10 TB free RAM does not
    # exist, so a multiplication instead of a division is caught here.
    assert 0 < mb < 10_000_000


def test_cpu_metrics_are_collected_when_psutil_is_available():
    from aegis.layers import health_monitor as hmod

    if not hmod.HAS_PSUTIL:
        pytest.skip("psutil not installed")
    metrics = HealthMonitor().check()["metrics"]
    assert "cpu" in metrics and "memory_percent" in metrics


def test_recovery_is_counted_when_returning_to_healthy():
    hm = HealthMonitor()
    hm._prev_status = "critical"
    report = hm.check()
    if report["status"] == "healthy":
        assert hm.recovery_count == 1
        # A second healthy check must not keep incrementing.
        hm.check()
        assert hm.recovery_count == 1


# ══════════════════════ meta_regulation: the state machine ═══════════

def _regulate(reg, energy=0.9, health="healthy", errors=0, mode="normal"):
    return reg.regulate(energy, health, errors, mode)


def test_normal_mode_enables_everything():
    reg = MetaRegulator()
    out = _regulate(reg, energy=0.9)
    assert out["mode"] == "normal"
    assert out["directives"] == {
        "skip_llm": False, "skip_dreams": False, "skip_learning": False,
        "reduce_sensors": False, "force_recharge": 0.0,
    }


@pytest.mark.parametrize("energy,health,errors", [
    (0.05, "healthy", 0),      # energy below the emergency floor
    (0.9, "critical", 0),      # health critical
    (0.9, "healthy", 5),       # too many consecutive errors
])
def test_each_emergency_trigger_fires_independently(energy, health, errors):
    reg = MetaRegulator()
    out = _regulate(reg, energy=energy, health=health, errors=errors)
    assert out["mode"] == "emergency"
    assert out["directives"]["skip_llm"] is True
    assert out["directives"]["skip_dreams"] is True
    assert out["directives"]["skip_learning"] is True
    assert out["directives"]["reduce_sensors"] is True
    assert out["directives"]["force_recharge"] == pytest.approx(0.15)
    assert reg.emergency_activations == 1


def test_four_errors_is_not_yet_an_emergency():
    reg = MetaRegulator()
    assert _regulate(reg, energy=0.9, errors=4)["mode"] != "emergency"


def test_eco_mode_saves_energy_but_keeps_learning():
    reg = MetaRegulator()
    out = _regulate(reg, energy=0.15)
    assert out["mode"] == "eco"
    assert out["directives"]["skip_llm"] is True
    assert out["directives"]["skip_dreams"] is True
    assert out["directives"]["reduce_sensors"] is True
    assert out["directives"]["skip_learning"] is False, "eco must not stop learning"
    assert out["directives"]["force_recharge"] == 0.0
    assert reg.savings_applied == 1 and reg.tick_skip_counter == 1


def test_eco_has_hysteresis_and_does_not_flip_back_immediately():
    """Between 0.2 and 0.35 an eco system STAYS in eco; a normal one does not
    enter it."""
    eco = MetaRegulator()
    _regulate(eco, energy=0.15)                 # enter eco
    assert eco.regulate(0.30, "healthy", 0, "normal")["mode"] == "eco"

    fresh = MetaRegulator()
    assert fresh.regulate(0.30, "healthy", 0, "normal")["mode"] == "normal"


def test_recovery_is_entered_from_emergency_only():
    reg = MetaRegulator()
    _regulate(reg, energy=0.05)                 # emergency
    out = reg.regulate(0.4, "healthy", 0, "normal")
    assert out["mode"] == "recovery"
    assert out["directives"]["skip_llm"] is True
    assert out["directives"]["force_recharge"] == pytest.approx(0.05)
    assert out["directives"]["skip_learning"] is False


def test_recovery_needs_energy_above_its_floor():
    reg = MetaRegulator()
    _regulate(reg, energy=0.05)                 # emergency
    assert reg.regulate(0.25, "healthy", 0, "normal")["mode"] == "emergency"


def test_recovery_returns_to_normal_only_above_its_own_threshold():
    reg = MetaRegulator()
    _regulate(reg, energy=0.05)
    reg.regulate(0.4, "healthy", 0, "normal")   # -> recovery
    assert reg.regulate(0.45, "healthy", 0, "normal")["mode"] == "recovery"
    assert reg.regulate(0.7, "healthy", 0, "normal")["mode"] == "normal"


def test_normal_requires_low_errors_and_non_critical_health():
    reg = MetaRegulator()
    reg.mode = "eco"
    assert reg.regulate(0.9, "healthy", 2, "normal")["mode"] != "normal"
    assert reg.regulate(0.9, "healthy", 1, "normal")["mode"] == "normal"


def test_only_real_transitions_are_recorded():
    reg = MetaRegulator()
    _regulate(reg, energy=0.9)                  # normal -> normal, no record
    assert len(reg.mode_history) == 0
    _regulate(reg, energy=0.15)                 # normal -> eco
    assert len(reg.mode_history) == 1
    _regulate(reg, energy=0.15)                 # eco -> eco, no new record
    assert len(reg.mode_history) == 1
    st = reg.status()
    assert st["mode_transitions"] == 1
    assert st["recent_transitions"][0] == {"from": "normal", "to": "eco", "energy": 0.15}
