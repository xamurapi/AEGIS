"""Tests for MetaRegulator."""
from aegis.layers.meta_regulation import MetaRegulator


def test_initial_state():
    r = MetaRegulator()
    assert r.mode == "normal"
    assert r.emergency_activations == 0


def test_normal_mode_high_energy():
    r = MetaRegulator()
    out = r.regulate(0.9, "healthy", 0, "heuristic")
    assert out["mode"] == "normal"
    assert out["directives"]["skip_llm"] is False


def test_emergency_low_energy():
    r = MetaRegulator()
    out = r.regulate(0.05, "healthy", 0, "heuristic")
    assert out["mode"] == "emergency"
    assert r.emergency_activations == 1
    d = out["directives"]
    assert d["skip_llm"] and d["skip_learning"] and d["reduce_sensors"]
    assert d["force_recharge"] == 0.15


def test_emergency_via_health_critical():
    r = MetaRegulator()
    out = r.regulate(0.9, "critical", 0, "heuristic")
    assert out["mode"] == "emergency"


def test_emergency_via_consecutive_errors():
    r = MetaRegulator()
    out = r.regulate(0.9, "healthy", 5, "heuristic")
    assert out["mode"] == "emergency"


def test_eco_mode():
    r = MetaRegulator()
    out = r.regulate(0.15, "healthy", 0, "heuristic")
    assert out["mode"] == "eco"
    d = out["directives"]
    assert d["skip_llm"] and d["skip_dreams"] and d["reduce_sensors"]
    assert r.savings_applied == 1
    assert r.tick_skip_counter == 1


def test_eco_is_sticky_between_thresholds():
    r = MetaRegulator()
    r.regulate(0.15, "healthy", 0, "heuristic")  # -> eco
    out = r.regulate(0.3, "healthy", 0, "heuristic")  # 0.2<=e<0.35, stays eco
    assert out["mode"] == "eco"


def test_recovery_from_emergency():
    r = MetaRegulator()
    r.regulate(0.05, "healthy", 0, "heuristic")  # -> emergency
    out = r.regulate(0.4, "healthy", 0, "heuristic")  # emergency & e>0.3 -> recovery
    assert out["mode"] == "recovery"
    d = out["directives"]
    assert d["skip_llm"] is True
    assert d["force_recharge"] == 0.05


def test_recovery_to_normal():
    r = MetaRegulator()
    r.regulate(0.05, "healthy", 0, "heuristic")  # emergency
    r.regulate(0.4, "healthy", 0, "heuristic")   # recovery
    out = r.regulate(0.65, "healthy", 0, "heuristic")  # recovery & e>0.6 -> normal
    assert out["mode"] == "normal"


def test_mode_transitions_logged():
    r = MetaRegulator()
    r.regulate(0.9, "healthy", 0, "heuristic")  # normal (no change)
    r.regulate(0.05, "healthy", 0, "heuristic")  # -> emergency (change)
    # one transition recorded
    assert len(r.mode_history) == 1
    assert r.mode_history[0]["to"] == "emergency"


def test_no_transition_when_mode_unchanged():
    r = MetaRegulator()
    r.regulate(0.9, "healthy", 0, "heuristic")
    r.regulate(0.85, "healthy", 0, "heuristic")
    assert len(r.mode_history) == 0


def test_status_reports_transitions():
    r = MetaRegulator()
    r.regulate(0.05, "healthy", 0, "heuristic")  # normal->emergency
    r.regulate(0.4, "healthy", 0, "heuristic")   # emergency->recovery
    st = r.status()
    assert st["mode"] == "recovery"
    assert st["emergency_activations"] == 1
    assert st["mode_transitions"] == 2
    assert len(st["recent_transitions"]) == 2
    assert st["recent_transitions"][0]["from"] == "normal"
