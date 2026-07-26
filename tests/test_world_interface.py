"""Tests for WorldInterface."""
import time
import types

from aegis.layers import world_interface
from aegis.layers.world_interface import WorldInterface


def test_perceive_real_psutil():
    wi = WorldInterface()
    data = wi.perceive()
    for key in ("cpu_load", "memory_usage_pct", "disk_free_gb",
                "network_bytes_sent", "uptime_hours", "process_count"):
        assert key in data
    assert data["platform"]
    assert data["python_version"]
    assert "timestamp" in data


def test_perceive_fallback(monkeypatch):
    monkeypatch.setattr(world_interface, "HAS_PSUTIL", False)
    wi = WorldInterface()
    data = wi.perceive()
    assert data["disk_free_gb"] == 200.0
    assert data["network_bytes_sent"] == 0
    assert data["process_count"] == 0
    assert 20 <= data["cpu_load"] <= 30


def test_act_observe():
    wi = WorldInterface()
    rec = wi.act({"type": "observe"})
    assert rec["result"] == "success"
    assert "data" in rec


def test_act_log_message():
    wi = WorldInterface()
    rec = wi.act({"type": "log_message", "message": "hello"})
    assert rec["result"] == "success"
    assert rec["message"] == "hello"


def test_act_internal_computation():
    wi = WorldInterface()
    rec = wi.act({"type": "internal_computation"})
    assert rec["result"] == "success"


def test_act_unknown_no_permission_required():
    wi = WorldInterface()
    rec = wi.act({"type": "something"})
    assert rec["result"] == "success"


def test_act_permission_granted():
    wi = WorldInterface()
    rec = wi.act({"type": "write", "permission_required": "filesystem_write"})
    assert rec["result"] == "success"


def test_act_permission_denied():
    wi = WorldInterface()
    wi.revoke_permission("network_write")
    rec = wi.act({"type": "send", "permission_required": "network_write"})
    assert rec["result"] == "denied"
    assert "not granted" in rec["reason"]


def test_act_irreversible_flag():
    wi = WorldInterface()
    rec = wi.act({"type": "internal_computation", "irreversible": True})
    assert rec["irreversible"] is True


def test_actions_log_capped():
    wi = WorldInterface()
    for _ in range(320):
        wi.act({"type": "internal_computation"})
    assert len(wi.actions_log) == 300


def test_grant_permission_valid():
    wi = WorldInterface()
    wi.revoke_permission("external_api")
    assert wi.permissions["external_api"] is False
    wi.grant_permission("external_api")
    assert wi.permissions["external_api"] is True


def test_grant_revoke_invalid_permission_noop():
    wi = WorldInterface()
    wi.grant_permission("nope")
    wi.revoke_permission("nope")
    assert "nope" not in wi.permissions


def test_status():
    wi = WorldInterface()
    wi.perceive()
    wi.act({"type": "observe"})
    st = wi.status()
    assert st["actions_total"] == 1
    assert "permissions" in st
    assert len(st["recent_actions"]) == 1
    assert st["recent_actions"][0]["type"] == "observe"


def test_uptime_hours_is_real_process_uptime():
    # uptime must be measured from start_time, NOT (time.time() % 86400)/3600
    # which is the wall-clock time of day and resets each midnight.
    wi = WorldInterface()
    wi.start_time = time.time() - 7200  # pretend the process started 2h ago
    data = wi.perceive()
    assert data["uptime_hours"] >= 1.9  # ~2 hours


def test_uptime_hours_small_for_fresh_instance():
    wi = WorldInterface()
    data = wi.perceive()
    assert 0.0 <= data["uptime_hours"] < 0.01  # brand new -> near zero


def test_perceive_disk_usage_failure_degrades(monkeypatch):
    fake = types.SimpleNamespace(
        cpu_percent=lambda interval=0: 5.0,
        virtual_memory=lambda: types.SimpleNamespace(percent=30.0),
        disk_usage=lambda p: (_ for _ in ()).throw(OSError("no disk")),
        net_io_counters=lambda: types.SimpleNamespace(bytes_sent=1, bytes_recv=2),
        pids=lambda: [1, 2, 3],
    )
    monkeypatch.setattr(world_interface, "HAS_PSUTIL", True)
    monkeypatch.setattr(world_interface, "psutil", fake, raising=False)
    wi = WorldInterface()
    data = wi.perceive()  # must not raise despite disk_usage error
    assert data["disk_free_gb"] == 0.0
    assert data["cpu_load"] == 5.0
    assert data["process_count"] == 3
