"""Tests for the HealthMonitor."""
import aegis.layers.health_monitor as hm_mod
from aegis.layers.health_monitor import HealthMonitor


class _FakeMem:
    def __init__(self, percent, available):
        self.percent = percent
        self.available = available


class _FakePsutil:
    def __init__(self, cpu, mem_percent, mem_available=1024 * 1024 * 100):
        self._cpu = cpu
        self._mem = _FakeMem(mem_percent, mem_available)

    def cpu_percent(self, interval=0):
        return self._cpu

    def virtual_memory(self):
        return self._mem


def _use_fake_psutil(monkeypatch, cpu, mem):
    monkeypatch.setattr(hm_mod, "HAS_PSUTIL", True)
    monkeypatch.setattr(hm_mod, "psutil", _FakePsutil(cpu, mem), raising=False)


def test_check_healthy(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=10, mem=20)
    h = HealthMonitor()
    report = h.check()
    assert report["status"] == "healthy"
    assert report["metrics"]["cpu"] == 10
    assert report["metrics"]["memory_percent"] == 20
    assert "memory_available_mb" in report["metrics"]


def test_check_cpu_warning(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=80, mem=20)  # > 95*0.8 = 76
    h = HealthMonitor()
    report = h.check()
    assert report["status"] == "warning"
    assert any("CPU high" in w for w in report["warnings"])


def test_check_cpu_critical(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=99, mem=20)
    h = HealthMonitor()
    report = h.check()
    assert report["status"] == "critical"
    assert any("CPU critical" in c for c in report["critical"])
    assert len(h.incidents) == 1


def test_check_memory_warning(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=10, mem=80)  # > 90*0.8 = 72
    h = HealthMonitor()
    report = h.check()
    assert report["status"] == "warning"
    assert any("Memory high" in w for w in report["warnings"])


def test_check_memory_critical(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=10, mem=95)
    h = HealthMonitor()
    report = h.check()
    assert report["status"] == "critical"
    assert any("Memory critical" in c for c in report["critical"])


def test_check_without_psutil(monkeypatch):
    monkeypatch.setattr(hm_mod, "HAS_PSUTIL", False)
    h = HealthMonitor()
    report = h.check()
    assert "cpu" not in report["metrics"]
    assert report["status"] == "healthy"


def test_slow_ticks_warning(monkeypatch):
    monkeypatch.setattr(hm_mod, "HAS_PSUTIL", False)
    h = HealthMonitor()
    h.record_tick(6000, success=True)  # over 5000 ms threshold
    report = h.check()
    assert "avg_tick_ms" in report["metrics"]
    assert any("Slow ticks" in w for w in report["warnings"])


def test_consecutive_errors_critical(monkeypatch):
    monkeypatch.setattr(hm_mod, "HAS_PSUTIL", False)
    h = HealthMonitor()
    for _ in range(5):
        h.record_tick(10, success=False)
    report = h.check()
    assert report["status"] == "critical"
    assert any("Consecutive errors" in c for c in report["critical"])


def test_record_tick_success_resets_consecutive_errors():
    h = HealthMonitor()
    h.record_tick(10, success=False)
    assert h.consecutive_errors == 1
    h.record_tick(10, success=True)
    assert h.consecutive_errors == 0
    assert h.successful_ticks == 1
    assert h.failed_ticks == 1
    assert h.error_count == 1


def test_incidents_truncate_at_50(monkeypatch):
    monkeypatch.setattr(hm_mod, "HAS_PSUTIL", False)
    h = HealthMonitor()
    for _ in range(6):
        h.record_tick(10, success=False)  # consecutive_errors >= 5 -> critical
    for _ in range(60):
        h.check()
    assert len(h.incidents) <= 50


def test_warnings_deque_capped(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=80, mem=20)
    h = HealthMonitor()
    for _ in range(40):
        h.check()
    assert len(h.warnings) <= 30


def test_status_health_states():
    h = HealthMonitor()
    assert h.status()["health_status"] == "healthy"
    h.record_tick(10, success=False)
    assert h.status()["health_status"] == "warning"
    for _ in range(5):
        h.record_tick(10, success=False)
    assert h.status()["health_status"] == "critical"


def test_status_shape():
    h = HealthMonitor()
    h.record_tick(10, success=True)
    s = h.status()
    for key in ("uptime_seconds", "total_ticks", "success_rate", "error_count",
                "recovery_count", "consecutive_errors", "health_status",
                "cpu_avg", "mem_avg", "recent_incidents", "has_psutil"):
        assert key in s
    assert s["total_ticks"] == 1
    assert s["success_rate"] == 100.0


def test_status_averages_with_history(monkeypatch):
    _use_fake_psutil(monkeypatch, cpu=50, mem=40)
    h = HealthMonitor()
    h.check()
    s = h.status()
    assert s["cpu_avg"] == 50.0
    assert s["mem_avg"] == 40.0


def test_recovery_count_increments_on_recovery(monkeypatch):
    h = HealthMonitor()
    _use_fake_psutil(monkeypatch, cpu=99, mem=20)  # critical
    h.check()
    assert h.recovery_count == 0
    _use_fake_psutil(monkeypatch, cpu=10, mem=20)  # back to healthy
    h.check()
    assert h.recovery_count == 1
    h.check()  # staying healthy must not double-count
    assert h.recovery_count == 1


def test_recovery_count_from_warning(monkeypatch):
    h = HealthMonitor()
    _use_fake_psutil(monkeypatch, cpu=80, mem=20)  # warning (degraded)
    h.check()
    assert h.recovery_count == 0
    _use_fake_psutil(monkeypatch, cpu=10, mem=20)  # healthy
    h.check()
    assert h.recovery_count == 1
