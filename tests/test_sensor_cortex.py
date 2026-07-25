"""Tests for the SensorCortex."""
import time
import aegis.layers.sensor_cortex as sc_mod
from aegis.layers.sensor_cortex import SensorCortex


class _FakePsutil:
    def __init__(self, cpu=42.0, mem_percent=55.0):
        self._cpu = cpu
        self._mem_percent = mem_percent

    def cpu_percent(self, interval=0):
        return self._cpu

    def virtual_memory(self):
        class M:
            percent = self._mem_percent
        return M()


class _FakeTime:
    """localtime stub returning a fixed hour/min."""
    def __init__(self, hour, minute=0):
        self.hour = hour
        self.minute = minute

    def __call__(self):
        return time.struct_time((2026, 7, 25, self.hour, self.minute, 0, 4, 206, 0))


def test_read_all_returns_enabled_values():
    s = SensorCortex()
    values = s.read_all()
    # all 8 sensors enabled by default
    assert len(values) == 8
    assert "cpu_load" in values
    assert len(s.readings) == 1


def test_read_all_skips_disabled():
    s = SensorCortex()
    s.disable_heavy_sensors()
    values = s.read_all()
    assert "vibration" not in values
    assert "noise_level" not in values


def test_read_real_with_psutil(monkeypatch):
    monkeypatch.setattr(sc_mod, "HAS_PSUTIL", True)
    monkeypatch.setattr(sc_mod, "psutil", _FakePsutil(cpu=42.0, mem_percent=55.0), raising=False)
    s = SensorCortex()
    assert s._read_real("cpu_load") == 42.0
    assert s._read_real("memory_usage") == 55.0


def test_read_real_time_of_day_and_uptime():
    s = SensorCortex()
    tod = s._read_real("time_of_day")
    assert 0 <= tod < 24
    up = s._read_real("system_uptime")
    assert up >= 0


def test_read_real_unknown_returns_zero():
    s = SensorCortex()
    assert s._read_real("nonexistent") == 0.0


def test_read_real_without_psutil(monkeypatch):
    monkeypatch.setattr(sc_mod, "HAS_PSUTIL", False)
    s = SensorCortex()
    assert s._read_real("cpu_load") == 0.0
    assert s._read_real("memory_usage") == 0.0


def test_read_derived_temperature():
    s = SensorCortex()
    temp = s._read_derived("temperature")
    assert 19.0 <= temp <= 25.0


def test_read_derived_light_day(monkeypatch):
    monkeypatch.setattr(sc_mod.time, "localtime", _FakeTime(12))
    s = SensorCortex()
    light = s._read_derived("light_level")
    assert light > 0.1  # daytime brighter than night baseline


def test_read_derived_light_night(monkeypatch):
    monkeypatch.setattr(sc_mod.time, "localtime", _FakeTime(3))
    s = SensorCortex()
    assert s._read_derived("light_level") == 0.1


def test_read_derived_noise_day(monkeypatch):
    monkeypatch.setattr(sc_mod.time, "localtime", _FakeTime(12))
    s = SensorCortex()
    noise = s._read_derived("noise_level")
    assert noise >= 0.2


def test_read_derived_noise_night(monkeypatch):
    monkeypatch.setattr(sc_mod.time, "localtime", _FakeTime(3))
    s = SensorCortex()
    assert s._read_derived("noise_level") == 0.1


def test_read_derived_vibration_with_psutil(monkeypatch):
    monkeypatch.setattr(sc_mod, "HAS_PSUTIL", True)
    monkeypatch.setattr(sc_mod, "psutil", _FakePsutil(cpu=100.0), raising=False)
    s = SensorCortex()
    vib = s._read_derived("vibration")
    assert vib == min(0.15, 100.0 / 1000)


def test_read_derived_vibration_without_psutil(monkeypatch):
    monkeypatch.setattr(sc_mod, "HAS_PSUTIL", False)
    s = SensorCortex()
    vib = s._read_derived("vibration")
    assert 0.0 <= vib <= 0.04


def test_read_derived_unknown_returns_half():
    s = SensorCortex()
    assert s._read_derived("nonexistent") == 0.5


def test_disable_and_enable_sensors():
    s = SensorCortex()
    s.disable_heavy_sensors()
    assert s.heavy_sensors_disabled is True
    assert s.sensors["vibration"]["enabled"] is False
    s.enable_all_sensors()
    assert s.heavy_sensors_disabled is False
    assert all(v["enabled"] for v in s.sensors.values())


def test_reduce_sensors_toggle():
    s = SensorCortex()
    s.reduce_sensors(True)
    assert s.heavy_sensors_disabled is True
    s.reduce_sensors(False)
    assert s.heavy_sensors_disabled is False


def test_status_shape():
    s = SensorCortex()
    s.read_all()
    st = s.status()
    assert st["sensors_count"] == 8
    assert st["enabled_count"] == 8
    assert st["heavy_disabled"] is False
    assert st["readings_count"] == 1
    assert "cpu_load" in st["current_values"]


def test_status_reflects_disabled():
    s = SensorCortex()
    s.disable_heavy_sensors()
    st = s.status()
    assert st["enabled_count"] == 6
    assert "vibration" not in st["current_values"]


def test_readings_capped_at_100():
    s = SensorCortex()
    for _ in range(120):
        s.read_all()
    assert len(s.readings) <= 100
