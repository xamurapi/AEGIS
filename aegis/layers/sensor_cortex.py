"""SensorCortex — real and deterministic sensor input system.

All readings come from actual system data (psutil) or deterministic
time-based functions. No random values.
"""
import time
import math
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class SensorCortex:
    """Provides real and deterministic sensor data for the perception cycle."""

    def __init__(self):
        self.sensors: dict[str, dict] = {
            "cpu_load": {"type": "real", "enabled": True, "value": 0.0, "unit": "%"},
            "memory_usage": {"type": "real", "enabled": True, "value": 0.0, "unit": "%"},
            "temperature": {"type": "derived", "enabled": True, "value": 22.0, "unit": "C"},
            "light_level": {"type": "derived", "enabled": True, "value": 0.7, "unit": "norm"},
            "noise_level": {"type": "derived", "enabled": True, "value": 0.3, "unit": "norm"},
            "vibration": {"type": "derived", "enabled": True, "value": 0.0, "unit": "norm"},
            "time_of_day": {"type": "real", "enabled": True, "value": 0.0, "unit": "hour"},
            "system_uptime": {"type": "real", "enabled": True, "value": 0.0, "unit": "sec"},
        }
        self.readings: deque = deque(maxlen=100)
        self.heavy_sensors_disabled = False
        self._start_time = time.time()

    def read_all(self) -> dict[str, float]:
        """Read all enabled sensors and return their values."""
        values = {}

        for name, sensor in self.sensors.items():
            if not sensor["enabled"]:
                continue

            if sensor["type"] == "real":
                sensor["value"] = self._read_real(name)
            else:
                sensor["value"] = self._read_derived(name)

            values[name] = round(sensor["value"], 3)

        self.readings.append({"time": time.time(), "values": values})
        return values

    def _read_real(self, name: str) -> float:
        if name == "cpu_load" and HAS_PSUTIL:
            return psutil.cpu_percent(interval=0)
        elif name == "memory_usage" and HAS_PSUTIL:
            return psutil.virtual_memory().percent
        elif name == "time_of_day":
            return time.localtime().tm_hour + time.localtime().tm_min / 60
        elif name == "system_uptime":
            return time.time() - self._start_time
        return 0.0

    def _read_derived(self, name: str) -> float:
        """Deterministic sensor readings derived from time (no random)."""
        t = time.time()

        if name == "temperature":
            # Smooth oscillation: 22C base + sine wave (period 10 min)
            return 22.0 + math.sin(t / 300) * 3

        elif name == "light_level":
            hour = time.localtime().tm_hour
            if 6 <= hour <= 20:
                return min(1.0, 0.5 + math.sin((hour - 6) / 14 * math.pi) * 0.5)
            return 0.1  # night baseline

        elif name == "noise_level":
            # Derives from hour: noisier during day
            hour = time.localtime().tm_hour
            if 8 <= hour <= 22:
                return 0.2 + 0.2 * math.sin((hour - 8) / 14 * math.pi)
            return 0.1

        elif name == "vibration":
            # Derives from cpu load if available
            if HAS_PSUTIL:
                return min(0.15, psutil.cpu_percent(interval=0) / 1000)
            return 0.02 + 0.01 * math.sin(t / 60)

        return 0.5

    def disable_heavy_sensors(self):
        """Disable expensive sensors to save resources."""
        self.heavy_sensors_disabled = True
        for name in ("vibration", "noise_level"):
            if name in self.sensors:
                self.sensors[name]["enabled"] = False

    def enable_all_sensors(self):
        """Re-enable all sensors."""
        self.heavy_sensors_disabled = False
        for sensor in self.sensors.values():
            sensor["enabled"] = True

    def reduce_sensors(self, reduce: bool):
        """Toggle heavy sensor reduction."""
        if reduce:
            self.disable_heavy_sensors()
        else:
            self.enable_all_sensors()

    def status(self) -> dict:
        return {
            "sensors_count": len(self.sensors),
            "enabled_count": sum(1 for s in self.sensors.values() if s["enabled"]),
            "heavy_disabled": self.heavy_sensors_disabled,
            "readings_count": len(self.readings),
            "current_values": {
                name: {"value": round(s["value"], 2), "unit": s["unit"], "type": s["type"]}
                for name, s in self.sensors.items() if s["enabled"]
            },
        }
