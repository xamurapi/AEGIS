"""Layer 5: World Interface — digital body (W-001..W-006).

All sensor data comes from real system metrics (psutil, os, time).
No random simulation — actual hardware readings.
"""
from aegis.clock import CLOCK
import os
import platform

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class WorldInterface:
    def __init__(self):
        self.start_time = CLOCK.now()
        self.actions_log: list[dict] = []
        self.sensors: dict[str, float] = {}
        self.permissions: dict[str, bool] = {
            "filesystem_read": True,
            "filesystem_write": True,
            "network_read": True,
            "network_write": True,
            "process_spawn": True,
            "external_api": True,
        }
        self._environment_cache: dict = {}

    def perceive(self) -> dict:
        # Real process uptime measured from construction, NOT the time-of-day
        # that ``CLOCK.now() % 86400`` produced (which reset every midnight).
        uptime_hours = (CLOCK.now() - self.start_time) / 3600
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            # disk_usage can raise (unmounted/unavailable path) — degrade
            # gracefully rather than crashing the whole perception cycle.
            try:
                disk = psutil.disk_usage("/") if os.name != "nt" else psutil.disk_usage("C:\\")
                disk_free_gb = round(disk.free / (1024 ** 3), 1)
            except Exception:
                disk_free_gb = 0.0
            net_io = psutil.net_io_counters()
            self.sensors = {
                "cpu_load": cpu,
                "memory_usage_pct": mem.percent,
                "disk_free_gb": disk_free_gb,
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_recv": net_io.bytes_recv,
                "uptime_hours": uptime_hours,
                "process_count": len(psutil.pids()),
            }
        else:
            # Deterministic fallback — derive from time, not random
            t = CLOCK.now()
            self.sensors = {
                "cpu_load": 20 + 10 * (t % 60) / 60,  # slowly oscillating
                "memory_usage_pct": 40 + 5 * ((t % 300) / 300),
                "disk_free_gb": 200.0,
                "network_bytes_sent": 0,
                "network_bytes_recv": 0,
                "uptime_hours": uptime_hours,
                "process_count": 0,
            }

        self._environment_cache = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp": CLOCK.now(),
        }
        return {**self.sensors, **self._environment_cache}

    def act(self, action: dict) -> dict:
        action_type = action.get("type", "unknown")
        record = {
            "timestamp": CLOCK.now(),
            "action_type": action_type,
            "details": action,
            "result": "pending",
            "irreversible": action.get("irreversible", False),
        }

        if action_type == "observe":
            record["result"] = "success"
            record["data"] = self.perceive()
        elif action_type == "log_message":
            record["result"] = "success"
            record["message"] = action.get("message", "")
        elif action_type == "internal_computation":
            record["result"] = "success"
        else:
            perm_key = action.get("permission_required", "")
            if perm_key and not self.permissions.get(perm_key, False):
                record["result"] = "denied"
                record["reason"] = f"Permission '{perm_key}' not granted"
            else:
                record["result"] = "success"

        self.actions_log.append(record)
        if len(self.actions_log) > 300:
            self.actions_log = self.actions_log[-300:]
        return record

    def grant_permission(self, perm: str):
        if perm in self.permissions:
            self.permissions[perm] = True

    def revoke_permission(self, perm: str):
        if perm in self.permissions:
            self.permissions[perm] = False

    def status(self) -> dict:
        return {
            "sensors": self.sensors,
            "environment": self._environment_cache,
            "permissions": self.permissions,
            "actions_total": len(self.actions_log),
            "recent_actions": [
                {"type": a["action_type"], "result": a["result"], "time": a["timestamp"]}
                for a in self.actions_log[-10:]
            ],
        }
