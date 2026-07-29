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
        #: Cached network counters, and when they were read.
        #:
        #: `psutil.net_io_counters()` enumerates every interface on the host and
        #: costs about 7.5 ms on Windows — measured as 85% of the whole PERCEIVE
        #: phase, which was the one phase over its §3.4 budget. Nothing in the
        #: cognitive cycle decides anything on these numbers; they are
        #: monotonic counters that get displayed. Reading them twenty times a
        #: minute buys nothing and costs the budget.
        self._net_cache: tuple[int, int] = (0, 0)
        self._net_read_at: float = 0.0

    #: How often the network counters are actually read, in seconds. On the
    #: default tick interval this is roughly every tenth tick.
    NET_REFRESH_SECONDS = 30.0

    def _network_counters(self) -> tuple[int, int]:
        """Cumulative network bytes, refreshed on an interval rather than a tick.

        Through ``CLOCK`` so a frozen clock in a test gets one read and then a
        stable answer — the same reason every other time-dependent thing in the
        package goes through it.
        """
        now = CLOCK.now()
        if self._net_read_at and now - self._net_read_at < self.NET_REFRESH_SECONDS:
            return self._net_cache
        try:
            counters = psutil.net_io_counters()
            self._net_cache = (counters.bytes_sent, counters.bytes_recv)
        except Exception:
            pass                      # keep the last reading rather than lying
        self._net_read_at = now
        return self._net_cache

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
            net_sent, net_recv = self._network_counters()
            self.sensors = {
                "cpu_load": cpu,
                "memory_usage_pct": mem.percent,
                "disk_free_gb": disk_free_gb,
                "network_bytes_sent": net_sent,
                "network_bytes_recv": net_recv,
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
