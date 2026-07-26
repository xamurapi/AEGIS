"""Health Monitor — system resource monitoring and emergency prevention."""
from aegis.clock import CLOCK
import platform
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class HealthMonitor:
    def __init__(self):
        self.start_time = CLOCK.now()
        self.cpu_history: deque = deque(maxlen=60)
        self.mem_history: deque = deque(maxlen=60)
        self.tick_durations: deque = deque(maxlen=100)
        self.error_count = 0
        self.successful_ticks = 0
        self.failed_ticks = 0
        self.consecutive_errors = 0
        self.recovery_count = 0
        self._prev_status = "healthy"   # last check() status, for recovery detection
        self.incidents: list[dict] = []
        self.warnings: deque = deque(maxlen=30)

        self.thresholds = {
            "cpu": 95,
            "memory": 90,
            "tick_duration_ms": 5000,
            "consecutive_errors": 5,
        }

    def check(self) -> dict:
        report = {
            "status": "healthy",
            "warnings": [],
            "critical": [],
            "metrics": {},
        }

        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            self.cpu_history.append(cpu)
            self.mem_history.append(mem.percent)
            report["metrics"]["cpu"] = round(cpu, 1)
            report["metrics"]["memory_percent"] = round(mem.percent, 1)
            report["metrics"]["memory_available_mb"] = round(mem.available / 1024 / 1024)

            if cpu > self.thresholds["cpu"]:
                report["critical"].append(f"CPU critical: {cpu}%")
            elif cpu > self.thresholds["cpu"] * 0.8:
                report["warnings"].append(f"CPU high: {cpu}%")

            if mem.percent > self.thresholds["memory"]:
                report["critical"].append(f"Memory critical: {mem.percent}%")
            elif mem.percent > self.thresholds["memory"] * 0.8:
                report["warnings"].append(f"Memory high: {mem.percent}%")

        if self.tick_durations:
            avg_ms = sum(self.tick_durations) / len(self.tick_durations)
            report["metrics"]["avg_tick_ms"] = round(avg_ms, 1)
            if avg_ms > self.thresholds["tick_duration_ms"]:
                report["warnings"].append(f"Slow ticks: {avg_ms:.0f}ms avg")

        if self.consecutive_errors >= self.thresholds["consecutive_errors"]:
            report["critical"].append(f"Consecutive errors: {self.consecutive_errors}")

        if report["critical"]:
            report["status"] = "critical"
        elif report["warnings"]:
            report["status"] = "warning"

        if report["status"] == "critical":
            self.incidents.append({"time": CLOCK.now(), "issues": report["critical"]})
            if len(self.incidents) > 50:
                self.incidents = self.incidents[-50:]

        for w in report["warnings"]:
            self.warnings.append(w)

        # Count a recovery when the system climbs back to healthy from a
        # degraded (warning) or critical state — previously recovery_count was
        # dead (never incremented) yet still reported in status().
        if self._prev_status in ("warning", "critical") and report["status"] == "healthy":
            self.recovery_count += 1
        self._prev_status = report["status"]

        return report

    def record_tick(self, duration_ms: float, success: bool = True):
        self.tick_durations.append(duration_ms)
        if success:
            self.successful_ticks += 1
            self.consecutive_errors = 0
        else:
            self.failed_ticks += 1
            self.consecutive_errors += 1
            self.error_count += 1

    def status(self) -> dict:
        uptime = CLOCK.now() - self.start_time
        total = self.successful_ticks + self.failed_ticks
        return {
            "uptime_seconds": round(uptime, 1),
            "total_ticks": total,
            "success_rate": round(self.successful_ticks / max(total, 1) * 100, 1),
            "error_count": self.error_count,
            "recovery_count": self.recovery_count,
            "consecutive_errors": self.consecutive_errors,
            "health_status": "critical" if self.consecutive_errors >= 5 else ("warning" if self.consecutive_errors > 0 else "healthy"),
            "cpu_avg": round(sum(self.cpu_history) / max(len(self.cpu_history), 1), 1) if self.cpu_history else 0,
            "mem_avg": round(sum(self.mem_history) / max(len(self.mem_history), 1), 1) if self.mem_history else 0,
            "recent_incidents": len(self.incidents),
            "has_psutil": HAS_PSUTIL,
        }
