"""Health Monitor — system resource monitoring and emergency prevention."""
from aegis.clock import CLOCK
import platform
from collections import deque

from aegis.config import PHASE_BUDGET_MS, PHASE_BUDGET_WINDOW

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

PHASES = ("perceive", "evaluate", "decide", "act", "reflect")


class HealthMonitor:
    def __init__(self):
        self.start_time = CLOCK.now()
        self.cpu_history: deque = deque(maxlen=60)
        self.mem_history: deque = deque(maxlen=60)
        self.tick_durations: deque = deque(maxlen=100)
        # Per-phase latency. Two series per phase: `local` holds only the ticks
        # where the phase did no external work and is what the budget is judged
        # on; `all` holds everything and is what the dashboard shows. Judging a
        # budget on the combined series would measure the LLM provider's
        # response time and call it a code regression.
        self.phase_local: dict[str, deque] = {
            p: deque(maxlen=PHASE_BUDGET_WINDOW * 5) for p in PHASES
        }
        self.phase_all: dict[str, deque] = {
            p: deque(maxlen=PHASE_BUDGET_WINDOW * 5) for p in PHASES
        }
        self.phase_budgets: dict[str, float] = dict(PHASE_BUDGET_MS)
        self.phase_breaches: dict[str, int] = {p: 0 for p in PHASES}
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

    @property
    def last_status(self) -> str:
        """Status of the most recent ``check()``.

        Exposed because telemetry samples health every tick and re-running the
        whole check (psutil probes included) purely to read its verdict would
        make observing the system a measurable part of its cost.
        """
        return self._prev_status

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

        # Per-phase budgets. A breach is a warning, never critical: a cognitive
        # phase that got slower is a regression to investigate, not a reason to
        # put the system into emergency mode.
        phases = self.phase_report()
        for phase, data in phases.items():
            if data["over_budget"]:
                self.phase_breaches[phase] += 1
                report["warnings"].append(
                    f"{phase} over budget: {data['avg_local_ms']:.1f}ms "
                    f"> {data['budget_ms']:.0f}ms")
        report["metrics"]["phases"] = phases

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

    # ── per-phase latency (spec §3.4) ────────────────────────────────

    def record_phase(self, phase: str, duration_ms: float, external: bool = False):
        """Record one phase's duration.

        ``external=True`` means the phase made a network / LLM / subprocess
        call this tick. Such samples are reported but excluded from the budget,
        so a slow provider can never be mistaken for a slow cycle.
        """
        if phase not in self.phase_all:
            return
        self.phase_all[phase].append(duration_ms)
        if not external:
            self.phase_local[phase].append(duration_ms)

    def phase_report(self) -> dict:
        """Per-phase latency against budget, computed on local ticks only."""
        report = {}
        for phase in PHASES:
            local = list(self.phase_local[phase])
            everything = list(self.phase_all[phase])
            window = local[-PHASE_BUDGET_WINDOW:]
            avg_local = sum(window) / len(window) if window else 0.0
            budget = self.phase_budgets.get(phase, 0.0)
            # A budget is only judged once the window is actually full —
            # otherwise the first cold tick of a run reads as a breach.
            over = bool(window) and len(window) >= PHASE_BUDGET_WINDOW and avg_local > budget
            report[phase] = {
                "budget_ms": round(budget, 2),
                "avg_local_ms": round(avg_local, 3),
                "max_local_ms": round(max(window), 3) if window else 0.0,
                "avg_all_ms": round(sum(everything) / len(everything), 3) if everything else 0.0,
                "samples_local": len(local),
                "samples_all": len(everything),
                "over_budget": over,
                "breaches": self.phase_breaches[phase],
            }
        return report

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
            "phases": self.phase_report(),
        }
