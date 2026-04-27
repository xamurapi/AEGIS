"""MetaReflection — deep self-analysis, insight generation, trend detection."""
import time
import random
from collections import deque


class MetaReflection:
    """Analyzes system's own history to generate insights and self-reports."""

    def __init__(self):
        self.insights: deque = deque(maxlen=50)
        self.trends: dict[str, list[float]] = {
            "energy": [],
            "mood_valence": [],
            "error_rate": [],
            "goal_completion": [],
        }
        self.reflection_count = 0
        self.self_reports: deque = deque(maxlen=20)

    def reflect(self, tick: int, energy: float, mood: str, mood_valence: float,
                error_rate: float, goals_completed: int, goals_total: int,
                consciousness_mode: str, recent_events: list[str]) -> dict:
        """Run a deep reflection cycle — analyze trends and generate insights."""
        self.reflection_count += 1

        # Track trends
        self.trends["energy"].append(energy)
        self.trends["mood_valence"].append(mood_valence)
        self.trends["error_rate"].append(error_rate)
        completion_rate = goals_completed / max(goals_total, 1)
        self.trends["goal_completion"].append(completion_rate)

        # Keep last 50 data points
        for k in self.trends:
            self.trends[k] = self.trends[k][-50:]

        insights = []

        # Trend analysis: energy
        if len(self.trends["energy"]) >= 5:
            recent_energy = self.trends["energy"][-5:]
            if all(recent_energy[i] < recent_energy[i - 1] for i in range(1, len(recent_energy))):
                insights.append("Energy is consistently declining — consider longer rest periods")
            elif all(recent_energy[i] > recent_energy[i - 1] for i in range(1, len(recent_energy))):
                insights.append("Energy is rising steadily — good recovery trajectory")

        # Trend analysis: mood
        if len(self.trends["mood_valence"]) >= 5:
            avg_mood = sum(self.trends["mood_valence"][-5:]) / 5
            if avg_mood < -0.3:
                insights.append(f"Average mood is negative ({avg_mood:.2f}) — emotional regulation needed")
            elif avg_mood > 0.5:
                insights.append(f"Positive emotional trend ({avg_mood:.2f}) — system is thriving")

        # Error pattern
        if len(self.trends["error_rate"]) >= 3:
            recent_errors = self.trends["error_rate"][-3:]
            if sum(recent_errors) / 3 > 0.3:
                insights.append("High error rate detected — consider switching to safer strategies")

        # Goal completion analysis
        if len(self.trends["goal_completion"]) >= 5:
            avg_completion = sum(self.trends["goal_completion"][-5:]) / 5
            if avg_completion < 0.2:
                insights.append("Low goal completion rate — goals may be too ambitious")
            elif avg_completion > 0.8:
                insights.append("High goal completion — consider setting more challenging goals")

        # Mode analysis
        if consciousness_mode == "instinctive" and energy > 0.6:
            insights.append("Still in instinctive mode despite good energy — consider upgrading to heuristic")

        # Event pattern analysis
        if recent_events:
            error_events = [e for e in recent_events if "error" in e.lower() or "fail" in e.lower()]
            if len(error_events) > len(recent_events) * 0.5:
                insights.append("More than half of recent events are errors — systemic issue likely")

        # Generate self-report
        report = {
            "time": time.time(),
            "tick": tick,
            "insights": insights,
            "trends_summary": {
                "energy_trend": self._trend_direction(self.trends["energy"]),
                "mood_trend": self._trend_direction(self.trends["mood_valence"]),
                "error_trend": self._trend_direction(self.trends["error_rate"]),
                "goal_trend": self._trend_direction(self.trends["goal_completion"]),
            },
            "consciousness_mode": consciousness_mode,
            "mood": mood,
            "energy": round(energy, 3),
        }

        for insight in insights:
            self.insights.append({"time": time.time(), "tick": tick, "insight": insight})

        self.self_reports.append(report)
        return report

    @staticmethod
    def _trend_direction(data: list[float]) -> str:
        if len(data) < 3:
            return "insufficient_data"
        recent = data[-3:]
        if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
            return "rising"
        elif all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
            return "falling"
        return "stable"

    def status(self) -> dict:
        return {
            "reflection_count": self.reflection_count,
            "total_insights": len(self.insights),
            "recent_insights": [i["insight"] for i in list(self.insights)[-5:]],
            "trends": {k: self._trend_direction(v) for k, v in self.trends.items()},
            "self_reports_count": len(self.self_reports),
        }
