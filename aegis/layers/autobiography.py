"""Autobiography — narrative self-history with impact scoring."""
from aegis.clock import CLOCK


class Autobiographer:
    def __init__(self):
        self.events: list[dict] = []
        self.total_impact = 0.0

    def log_event(self, category: str, summary: str, impact: float):
        entry = {
            "time": CLOCK.now(),
            "category": category,
            "summary": summary,
            "impact": round(min(max(impact, 0.0), 1.0), 3),
        }
        self.events.append(entry)
        self.total_impact += entry["impact"]
        if len(self.events) > 500:
            # Keep total_impact consistent with the retained events: subtract the
            # impact of the events being dropped instead of letting the sum drift
            # above what the (truncated) list actually holds.
            dropped = self.events[:-500]
            self.total_impact -= sum(e["impact"] for e in dropped)
            self.events = self.events[-500:]

    def generate_narrative(self, last_n: int = 10) -> str:
        if not self.events:
            return "No significant events yet."
        lines = ["Life narrative:"]
        for e in self.events[-last_n:]:
            lines.append(f"  [{e['category']}] {e['summary']} (impact: {e['impact']})")
        return "\n".join(lines)

    def get_milestones(self, threshold: float = 0.7) -> list[dict]:
        return [e for e in self.events if e["impact"] >= threshold]

    def status(self) -> dict:
        return {
            "total_events": len(self.events),
            "total_impact": round(self.total_impact, 2),
            "milestones": len(self.get_milestones()),
            "recent": [
                {"category": e["category"], "summary": e["summary"][:60], "impact": e["impact"]}
                for e in self.events[-5:]
            ],
        }
