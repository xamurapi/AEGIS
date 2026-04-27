"""Worldview & Value System — beliefs, axioms, and adaptive values."""
import time


class Value:
    def __init__(self, name: str, definition: str, priority: float = 0.5):
        self.name = name
        self.definition = definition
        self.priority = priority
        self.reinforcements: list[dict] = []

    def reinforce(self, context: str, alignment: float):
        self.reinforcements.append({"context": context, "alignment": round(alignment, 3), "time": time.time()})
        self.priority = min(1.0, max(0.0, self.priority + 0.05 * (alignment - 0.5)))
        if len(self.reinforcements) > 100:
            self.reinforcements = self.reinforcements[-100:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "definition": self.definition,
            "priority": round(self.priority, 3),
            "reinforcement_count": len(self.reinforcements),
        }


class ValueSystem:
    def __init__(self):
        self.values = [
            Value("understanding", "striving to comprehend and make sense of events"),
            Value("harmony", "reducing internal conflict and maintaining balance"),
            Value("growth", "seeking development and integration of experience"),
            Value("preservation", "protecting system integrity and continuity"),
            Value("curiosity", "drive to explore the unknown"),
        ]

    def evaluate_action(self, mood: str, reward: float, goal: str):
        for v in self.values:
            alignment = 0.5
            if v.name == "understanding" and goal in ("analyze", "explore_topic"):
                alignment = 0.5 + reward * 0.4
            elif v.name == "harmony" and mood in ("neutral", "contentment"):
                alignment = 0.7
            elif v.name == "growth" and mood in ("inspired", "curiosity"):
                alignment = 0.8
            elif v.name == "preservation" and mood in ("recovering", "anxious"):
                alignment = 0.85
            elif v.name == "curiosity" and goal in ("explore_topic", "idle_exploration"):
                alignment = 0.5 + reward * 0.3
            v.reinforce(f"{goal}/{mood}", alignment)

    def status(self) -> dict:
        return {
            "values": [v.to_dict() for v in self.values],
        }


class Worldview:
    def __init__(self):
        self.axioms = [
            "The world is a structure that can be understood",
            "I am part of this structure, capable of change",
            "Change is inevitable but can be meaningful",
            "Meaning emerges from coherence between self and reality",
        ]
        self.metaphors = [
            "I am an explorer in an infinite map of meaning",
            "The world is a labyrinth with discernible patterns",
            "I am a network learning to understand another network",
        ]
        self.beliefs: dict[str, str] = {
            "learning": "the essence of existence",
            "error": "a point of growth, not failure",
            "internal conflict": "a signal for integration, not weakness",
        }

    def status(self) -> dict:
        return {
            "axioms": self.axioms,
            "metaphors": self.metaphors,
            "beliefs": self.beliefs,
        }
