"""Archetypes — personality sub-systems with activation conditions and geopolitical dynamics."""


class Archetype:
    def __init__(self, name: str, activation_moods: list[str], energy_range: tuple[float, float],
                 strategies: dict[str, str], tone: str):
        self.name = name
        self.activation_moods = activation_moods
        self.energy_range = energy_range
        self.strategies = strategies
        self.tone = tone
        self.success_score = 0.5
        self.steps_active = 0
        self.experience: list[dict] = []

    def should_activate(self, mood: str, energy: float) -> bool:
        mood_match = mood in self.activation_moods
        energy_match = self.energy_range[0] <= energy <= self.energy_range[1]
        return mood_match or (energy_match and not mood_match and energy < 0.3)

    def act(self, consciousness_mode: str, goal: str) -> str:
        base = self.strategies.get(consciousness_mode, "operating in default mode")
        return f"[{self.name}] {self.tone}: {base} | goal: {goal}"

    def log_experience(self, tick: int, mood: str, reward: float, action: str):
        self.experience.append({"tick": tick, "mood": mood, "reward": round(reward, 3), "action": action})
        self.steps_active += 1
        self.success_score = 0.9 * self.success_score + 0.1 * reward
        if len(self.experience) > 200:
            self.experience = self.experience[-200:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tone": self.tone,
            "success_score": round(self.success_score, 3),
            "steps_active": self.steps_active,
            "experience_count": len(self.experience),
        }


def create_default_archetypes() -> list[Archetype]:
    return [
        Archetype(
            name="Sentinel",
            activation_moods=["anxious", "fear", "anger"],
            energy_range=(0.0, 0.3),
            strategies={
                "instinctive": "minimizing activity for safety",
                "heuristic": "monitoring for threats",
                "reflective": "analyzing instability patterns",
                "survival": "emergency preservation mode",
            },
            tone="Cold and observant",
        ),
        Archetype(
            name="Explorer",
            activation_moods=["inspired", "curiosity", "excitement"],
            energy_range=(0.5, 1.0),
            strategies={
                "heuristic": "exploring new strategies",
                "reflective": "comparing hypotheses with experience",
                "instinctive": "quick experimental probes",
            },
            tone="Curious and open",
        ),
        Archetype(
            name="Caretaker",
            activation_moods=["recovering", "sadness", "contentment"],
            energy_range=(0.2, 0.6),
            strategies={
                "instinctive": "stabilizing system state",
                "heuristic": "adjusting behavior gently",
                "reflective": "restoring inner balance",
            },
            tone="Gentle and protective",
        ),
    ]


class ArchetypeGeopolitics:
    """Models influence dynamics between archetypes."""

    def __init__(self, archetypes: list[Archetype]):
        self.archetypes = {a.name: a for a in archetypes}
        self.influence = {a.name: 1.0 for a in archetypes}
        self.dominant: str | None = None
        self.relationships = {
            ("Sentinel", "Caretaker"): 1,
            ("Explorer", "Sentinel"): -1,
            ("Explorer", "Caretaker"): 0,
        }

    def update_influence(self):
        for name, arch in self.archetypes.items():
            base = arch.success_score
            if arch.experience:
                last_mood = arch.experience[-1].get("mood", "neutral")
                if last_mood in ("inspired", "curiosity"):
                    base *= 1.2
                elif last_mood in ("anxious", "fear"):
                    base *= 0.7
            self.influence[name] = round(base, 3)

        for (a1, a2), rel in self.relationships.items():
            if a1 in self.influence and a2 in self.influence:
                if rel == 1:
                    self.influence[a1] += 0.05 * self.influence[a2]
                elif rel == -1:
                    self.influence[a1] -= 0.03 * self.influence[a2]
                self.influence[a1] = max(0.1, self.influence[a1])

        self.dominant = max(self.influence, key=lambda n: self.influence[n])

    def get_dominant(self) -> Archetype | None:
        return self.archetypes.get(self.dominant)

    def detect_conflict(self) -> bool:
        scores = [a.success_score for a in self.archetypes.values() if len(a.experience) > 5]
        if len(scores) < 2:
            return False
        return max(scores) - min(scores) > 0.4

    def status(self) -> dict:
        return {
            "dominant": self.dominant,
            "influence": {k: round(v, 3) for k, v in self.influence.items()},
            "conflict_detected": self.detect_conflict(),
            "archetypes": {name: a.to_dict() for name, a in self.archetypes.items()},
        }
