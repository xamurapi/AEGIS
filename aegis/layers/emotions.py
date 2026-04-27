"""Emotional system — VAD model (Valence-Arousal-Dominance) with mixed emotions.

All state transitions are deterministic, driven by real system metrics (reward,
context flags, energy).  No random noise — emotion is a pure function of inputs.
"""
import math
import time
from collections import deque


EMOTION_MAP = {
    "joy":            {"valence": 0.9, "arousal": 0.7, "dominance": 0.7},
    "sadness":        {"valence": 0.2, "arousal": 0.3, "dominance": 0.3},
    "anger":          {"valence": 0.2, "arousal": 0.9, "dominance": 0.8},
    "fear":           {"valence": 0.1, "arousal": 0.8, "dominance": 0.2},
    "surprise":       {"valence": 0.5, "arousal": 0.9, "dominance": 0.5},
    "disgust":        {"valence": 0.1, "arousal": 0.6, "dominance": 0.6},
    "curiosity":      {"valence": 0.6, "arousal": 0.7, "dominance": 0.6},
    "pride":          {"valence": 0.8, "arousal": 0.6, "dominance": 0.9},
    "shame":          {"valence": 0.2, "arousal": 0.4, "dominance": 0.1},
    "excitement":     {"valence": 0.8, "arousal": 0.9, "dominance": 0.7},
    "contentment":    {"valence": 0.7, "arousal": 0.3, "dominance": 0.6},
    "anxiety":        {"valence": 0.3, "arousal": 0.7, "dominance": 0.3},
    "hope":           {"valence": 0.7, "arousal": 0.6, "dominance": 0.5},
    "disappointment": {"valence": 0.3, "arousal": 0.4, "dominance": 0.3},
    "relief":         {"valence": 0.6, "arousal": 0.2, "dominance": 0.5},
    "neutral":        {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
    "inspired":       {"valence": 0.8, "arousal": 0.8, "dominance": 0.8},
    "anxious":        {"valence": 0.3, "arousal": 0.7, "dominance": 0.3},
    "recovering":     {"valence": 0.5, "arousal": 0.3, "dominance": 0.4},
    "contemplative":  {"valence": 0.5, "arousal": 0.4, "dominance": 0.6},
    "determined":     {"valence": 0.6, "arousal": 0.7, "dominance": 0.8},
}

MOOD_MODIFIERS = {
    "inspired": 1.3, "joy": 1.2, "excitement": 1.25, "determined": 1.15,
    "contentment": 1.1, "neutral": 1.0, "contemplative": 0.95, "anxious": 0.85,
    "fear": 0.8, "sadness": 0.75, "anger": 0.9, "shame": 0.7, "recovering": 0.9,
}


class EmotionalSystem:
    def __init__(self):
        self.energy = 1.0
        self.valence = 0.5
        self.arousal = 0.5
        self.dominance = 0.5
        self.certainty = 0.5
        self.mood = "neutral"
        self.success_rate = 0.5
        self.mood_duration = 0
        self.mixed_emotions: list[tuple[str, float]] = []
        self.history: deque = deque(maxlen=100)
        self.emotional_memories: dict[str, list] = {}

    def update(self, reward: float, context: dict | None = None):
        self.success_rate = 0.9 * self.success_rate + 0.1 * reward
        self.energy = max(0.0, self.energy - 0.005)

        if context:
            if context.get("unexpected"):
                self.arousal = min(1.0, self.arousal + 0.15)
            if context.get("repetitive"):
                self.arousal = max(0.0, self.arousal - 0.08)
            if context.get("error"):
                self.arousal = min(1.0, self.arousal + 0.1)
            if context.get("new_knowledge"):
                self.arousal = min(1.0, self.arousal + 0.05)
        else:
            # Decay arousal toward baseline (0.5) when nothing notable happens
            self.arousal = 0.95 * self.arousal + 0.05 * 0.5

        self.valence = 0.7 * self.valence + 0.3 * self.success_rate
        self.dominance = 0.9 * self.dominance + 0.1 * (reward * self.energy)

        recent_rewards = [h["reward"] for h in list(self.history)[-10:] if "reward" in h]
        if len(recent_rewards) > 2:
            variance = sum((r - sum(recent_rewards)/len(recent_rewards))**2 for r in recent_rewards) / len(recent_rewards)
            self.certainty = 1.0 - min(variance * 2, 1.0)

        old_mood = self.mood
        self.mood = self._determine_mood()
        self.mood_duration = self.mood_duration + 1 if self.mood == old_mood else 0

        if old_mood != self.mood and context:
            key = f"{old_mood}->{self.mood}"
            self.emotional_memories.setdefault(key, []).append({
                "success": self.success_rate, "energy": self.energy,
            })

        self.mixed_emotions = self._determine_mixed()
        self._regulate()

        self.history.append({
            "mood": self.mood, "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3), "dominance": round(self.dominance, 3),
            "energy": round(self.energy, 3), "reward": round(reward, 3),
        })

    def _determine_mood(self) -> str:
        state = (self.valence, self.arousal, self.dominance)
        best, best_dist = "neutral", float("inf")
        for name, params in EMOTION_MAP.items():
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(state, (params["valence"], params["arousal"], params["dominance"]))))
            if d < best_dist:
                best_dist = d
                best = name
        return best

    def _determine_mixed(self) -> list[tuple[str, float]]:
        state = (self.valence, self.arousal, self.dominance)
        mixed = []
        for name, params in EMOTION_MAP.items():
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(state, (params["valence"], params["arousal"], params["dominance"]))))
            if d < 0.3:
                mixed.append((name, round(1.0 - d / 0.3, 2)))
        mixed.sort(key=lambda x: x[1], reverse=True)
        return mixed[:3]

    def _regulate(self):
        if self.mood == "anxious" and self.success_rate > 0.4:
            self.valence += 0.05
            self.arousal -= 0.025
            self.mood = "recovering"
        if self.mood_duration > 20:
            # Prolonged same mood → push arousal and valence toward baseline
            self.arousal = 0.97 * self.arousal + 0.03 * 0.5
            self.valence = 0.97 * self.valence + 0.03 * 0.5
        if self.arousal > 0.9:
            self.arousal *= 0.95
        if self.arousal < 0.1:
            self.arousal += 0.05
        if self.energy < 0.2:
            self.arousal = min(self.arousal, 0.6)
            self.dominance *= 0.95

    def emotional_modifier(self) -> float:
        base = MOOD_MODIFIERS.get(self.mood, 1.0)
        return base * (0.5 + 0.5 * self.energy) * (0.8 + 0.2 * self.certainty)

    def get_color(self) -> str:
        r = int(255 * (1 - self.valence))
        g = int(255 * self.valence)
        b = int(255 * (0.5 + 0.5 * self.dominance))
        brightness = 0.3 + 0.7 * self.arousal
        r, g, b = int(r * brightness), int(g * brightness), int(b * brightness)
        return f"#{min(r,255):02x}{min(g,255):02x}{min(b,255):02x}"

    def recharge(self, amount: float = 0.3):
        self.energy = min(1.0, self.energy + amount)

    def status(self) -> dict:
        return {
            "mood": self.mood,
            "energy": round(self.energy, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "certainty": round(self.certainty, 3),
            "mood_duration": self.mood_duration,
            "mixed_emotions": self.mixed_emotions,
            "color": self.get_color(),
            "modifier": round(self.emotional_modifier(), 3),
            "success_rate": round(self.success_rate, 3),
        }
