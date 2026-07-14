"""Consciousness — mode switching based on emotional state and energy (instinctive/heuristic/reflective)."""
import time


class ConsciousnessState:
    def __init__(self):
        self.mode = "heuristic"
        self.switch_log: list[dict] = []
        self.mode_durations: dict[str, float] = {
            "survival": 0, "instinctive": 0, "heuristic": 0, "reflective": 0,
        }
        self._mode_start = time.time()

    def update_mode(self, mood: str, energy: float, arousal: float = 0.5) -> str:
        old_mode = self.mode

        # Order matters: the most critical energy floor must be tested first,
        # otherwise "instinctive" (energy < 0.25) captures every case that
        # should be "survival" (energy < 0.15).
        if energy < 0.15:
            self.mode = "survival"
        elif mood in ("anxious", "fear", "anger") or energy < 0.25:
            self.mode = "instinctive"
        elif mood in ("inspired", "curiosity", "excitement") and energy > 0.5 and arousal < 0.85:
            self.mode = "reflective"
        else:
            self.mode = "heuristic"

        now = time.time()
        elapsed = now - self._mode_start
        if old_mode in self.mode_durations:
            self.mode_durations[old_mode] += elapsed
        self._mode_start = now

        if old_mode != self.mode:
            self.switch_log.append({
                "from": old_mode,
                "to": self.mode,
                "time": now,
                "trigger_mood": mood,
                "trigger_energy": round(energy, 3),
            })
            if len(self.switch_log) > 200:
                self.switch_log = self.switch_log[-200:]

        return self.mode

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "mode_durations": {k: round(v, 1) for k, v in self.mode_durations.items()},
            "recent_switches": self.switch_log[-10:],
            "total_switches": len(self.switch_log),
        }
