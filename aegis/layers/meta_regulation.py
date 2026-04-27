"""MetaRegulation — energy management, energy-saving and emergency modes."""
import time
from collections import deque


class MetaRegulator:
    """Regulates energy consumption across the system, introduces power-saving and emergency modes."""

    # Modes: normal, eco (energy-saving), emergency, recovery
    MODES = ("normal", "eco", "emergency", "recovery")

    def __init__(self):
        self.mode = "normal"
        self.mode_history: deque = deque(maxlen=50)
        self.energy_budget = 1.0      # normalized 0..1
        self.consumption_rate = 0.02  # energy consumed per tick
        self.savings_applied = 0
        self.emergency_activations = 0
        self.tick_skip_counter = 0    # how many expensive ticks were skipped in eco mode

    def regulate(self, energy: float, health_status: str, consecutive_errors: int,
                 consciousness_mode: str) -> dict:
        """Determine regulation mode and return directives for the tick."""

        old_mode = self.mode
        directives = {
            "skip_llm": False,
            "skip_dreams": False,
            "skip_learning": False,
            "reduce_sensors": False,
            "force_recharge": 0.0,
        }

        # Decide mode
        if energy < 0.08 or health_status == "critical" or consecutive_errors >= 5:
            self.mode = "emergency"
            self.emergency_activations += 1
        elif energy < 0.2:
            self.mode = "eco"
        elif energy < 0.35 and self.mode == "eco":
            self.mode = "eco"  # stay in eco until energy recovers above threshold
        elif energy > 0.5 and health_status != "critical" and consecutive_errors < 2:
            self.mode = "normal"
        elif self.mode == "emergency" and energy > 0.3 and consecutive_errors < 3:
            self.mode = "recovery"
        elif self.mode == "recovery" and energy > 0.6:
            self.mode = "normal"

        # Apply directives based on mode
        if self.mode == "eco":
            directives["skip_llm"] = True
            directives["skip_dreams"] = True
            directives["reduce_sensors"] = True
            self.savings_applied += 1
            self.tick_skip_counter += 1

        elif self.mode == "emergency":
            directives["skip_llm"] = True
            directives["skip_dreams"] = True
            directives["skip_learning"] = True
            directives["reduce_sensors"] = True
            directives["force_recharge"] = 0.15

        elif self.mode == "recovery":
            directives["skip_llm"] = True
            directives["force_recharge"] = 0.05

        if old_mode != self.mode:
            self.mode_history.append({
                "time": time.time(),
                "from": old_mode,
                "to": self.mode,
                "energy": round(energy, 3),
            })

        return {
            "mode": self.mode,
            "directives": directives,
            "energy": round(energy, 3),
        }

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "energy_budget": round(self.energy_budget, 3),
            "savings_applied": self.savings_applied,
            "emergency_activations": self.emergency_activations,
            "tick_skips": self.tick_skip_counter,
            "mode_transitions": len(self.mode_history),
            "recent_transitions": [
                {"from": t["from"], "to": t["to"], "energy": t["energy"]}
                for t in list(self.mode_history)[-5:]
            ],
        }
