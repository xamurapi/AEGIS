"""MotorCortex — action execution system (console output, voice synthesis, device control)."""
import time
from collections import deque


class MotorCortex:
    """Executes actions: console logging, voice output (optional), action dispatching."""

    def __init__(self):
        self.action_log: deque = deque(maxlen=100)
        self.actions_executed = 0
        self.voice_enabled = False
        self.voice_engine = None
        self._init_voice()

    def _init_voice(self):
        """Try to initialize voice synthesis."""
        try:
            import pyttsx3
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty("rate", 150)
            self.voice_enabled = True
        except Exception:
            self.voice_enabled = False

    def execute(self, action_type: str, payload: dict | None = None,
                archetype: str | None = None, goal: str | None = None) -> dict:
        """Execute an action based on type and context."""
        payload = payload or {}
        result = {
            "time": time.time(),
            "type": action_type,
            "archetype": archetype,
            "goal": goal,
            "success": True,
            "output": "",
        }

        if action_type == "speak" and self.voice_enabled and self.voice_engine:
            text = payload.get("text", "")
            try:
                self.voice_engine.say(text)
                self.voice_engine.runAndWait()
                result["output"] = f"Spoke: {text[:50]}"
            except Exception as e:
                result["success"] = False
                result["output"] = str(e)

        elif action_type == "log":
            message = payload.get("message", "")
            result["output"] = f"Logged: {message[:100]}"

        elif action_type == "internal_computation":
            result["output"] = "Internal computation cycle completed"

        elif action_type == "alert":
            level = payload.get("level", "info")
            message = payload.get("message", "")
            result["output"] = f"[{level.upper()}] {message[:100]}"

        elif action_type == "recharge":
            amount = payload.get("amount", 0.1)
            result["output"] = f"Recharge requested: +{amount}"

        else:
            result["output"] = f"Action dispatched: {action_type}"

        self.actions_executed += 1
        self.action_log.append(result)
        return result

    def status(self) -> dict:
        return {
            "actions_executed": self.actions_executed,
            "voice_enabled": self.voice_enabled,
            "recent_actions": [
                {"type": a["type"], "output": a["output"][:60], "success": a["success"]}
                for a in list(self.action_log)[-5:]
            ],
        }
