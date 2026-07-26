"""MotorCortex — action execution system (console output, voice synthesis, device control)."""
from aegis.clock import CLOCK
import queue
import threading
from collections import deque

# Bound on pending speech utterances. pyttsx3 is NOT thread-safe: overlapping
# runAndWait() calls raise "run loop already started", and one thread per
# utterance grows unbounded under load. We therefore serialize ALL speech
# through a single worker thread draining a bounded queue.
_SPEECH_QUEUE_MAX = 32


class MotorCortex:
    """Executes actions: console logging, voice output (optional), action dispatching."""

    def __init__(self):
        self.action_log: deque = deque(maxlen=100)
        self.actions_executed = 0
        self.voice_enabled = False
        self.voice_engine = None
        # Speech is serialized through ONE background worker + a bounded queue so
        # at most one utterance ever touches the shared pyttsx3 engine at a time
        # and callers (the async tick loop) never block on speech.
        self._speech_queue: "queue.Queue[str]" = queue.Queue(maxsize=_SPEECH_QUEUE_MAX)
        self._speech_worker: threading.Thread | None = None
        self._speech_lock = threading.Lock()
        self.speech_dropped = 0
        self._init_voice()

    def _ensure_speech_worker(self):
        """Lazily start the single speech worker (idempotent, thread-safe)."""
        if self._speech_worker is not None and self._speech_worker.is_alive():
            return
        with self._speech_lock:
            if self._speech_worker is not None and self._speech_worker.is_alive():
                return
            t = threading.Thread(target=self._speech_loop, name="motor-speech", daemon=True)
            self._speech_worker = t
            t.start()

    def _speech_loop(self):
        """Drain the speech queue one utterance at a time. runAndWait() blocks
        until the current utterance finishes, guaranteeing serialization."""
        while True:
            text = self._speech_queue.get()
            try:
                engine = self.voice_engine
                if engine is not None:
                    engine.say(text)
                    engine.runAndWait()
            except Exception:
                # A single bad utterance (engine hiccup, no audio device) must
                # never kill the worker or propagate to the caller.
                pass
            finally:
                self._speech_queue.task_done()

    def _enqueue_speech(self, text: str) -> bool:
        """Queue an utterance for the single worker. Never blocks: returns False
        if the utterance was dropped because the queue is full."""
        self._ensure_speech_worker()
        try:
            self._speech_queue.put_nowait(text)
            return True
        except queue.Full:
            self.speech_dropped += 1
            return False

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
            "time": CLOCK.now(),
            "type": action_type,
            "archetype": archetype,
            "goal": goal,
            "success": True,
            "output": "",
        }

        if action_type == "speak" and self.voice_enabled and self.voice_engine:
            text = payload.get("text", "")
            # Hand the utterance to the single serialized speech worker instead
            # of spawning a fresh thread against the shared (non-thread-safe)
            # engine. Never blocks the caller / tick loop.
            if self._enqueue_speech(text):
                result["output"] = f"Speaking (async): {text[:50]}"
            else:
                result["success"] = False
                result["output"] = f"Speech dropped (queue full): {text[:50]}"

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
