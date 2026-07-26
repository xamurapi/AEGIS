"""MetaConsciousness — analyzes fragmentation, coherence and integration of the system's 'personality'."""
from collections import deque
from aegis.clock import CLOCK


class MetaConsciousness:
    """Evaluates how well internal subsystems are aligned and integrated."""

    def __init__(self):
        self.evaluations: deque = deque(maxlen=50)
        self.fragmentation_score = 0.0  # 0 = fully integrated, 1 = fully fragmented
        self.coherence_score = 1.0      # 1 = perfectly coherent
        self.recommendations: list[str] = []
        self.integration_events: deque = deque(maxlen=30)

    def evaluate(self, consciousness_mode: str, active_archetype_name: str | None,
                 mood: str, energy: float, goal_focus: str | None,
                 archetypes: list | None = None) -> dict:
        """Run a meta-consciousness evaluation — detect fragmentation and conflicts."""

        conflicts = []
        alignment_score = 1.0

        # Conflict: survival mode but explorer archetype
        if consciousness_mode == "survival" and active_archetype_name == "Explorer":
            conflicts.append("Survival mode conflicts with Explorer archetype — system wants safety but archetype pushes exploration")
            alignment_score *= 0.6

        # Conflict: low energy but reflective mode (expensive)
        if energy < 0.2 and consciousness_mode == "reflective":
            conflicts.append("Reflective mode is energy-expensive but energy is critical")
            alignment_score *= 0.7

        # Conflict: negative mood but growth goals. Use the actual emotion
        # names the system emits (see emotions.EMOTION_MAP) — "frustrated"/"sad"
        # never existed, so this check could never fire.
        if mood in ("anxious", "fear", "sadness", "shame") and goal_focus and "optim" in goal_focus.lower():
            conflicts.append(f"Negative mood '{mood}' conflicts with optimization goal")
            alignment_score *= 0.8

        # Check archetype alignment. Archetype objects expose success_score, not
        # an "influence" attribute (that lives in ArchetypeGeopolitics), so the
        # old getattr default of 0.5 always produced a zero spread.
        if archetypes and len(archetypes) > 1:
            influences = [getattr(a, "success_score", 0.5) for a in archetypes]
            spread = max(influences) - min(influences)
            if spread > 0.5:
                conflicts.append(f"High archetype influence spread ({spread:.2f}) — personality fragmentation")
                alignment_score *= 0.75

        # Fragmentation = number and severity of conflicts
        self.fragmentation_score = max(0.0, min(1.0, 1.0 - alignment_score))
        self.coherence_score = alignment_score

        # Generate recommendations
        self.recommendations = []
        if self.fragmentation_score > 0.4:
            self.recommendations.append("Consider archetype integration to reduce fragmentation")
        if energy < 0.15:
            self.recommendations.append("Switch to instinctive mode to conserve energy")
        if conflicts:
            self.recommendations.append(f"Resolve {len(conflicts)} internal conflict(s)")
        if self.coherence_score > 0.9:
            self.recommendations.append("System is well-integrated — continue current strategy")

        result = {
            "time": CLOCK.now(),
            "fragmentation": round(self.fragmentation_score, 3),
            "coherence": round(self.coherence_score, 3),
            "conflicts": conflicts,
            "recommendations": self.recommendations[:3],
            "alignment_score": round(alignment_score, 3),
        }

        self.evaluations.append(result)
        return result

    def log_integration_event(self, event_type: str, description: str):
        """Log when archetypes merge, conflicts resolve, etc."""
        self.integration_events.append({
            "time": CLOCK.now(),
            "type": event_type,
            "description": description,
        })

    def status(self) -> dict:
        return {
            "fragmentation": round(self.fragmentation_score, 3),
            "coherence": round(self.coherence_score, 3),
            "recommendations": self.recommendations[:3],
            "evaluations_count": len(self.evaluations),
            "integration_events": len(self.integration_events),
            "last_evaluation": self.evaluations[-1] if self.evaluations else None,
        }
