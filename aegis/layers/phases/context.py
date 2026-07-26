"""TickContext — the object the five phases hand to each other (spec §3.9).

Per-tick state used to live on the substrate as a scatter of ``_tick_*``
attributes reset at the top of ``tick()``. That worked while there were three
of them; with a plan, a prediction, a resource lease and an open experience
still to come it would not. Collecting them in one object also makes "what a
tick produced" a thing that can be inspected, logged and asserted on.

The substrate keeps property aliases for the older attribute names, so the
existing suite and the API surface are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from aegis.clock import CLOCK

PHASE_ORDER = ("perceive", "evaluate", "decide", "act", "reflect")


@dataclass
class TickContext:
    """Everything one pass through the cycle accumulates."""

    tick: int = 0
    started_at: float = field(default_factory=CLOCK.now)

    # ── what this tick learned (drives goal progress and importance) ──
    new_concepts: int = 0
    new_episodic: int = 0
    llm_insights: int = 0

    # ── what this tick decided ───────────────────────────────────────
    regulation_directives: dict = field(default_factory=dict)
    pending_experiences: dict = field(default_factory=dict)
    decision: str | None = None
    confidence: float | None = None

    # ── phases that blocked on the outside world this tick ───────────
    # Network, hosted model or subprocess. The per-phase latency budget skips
    # these samples: including them would measure the provider's response time
    # and report it as a slow cognitive cycle.
    external: set[str] = field(default_factory=set)

    # ── per-phase wall time, filled in by the substrate ──────────────
    durations_ms: dict[str, float] = field(default_factory=dict)

    def mark_external(self, phase: str) -> None:
        self.external.add(phase)

    def did_external(self, phase: str) -> bool:
        return phase in self.external

    def record_duration(self, phase: str, duration_ms: float) -> None:
        self.durations_ms[phase] = duration_ms

    def learned_something(self) -> bool:
        """Whether this tick produced knowledge — the success signal the
        feedback loop closes an experience with."""
        return self.new_concepts > 0 or self.llm_insights > 0

    def summary(self) -> dict:
        return {
            "tick": self.tick,
            "new_concepts": self.new_concepts,
            "new_episodic": self.new_episodic,
            "llm_insights": self.llm_insights,
            "decision": self.decision,
            "confidence": self.confidence,
            "external_phases": sorted(self.external),
            "durations_ms": {k: round(v, 3) for k, v in self.durations_ms.items()},
        }
