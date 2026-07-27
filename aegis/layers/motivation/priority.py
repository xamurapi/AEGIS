"""Priority: the step between wanting something and paying for it (spec M4.4).

``goal → value → priority → resource → action``. Value says what an objective
is worth in general; priority says what it is worth *now*. The difference is
everything the current situation contributes: a coherence goal is worth the
same at every moment, but it becomes urgent precisely when the error rate is
climbing.

The formula is a weighted sum, and every weight is a gene (§M5.3) rather than a
constant — the system is meant to discover how much urgency should outrank
value, not be told.

    priority = value·w_v + urgency·w_u + drive_pressure·w_d
             + aging·w_a + plan_ev·w_p − cost_norm·w_c
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.motivation.roi import DRIVES, normalize_cost

logger = logging.getLogger("aegis.priority")

#: Default weights (Appendix C, ``priority_w_*``).
DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 1.0,
    "urgency": 0.7,
    "drive": 0.5,
    "aging": 0.3,
    "plan": 0.8,
    "cost": 0.4,
}

#: Cost magnitude treated as "expensive". Normalising against a fixed scale
#: rather than against the most expensive candidate keeps a priority comparable
#: between ticks — otherwise a cheap tick would inflate every score in it.
COST_SCALE = 10.0


@dataclass
class Candidate:
    """Something that wants to happen, and what is known about it."""

    objective: str
    drive: str = "knowledge"
    value: float = 0.0
    plan_ev: float = 0.0
    cost: ResourceCost = field(default_factory=ResourceCost)
    safety_critical: bool = False
    payload: dict = field(default_factory=dict)
    priority: float = 0.0
    breakdown: dict = field(default_factory=dict)


class PriorityScheduler:
    """Turns candidates into an order."""

    def __init__(self, resources=None, goal_intelligence=None, roi=None,
                 weights: dict[str, float] | None = None):
        self.resources = resources
        self.goal_intelligence = goal_intelligence
        self.roi = roi
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.set_weights(weights)
        self.ordered_count = 0
        self.last_order: list[dict] = []

    def set_weights(self, weights: dict[str, float]) -> None:
        """Apply genome weights, ignoring anything unrecognised."""
        for name, value in (weights or {}).items():
            key = name[len("priority_w_"):] if name.startswith("priority_w_") else name
            if key in self.weights:
                try:
                    self.weights[key] = float(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring non-numeric priority weight %s=%r", name, value)

    # ── the components ───────────────────────────────────────────────

    @staticmethod
    def urgency(drive: str, ctx) -> float:
        """How much the current situation is pressing on this drive.

        This is the term that makes priority differ from value: a stability
        goal is worth the same at every moment, but it becomes urgent exactly
        when energy is draining, and a coherence goal when errors are climbing.
        """
        error_rate = _ctx_float(ctx, "error_rate", 0.0)
        energy = _ctx_float(ctx, "energy", 1.0)
        health_degraded = _ctx_str(ctx, "health_status", "healthy") != "healthy"

        if drive == "coherence":
            return min(1.0, error_rate * 5.0)
        if drive == "stability":
            pressure = max(0.0, 1.0 - energy)
            return min(1.0, pressure + (0.4 if health_degraded else 0.0))
        if drive == "competence":
            # Falling capability is urgent; steady capability is merely valuable.
            return min(1.0, max(0.0, -_ctx_float(ctx, "bench_trend", 0.0)) * 5.0)
        return min(1.0, _ctx_float(ctx, "curiosity", 0.0))

    def drive_pressure(self, drive: str) -> float:
        """Unmet demand for a drive — its budget share, as allocated by ROI.

        A drive the system has decided to fund is a drive whose work should be
        ordered earlier; using the same number for both keeps the allocation
        and the ordering from disagreeing with each other.
        """
        if self.roi is None:
            return 1.0 / len(DRIVES)
        return self.roi.share(drive)

    def aging(self, objective: str) -> float:
        if self.resources is None:
            return 0.0
        return self.resources.aging_bonus(objective)

    @staticmethod
    def cost_norm(cost: ResourceCost) -> float:
        return min(1.0, normalize_cost(cost) / COST_SCALE)

    def value_of(self, objective: str, ctx) -> float:
        if self.goal_intelligence is None:
            return 0.0
        try:
            return float(self.goal_intelligence.expected_value(objective, _as_dict(ctx)))
        except Exception:
            logger.exception("Value lookup failed for %s", objective)
            return 0.0

    # ── scoring ──────────────────────────────────────────────────────

    def priority(self, candidate: Candidate | str, ctx=None) -> float:
        """Score one candidate, recording how the number was reached."""
        if isinstance(candidate, str):
            candidate = Candidate(objective=candidate)

        weights = self.weights
        value = candidate.value if candidate.value else self.value_of(candidate.objective, ctx)
        urgency = self.urgency(candidate.drive, ctx)
        pressure = self.drive_pressure(candidate.drive)
        aged = self.aging(candidate.objective)
        cost = self.cost_norm(candidate.cost)

        score = (value * weights["value"]
                 + urgency * weights["urgency"]
                 + pressure * weights["drive"]
                 + aged * weights["aging"]
                 + candidate.plan_ev * weights["plan"]
                 - cost * weights["cost"])

        candidate.breakdown = {
            "value": round(value, 5), "urgency": round(urgency, 5),
            "drive_pressure": round(pressure, 5), "aging": round(aged, 5),
            "plan_ev": round(candidate.plan_ev, 5), "cost_norm": round(cost, 5),
            "weights": dict(weights),
        }
        candidate.priority = round(score, 6)
        return candidate.priority

    def order(self, candidates: list[Candidate], ctx=None) -> list[Candidate]:
        """Rank candidates, highest priority first.

        Safety-critical work sorts ahead of everything else regardless of score.
        It is not a very large bonus applied to the score — a bonus can always
        be out-weighed by a sufficiently attractive alternative, and "the health
        check lost on points" is not an outcome worth allowing.

        Ties break on the objective name so the order never depends on dict
        iteration (§3.1).
        """
        for candidate in candidates:
            self.priority(candidate, ctx)
        ordered = sorted(
            candidates,
            key=lambda c: (not c.safety_critical, -c.priority, c.objective),
        )
        self.ordered_count += 1
        self.last_order = [
            {"objective": c.objective, "priority": c.priority,
             "drive": c.drive, "safety_critical": c.safety_critical}
            for c in ordered[:8]
        ]
        return ordered

    def status(self) -> dict:
        return {
            "weights": dict(self.weights),
            "orderings": self.ordered_count,
            "last_order": list(self.last_order),
        }


# ── context access ───────────────────────────────────────────────────
# The scheduler is handed either a TickContext or a plain dict depending on the
# caller, so reads go through these rather than assuming one shape.

def _as_dict(ctx) -> dict:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return ctx
    return getattr(ctx, "metrics", {}) or {}


def _ctx_float(ctx, key: str, default: float) -> float:
    try:
        return float(_as_dict(ctx).get(key, default))
    except (TypeError, ValueError):
        return default


def _ctx_str(ctx, key: str, default: str) -> str:
    value = _as_dict(ctx).get(key, default)
    return str(value) if value is not None else default
