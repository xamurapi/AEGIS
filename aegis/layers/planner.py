"""The planner: decisions made by comparing plans (spec M2).

    прогноз → решение

Before this, a decision was "take the goal with the highest
``priority·(1−progress)``". That is a ranking of *wishes*; nothing in it
consulted what the system had learned about what actually works. The world
model (M1) can now answer "what is this likely to be worth", so a decision
becomes a comparison of plans priced by that model.

The score is a weighted sum, and every weight is a gene (§M5.3):

    score = EV·w_ev + value·w_val + explore·w_exp − cost·w_cost − risk·w_risk
            + policy_delta

Two properties this design is built around:

**The planner proposes; the gates dispose.** Nothing here can approve an
action. Ordering, resource leases, the behaviour policy, ethics and
self-preservation are applied *after* planning, in the order Appendix J fixes,
and none of them can be argued with. The planner's only power is to say what
looks best.

**The cortex may re-rank, never invent.** A model can reorder the top three
candidates and explain why. It cannot introduce an action outside the registry,
change a weight, or skip a gate — the shortlist it is given is the only thing
it may permute, and the result is re-checked against that shortlist.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.motivation.resources import ResourceCost
from aegis.layers.motivation.roi import normalize_cost
from aegis.util.stats import exponential_smooth

logger = logging.getLogger("aegis.planner")

#: Default scoring weights (Appendix C). Genes, not constants: how much
#: expected value should outrank curiosity is something to discover, not decree.
DEFAULT_WEIGHTS: dict[str, float] = {
    "ev": 1.0, "val": 0.6, "exp": 0.3, "cost": 0.4, "risk": 0.5,
}

#: Cost magnitude treated as "expensive", so a score stays comparable between
#: ticks rather than being normalised against whatever else was on offer.
COST_SCALE = 10.0

#: Weight of the smoothed gap between promised and realised value.
GAP_ALPHA = 0.05


@dataclass
class Plan:
    """One candidate course of action, priced by the world model."""

    objective: str
    steps: list[str] = field(default_factory=list)
    expected_value: float = 0.0
    expected_cost: ResourceCost = field(default_factory=ResourceCost)
    risk: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    source: str = "planner"
    alternatives: list[dict] = field(default_factory=list)
    drive: str = "knowledge"
    safety_critical: bool = False
    score: float = 0.0
    breakdown: dict = field(default_factory=dict)

    @property
    def action(self) -> str | None:
        """The step that would happen now — what everything downstream keys on."""
        return self.steps[0] if self.steps else None

    def as_dict(self) -> dict:
        return {
            "objective": self.objective,
            "steps": list(self.steps),
            "action": self.action,
            "expected_value": round(self.expected_value, 5),
            "expected_cost": self.expected_cost.as_dict(),
            "risk": round(self.risk, 5),
            "confidence": round(self.confidence, 5),
            "score": round(self.score, 5),
            "rationale": self.rationale,
            "source": self.source,
            "drive": self.drive,
            "safety_critical": self.safety_critical,
            "alternatives": list(self.alternatives),
            "breakdown": dict(self.breakdown),
        }


class Planner:
    """Builds and ranks plans. Decides nothing on its own."""

    def __init__(self, world_model, actions, goal_intelligence=None,
                 policy=None, telemetry=None, weights: dict | None = None):
        self.world_model = world_model
        self.actions = actions
        self.goal_intelligence = goal_intelligence
        #: Set from stage 5. Until then rules contribute nothing, which is the
        #: correct behaviour for a policy that has learned nothing yet.
        self.policy = policy
        self.telemetry = telemetry

        self.weights = dict(DEFAULT_WEIGHTS)
        self.depth = cfg.WM_DEPTH
        self.beam = cfg.WM_BEAM
        self.discount = cfg.WM_DISCOUNT
        if weights:
            self.set_weights(weights)

        self.plans_built = 0
        self.overrides = 0
        self.decisions = 0
        self.ev_gap: float | None = None
        self.last_latency_ms = 0.0
        self.blocked: dict[str, int] = {}
        self.last_plan: Plan | None = None

    # ── genome ───────────────────────────────────────────────────────

    def set_weights(self, genome: dict) -> None:
        """Adopt evolved scoring weights and search shape (Appendix C)."""
        mapping = {"w_ev": "ev", "w_val": "val", "w_exp": "exp",
                   "w_cost": "cost", "w_risk": "risk"}
        for gene, key in mapping.items():
            if gene in (genome or {}):
                try:
                    self.weights[key] = float(genome[gene])
                except (TypeError, ValueError):
                    logger.debug("Ignoring unusable planner weight %s", gene)
        for gene, attribute, caster in (("plan_depth", "depth", int),
                                        ("plan_beam", "beam", int),
                                        ("plan_discount", "discount", float)):
            if gene in (genome or {}):
                try:
                    setattr(self, attribute, caster(genome[gene]))
                except (TypeError, ValueError):
                    logger.debug("Ignoring unusable planner gene %s", gene)

    # ── candidates ───────────────────────────────────────────────────

    def collect_objectives(self, substrate, ctx=None) -> list[str]:
        """What the system could pursue this tick (Appendix J, step 1).

        Active goals, meta-goal proposals and whatever the knowledge graph
        connects to the current focus. Sorted and de-duplicated, because the
        order candidates arrive in must not decide anything (§3.1).
        """
        found: set[str] = set()
        try:
            for goal in substrate.goals.goals:
                if goal.status == "active" and goal.level != "axiom":
                    found.add(goal.name)
        except Exception:
            logger.exception("Reading active goals failed")

        try:
            focus = substrate.goals.get_current_focus()
            if focus:
                found.add(focus["name"])
                for related in substrate.cognitive_graph.related(focus["name"]):
                    if related.get("type") == "concept" and related.get("node"):
                        found.add(str(related["node"])[:60])
        except Exception:
            logger.exception("Reading graph-related objectives failed")

        try:
            # MetaGoalGenerator keeps its proposals on two attributes: the ones
            # currently pursued (active_meta_goals) and the rolling history
            # (generated_goals). This block used to read a `.goals` attribute
            # that never existed, so the AttributeError was swallowed below and
            # meta-goals silently never reached the shortlist. Prefer the
            # active set; fall back to the history when it is empty.
            meta = substrate.meta_goals
            proposals = (list(getattr(meta, "active_meta_goals", None) or ())
                         or list(getattr(meta, "generated_goals", None) or ()))
            for proposal in proposals[-5:]:
                name = proposal.get("name") or proposal.get("description", "")
                if name:
                    found.add(str(name)[:60])
        except Exception:
            # logger.exception, not debug: a debug-level message is how the
            # missing attribute above went unnoticed in the first place.
            logger.exception("Reading meta-goal proposals failed")

        # Always keep the fallback: a system with no goals still has to decide
        # something, and "idle" is an honest answer rather than an empty one.
        found.add("idle_exploration")
        return sorted(found)[:max(1, cfg.PLAN_MAX_CANDIDATES)]

    def actions_for(self, objective: str, available: list) -> list:
        """Which available actions serve an objective.

        Matched by drive: an objective about knowledge is served by the actions
        that pursue knowledge. Safety-critical work is deliberately *not* folded
        into every objective's set — being unsuppressable and having a reserved
        budget makes it protected, not universally relevant, and including it
        everywhere would let a checkpoint win a tie for an objective about
        reasoning.

        The fallback to everything matters: an objective whose drive nothing
        currently serves must still produce a plan, or it would silently drop
        out of consideration altogether.
        """
        drive = self.drive_of(objective)
        matched = [spec for spec in available if spec.drive == drive]
        return matched or list(available)

    def drive_of(self, objective: str) -> str:
        if self.goal_intelligence is None:
            return "knowledge"
        try:
            return self.goal_intelligence._classify_drive(objective)
        except Exception:
            return "knowledge"

    # ── building one plan ────────────────────────────────────────────

    def plan_for(self, objective: str, ctx, available: list) -> Plan | None:
        """Price one objective by rolling the world model forward."""
        specs = self.actions_for(objective, available)
        if not specs:
            return None
        by_name = {spec.name: spec for spec in specs}

        result = self.world_model.rollout(
            ctx.state, sorted(by_name), depth=self.depth, beam=self.beam)
        if not result.sequence:
            return None

        first = result.sequence[0]
        spec = by_name.get(first)
        cost = ResourceCost()
        for step in result.sequence:
            step_spec = by_name.get(step)
            if step_spec is not None:
                cost = cost + step_spec.cost

        plan = Plan(
            objective=objective,
            steps=list(result.sequence),
            expected_value=result.value,
            expected_cost=cost,
            risk=self.risk_of(objective, ctx, first),
            confidence=self.confidence_of(ctx, first),
            drive=spec.drive if spec else self.drive_of(objective),
            safety_critical=bool(spec and spec.safety_critical),
        )
        self.plans_built += 1
        return plan

    def risk_of(self, objective: str, ctx, action: str) -> float:
        """How badly this could go, on 0..1.

        Two independent sources, because they know different things: the causal
        memory of what has failed around this topic, and the spread of the
        reward the action itself returns. An action that averages well but
        varies wildly is not the same proposition as a steady one.
        """
        risk = 0.0
        try:
            known_failures = self.world_model.risks_for([objective, action])
            if known_failures:
                risk += min(0.6, sum(r["failure_rate"] for r in known_failures) / 10)
        except Exception:
            logger.exception("Risk lookup failed for %s", objective)
        try:
            outcome = self.world_model.predict_outcome(ctx.state, action)
            risk += min(0.4, outcome.reward_sd * (1.0 - outcome.p_success_pessimistic))
        except Exception:
            logger.exception("Outcome risk lookup failed for %s", action)
        return min(1.0, risk)

    def confidence_of(self, ctx, action: str) -> float:
        """How much the model trusts its own estimate for this action."""
        try:
            return round(self.world_model.knows(ctx.state, action), 4)
        except Exception:
            return 0.0

    # ── scoring ──────────────────────────────────────────────────────

    def score(self, plan: Plan, ctx) -> float:
        """The weighted sum of Appendix J, step 4."""
        weights = self.weights
        value = self.value_of(plan.objective, ctx)
        explore = self.explore_bonus(ctx, plan.action)
        cost = min(1.0, normalize_cost(plan.expected_cost) / COST_SCALE)
        delta = self.policy_delta(ctx, plan.action)

        total = (plan.expected_value * weights["ev"]
                 + value * weights["val"]
                 + explore * weights["exp"]
                 - cost * weights["cost"]
                 - plan.risk * weights["risk"]
                 + delta)

        plan.breakdown = {
            "expected_value": round(plan.expected_value, 5),
            "value": round(value, 5),
            "explore": round(explore, 5),
            "cost_norm": round(cost, 5),
            "risk": round(plan.risk, 5),
            "policy_delta": round(delta, 5),
            "weights": dict(weights),
        }
        plan.score = round(total, 6)
        plan.rationale = self.explain(plan)
        return plan.score

    def value_of(self, objective: str, ctx) -> float:
        if self.goal_intelligence is None:
            return 0.0
        try:
            return float(self.goal_intelligence.expected_value(
                objective, self._context_metrics(ctx)))
        except Exception:
            logger.exception("Value lookup failed for %s", objective)
            return 0.0

    def explore_bonus(self, ctx, action: str | None) -> float:
        """How much there is left to learn here — 1 for the entirely untried.

        This is what makes exploration *directed*: the bonus points at the
        specific state/action pairs the model cannot yet predict, rather than
        at novelty in general.
        """
        if action is None:
            return 0.0
        try:
            return 1.0 - self.world_model.knows(ctx.state, action)
        except Exception:
            return 0.0

    def policy_delta(self, ctx, action: str | None) -> float:
        """What learned behaviour rules add or subtract (M3).

        Zero until stage 5 attaches a policy — which is the honest value for
        rules that do not exist yet, not a placeholder.
        """
        if self.policy is None or action is None:
            return 0.0
        try:
            return float(self.policy.delta(ctx.state, action))
        except Exception:
            logger.exception("Policy delta lookup failed for %s", action)
            return 0.0

    @staticmethod
    def _context_metrics(ctx) -> dict:
        return dict(getattr(ctx, "state_inputs", {}) or {})

    # ── the whole shortlist ──────────────────────────────────────────

    def build(self, substrate, ctx, available: list) -> list[Plan]:
        """Every candidate plan, best first (Appendix J, steps 1–4).

        Ties break on the objective name so two identical runs rank identically
        (§3.1) — a planner whose order depended on set iteration would make
        every A/B comparison meaningless.
        """
        started = CLOCK.monotonic()
        plans: list[Plan] = []
        for objective in self.collect_objectives(substrate, ctx):
            plan = self.plan_for(objective, ctx, available)
            if plan is None:
                continue
            self.score(plan, ctx)
            plans.append(plan)

        plans.sort(key=lambda p: (-p.score, p.objective))
        for plan in plans:
            plan.alternatives = [
                {"objective": other.objective, "action": other.action,
                 "expected_value": round(other.expected_value, 5),
                 "score": round(other.score, 5)}
                for other in plans if other is not plan
            ][:3]
        self.last_latency_ms = (CLOCK.monotonic() - started) * 1000
        return plans

    # ── explaining ───────────────────────────────────────────────────

    def explain(self, plan: Plan) -> str:
        """Why this plan scored what it did, deterministically.

        Assembled from the same numbers that produced the score, so the
        explanation cannot drift from the decision it explains.
        """
        parts = [f"{plan.objective} via {plan.action or 'nothing'}"]
        breakdown = plan.breakdown
        if not breakdown:
            return parts[0]

        contributions = {
            "expected value": breakdown["expected_value"] * breakdown["weights"]["ev"],
            "learned value": breakdown["value"] * breakdown["weights"]["val"],
            "room to learn": breakdown["explore"] * breakdown["weights"]["exp"],
            "cost": -breakdown["cost_norm"] * breakdown["weights"]["cost"],
            "risk": -breakdown["risk"] * breakdown["weights"]["risk"],
            "learned rules": breakdown["policy_delta"],
        }
        ranked = sorted(contributions.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        driver = ", ".join(f"{name} {value:+.3f}" for name, value in ranked[:3]
                           if abs(value) > 1e-9)
        parts.append(f"score {plan.score:+.3f} ({driver or 'no clear driver'})")
        if plan.confidence < 0.5:
            parts.append("on thin evidence")
        return "; ".join(parts)

    def evaluate(self, ctx, sequence: list[str]) -> float:
        """Price a sequence somebody else proposed, by the same yardstick."""
        try:
            return self.world_model.evaluate_sequence(ctx.state, sequence)
        except Exception:
            logger.exception("Evaluating a proposed sequence failed")
            return 0.0

    # ── measuring whether it helped ──────────────────────────────────

    def record_choice(self, chosen: Plan, greedy_objective: str | None) -> None:
        """Note whether the planner actually changed the decision.

        ``planner_override_rate`` is the honest answer to "is this contour
        doing anything": a planner that always agrees with the greedy pick has
        added latency and nothing else.
        """
        self.decisions += 1
        if greedy_objective is not None and chosen.objective != greedy_objective:
            self.overrides += 1
        self.last_plan = chosen

    def record_outcome(self, plan: Plan, realised: float) -> None:
        """Close the loop on a plan's promise.

        The gap between promised and realised value is what says whether the
        model underneath the planner is learning; §M2.8 requires it to fall.
        """
        gap = abs(plan.expected_value - float(realised))
        self.ev_gap = exponential_smooth(self.ev_gap, gap, GAP_ALPHA)

    def note_blocked(self, reason: str) -> None:
        self.blocked[reason] = self.blocked.get(reason, 0) + 1

    def override_rate(self) -> float:
        return round(self.overrides / self.decisions, 4) if self.decisions else 0.0

    # ── reporting ────────────────────────────────────────────────────

    def publish_metrics(self, tick: int) -> None:
        if self.telemetry is None:
            return
        from aegis.telemetry import metrics as M
        try:
            self.telemetry.record(M.PLAN_OVERRIDE_RATE, self.override_rate(), tick)
            self.telemetry.record(M.PLAN_LATENCY_MS, self.last_latency_ms, tick)
            if self.ev_gap is not None:
                self.telemetry.record(M.PLAN_EV_GAP, self.ev_gap, tick)
            if self.last_plan is not None:
                self.telemetry.record(M.PLAN_DEPTH, len(self.last_plan.steps), tick)
            for reason in ("policy", "resources", "ethics", "self_preservation"):
                self.telemetry.record(M.PLAN_BLOCKED, self.blocked.get(reason, 0),
                                      tick, tags={"reason": reason})
        except Exception:
            logger.exception("Planner metric publication failed")

    def last_plan_report(self) -> dict:
        """The last plan and what it beat — the panel of §M10.1.

        Separate from :meth:`status` because the plan panel wants the losing
        alternatives and their values, and a status dict that always carried
        the whole shortlist would put it in every WebSocket frame.
        """
        if self.last_plan is None:
            return {"plan": None, "alternatives": [], "explanation": "",
                    "latency_ms": round(self.last_latency_ms, 3)}
        return {
            "plan": self.last_plan.as_dict(),
            "alternatives": list(self.last_plan.alternatives),
            "explanation": self.explain(self.last_plan),
            "latency_ms": round(self.last_latency_ms, 3),
            "override_rate": self.override_rate(),
            "ev_gap": round(self.ev_gap, 5) if self.ev_gap is not None else None,
            "blocked": dict(sorted(self.blocked.items())),
        }

    def status(self) -> dict:
        return {
            "enabled": cfg.PLAN_ENABLED,
            "weights": dict(self.weights),
            "depth": self.depth,
            "beam": self.beam,
            "discount": self.discount,
            "plans_built": self.plans_built,
            "decisions": self.decisions,
            "override_rate": self.override_rate(),
            "ev_gap": round(self.ev_gap, 5) if self.ev_gap is not None else None,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "blocked": dict(sorted(self.blocked.items())),
            "last_plan": self.last_plan.as_dict() if self.last_plan else None,
        }
