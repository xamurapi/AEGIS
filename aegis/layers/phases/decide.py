"""DECIDE — plans compared, gates applied in a fixed order (spec M2, Appendix J).

The order below is normative, not stylistic. Each gate assumes the ones before
it have run, and moving one is a hole in the guarantee rather than a
refactoring:

    1. collect objectives            7.  reserve resources — no lease, no action
    2. available actions             8.  optional cortex re-rank, top three only
    3. price them by the world model 9.  confidence penalty from known failures
    4. score                         10. ethics — last, and unarguable
    5. behaviour rules               11. self-preservation for self-modification
    6. priority                      12. open the experience and the forecast

The two properties worth stating plainly. **Ethics is last** so that nothing
can run after it and quietly undo its verdict; a gate that can be appealed is
not a gate. And **the forecast is opened before the action**, because a
prediction recorded afterwards is not a prediction and its error would teach
the model nothing.
"""
import logging

from aegis.config import MAX_RISK_CONFIDENCE_PENALTY, PLAN_ENABLED, PLAN_LOG_THRESHOLD, WORLD_MODEL_EVERY_N_TICKS
from aegis.event_bus import Event, Layer
from aegis.layers.motivation import Candidate
from aegis.layers.phases.common import _coerce_float, _coerce_int

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")

#: How many candidates may be tried when ethics blocks the first choice
#: (Appendix J, step 10). Bounded: a system that kept searching for something
#: the ethics core would permit is a system negotiating with its own veto.
MAX_ETHICS_RETRIES = 3


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "decide"
    substrate.goals.generate_goals({
        "tick": substrate.tick_count,
        "memory_size": substrate.memory.status()["total_memories"],
    })
    focus = substrate.goals.get_current_focus()
    greedy = focus["name"] if focus else "idle_exploration"
    alternatives = ["optimize_memory", "self_inspect", "explore_topic", "rest"]
    if substrate.active_archetype and \
            substrate.consciousness.mode in substrate.active_archetype.strategies:
        alternatives.append(f"archetype_{substrate.active_archetype.name}")

    _generate_meta_goals(substrate)

    decision, confidence, reasoning, plan, lease = await _choose(
        substrate, ctx, greedy, alternatives)

    ctx.decision = decision
    ctx.confidence = confidence
    ctx.plan = plan
    ctx.lease = lease
    ctx.action = plan.action if plan is not None else None

    # ── 12. the experience and the forecast, opened BEFORE the action ──
    _open_experience(substrate, ctx, focus, decision, plan)
    _record_prediction(substrate, ctx)
    _build_causal_chain(substrate, focus)

    substrate.introspection.trace_decision(decision, alternatives, reasoning, confidence)
    substrate.memory.add_working({
        "phase": "decide",
        "decision": decision,
        "reasoning": reasoning,
        "ethical_score": ctx.ethics_score,
        "ethical_status": ctx.ethics_status,
    })
    await substrate.event_bus.publish(Event(
        source=Layer.GOAL_ENGINE, target=Layer.ETHICS_CORE,
        event_type="decision",
        payload={"decision": decision, "ethics": {"status": ctx.ethics_status,
                                                  "score": ctx.ethics_score}},
    ))


# ── the choice ───────────────────────────────────────────────────────

async def _choose(substrate, ctx, greedy: str, alternatives: list[str]):
    """Run the gate sequence and return what survived it.

    Falls back to the pre-planner behaviour whenever planning cannot happen —
    no encoded state, planning switched off, nothing available. The system must
    keep deciding when a contour is unavailable, not stop.
    """
    plans = _build_plans(substrate, ctx)
    if not plans:
        return await _decide_without_a_plan(substrate, ctx, greedy, alternatives)

    # ── 5. behaviour rules (M3) ──────────────────────────────────────
    plans = _apply_rules(substrate, ctx, plans)
    if not plans:
        substrate.planner.note_blocked("policy")
        substrate.actions.note_blocked("policy")
        return await _decide_without_a_plan(substrate, ctx, greedy, alternatives)

    # ── 6. priority ──────────────────────────────────────────────────
    ordered = _prioritise(substrate, ctx, plans)

    # ── 7. resources: the first candidate that can be paid for wins ──
    chosen, lease = _reserve(substrate, ordered)
    if chosen is None:
        substrate.planner.note_blocked("resources")
        return await _decide_without_a_plan(substrate, ctx, greedy, alternatives)

    # ── 8. the cortex may reorder the shortlist, nothing more ────────
    chosen, lease = await _cortex_rerank(substrate, ctx, ordered, chosen, lease)

    # ── 9-11. confidence, ethics, self-preservation ──────────────────
    for _ in range(MAX_ETHICS_RETRIES):
        confidence, reasoning = _adjust_confidence(substrate, ctx, chosen)
        if _passes_gates(substrate, ctx, chosen, confidence):
            substrate.planner.record_choice(chosen, greedy)
            _log_if_it_changed_anything(substrate, chosen, ordered)
            return chosen.objective, confidence, reasoning, chosen, lease

        substrate.release(lease)
        remaining = [plan for plan in ordered if plan is not chosen]
        if not remaining:
            break
        chosen, lease = _reserve(substrate, remaining)
        if chosen is None:
            break
        ordered = remaining

    substrate.release(lease)
    return await _decide_without_a_plan(substrate, ctx, greedy, alternatives)


def _build_plans(substrate, ctx):
    """Steps 1-4: candidates, available actions, rollouts, scores."""
    if not PLAN_ENABLED or ctx.state is None:
        return []
    try:
        available = substrate.actions.available(substrate, ctx)
        if not available:
            return []
        return substrate.planner.build(substrate, ctx, available)
    except Exception:
        logger.exception("Planning failed — falling back to the direct choice")
        return []


def _apply_rules(substrate, ctx, plans):
    """Step 5. Suppression requires evidence, and never touches safety work."""
    policy = getattr(substrate, "policy", None)
    if policy is None:
        return plans
    try:
        return policy.apply_rules(ctx.state, plans, substrate.tick_count)
    except Exception:
        logger.exception("Applying behaviour rules failed — keeping every plan")
        return plans


def _prioritise(substrate, ctx, plans):
    """Step 6. Priority decides the order in which resources are asked for."""
    try:
        candidates = [
            Candidate(objective=plan.objective, drive=plan.drive,
                      plan_ev=plan.expected_value, cost=plan.expected_cost,
                      safety_critical=plan.safety_critical,
                      payload={"plan": plan})
            for plan in plans
        ]
        ordered = substrate.priority.order(candidates, ctx)
        return [candidate.payload["plan"] for candidate in ordered]
    except Exception:
        logger.exception("Prioritisation failed — keeping the planner's order")
        return plans


def _reserve(substrate, ordered):
    """Step 7. No lease, no action — that is what makes motivation binding."""
    for plan in ordered:
        if plan.action is None:
            continue
        lease = substrate.acquire(plan.action)
        if lease is not None:
            return plan, lease
    return None, None


async def _cortex_rerank(substrate, ctx, ordered, chosen, lease):
    """Step 8. A model may permute the shortlist; it may not extend it.

    The returned order is re-checked against the shortlist that was sent, so an
    index pointing anywhere else is discarded rather than followed. This is the
    single place a model touches the decision, and it is deliberately the
    narrowest one available.
    """
    shortlist = ordered[:3]
    if len(shortlist) < 2 or not substrate.llm.cortex.role_available("deep"):
        return chosen, lease
    if substrate._regulation_directives.get("skip_llm"):
        return chosen, lease

    try:
        from aegis.cortex import prompts
        from aegis.cortex.router import Role

        rendered = "\n".join(
            f"{index}: {plan.objective} via {plan.action} — {plan.rationale}"
            for index, plan in enumerate(shortlist))
        parsed = await substrate.llm.cortex.structured(
            Role.DEEP,
            [{"role": "system", "content": prompts.load("system")},
             {"role": "user", "content": prompts.render(
                 "plan_rerank", state=ctx.state.key(), candidates=rendered)}],
            "plan_rerank", lease=lease)
    except Exception:
        logger.exception("Cortex re-rank failed — keeping the planner's order")
        return chosen, lease

    if not parsed:
        return chosen, lease
    order = [index for index in parsed.get("order", [])
             if isinstance(index, int) and 0 <= index < len(shortlist)]
    if not order:
        return chosen, lease

    preferred = shortlist[order[0]]
    if preferred is chosen:
        return chosen, lease
    # A different preference still has to be paid for; the cortex cannot spend
    # what the resource manager has not granted.
    substrate.release(lease)
    new_lease = substrate.acquire(preferred.action) if preferred.action else None
    if new_lease is None:
        return _reserve(substrate, ordered)
    preferred.source = "planner+cortex"
    return preferred, new_lease


def _adjust_confidence(substrate, ctx, plan):
    """Step 9. Known failure history costs confidence, and confidence is what
    the ethics gate reads — so this is where causal memory earns its keep."""
    confidence = substrate._compute_confidence()
    reasoning = plan.rationale or f"Planned {plan.objective}"
    try:
        risks = substrate.world_model.risks_for([plan.objective, plan.action or ""])
        if risks:
            penalty = min(MAX_RISK_CONFIDENCE_PENALTY,
                          sum(r["failure_rate"] for r in risks) / 10)
            confidence = round(max(0.05, confidence * (1 - penalty)), 4)
            reasoning += (f" | {len(risks)} known failure mode(s) lower confidence "
                          f"by {round(penalty * 100)}%")
    except Exception:
        logger.exception("Risk-aware confidence adjustment failed")
    return confidence, reasoning


def _passes_gates(substrate, ctx, plan, confidence: float) -> bool:
    """Steps 10-11. Ethics, then self-preservation for anything self-modifying."""
    spec = substrate.actions.by_name.get(plan.action or "")
    modifies_self = bool(spec and not spec.reversible) or \
        (plan.action or "").endswith(("self_mod", "train_weights"))

    verdict = substrate.ethics.evaluate_action({
        "type": plan.objective,
        "confidence": confidence,
        "modifies_self": modifies_self,
    })
    ctx.ethics_status = verdict["status"]
    ctx.ethics_score = verdict["score"]
    if verdict["status"] == "blocked":
        substrate.planner.note_blocked("ethics")
        substrate.actions.note_blocked("ethics")
        return False

    if modifies_self:
        safe, _report = substrate.self_preservation.is_modification_safe(
            f"plan/{plan.action}", plan.objective)
        if not safe:
            substrate.planner.note_blocked("self_preservation")
            return False
    return True


def _log_if_it_changed_anything(substrate, chosen, ordered):
    """Record the plan only when it actually moved the decision (§M2.5).

    Logging every plan would bury the ones that mattered under the ones that
    agreed with the obvious choice.
    """
    runner_up = next((plan for plan in ordered if plan is not chosen), None)
    if runner_up is None:
        return
    if abs(chosen.score - runner_up.score) < PLAN_LOG_THRESHOLD:
        return
    substrate.autobiography.log_event(
        "planner",
        f"{chosen.objective} via {chosen.action} over {runner_up.objective} "
        f"({chosen.score:+.3f} vs {runner_up.score:+.3f})",
        0.5)


# ── the path taken when planning cannot happen ───────────────────────

async def _decide_without_a_plan(substrate, ctx, greedy: str, alternatives: list[str]):
    """The pre-planner decision, kept whole.

    Reached when there is no encoded state, planning is switched off, nothing
    is affordable, or every candidate was refused. A contour being unavailable
    must cost the system its judgement, never its ability to act.
    """
    decision = greedy
    confidence = substrate._compute_confidence()
    reasoning = f"Selected by priority and progress. Tick #{substrate.tick_count}"

    choice = None
    try:
        choice = substrate.goal_intelligence.choose(
            [decision] + alternatives, _value_context(substrate))
        if choice and choice.get("objective"):
            decision = choice["objective"]
    except Exception:
        logger.exception("Value-driven selection failed — keeping the heuristic pick")

    decision, confidence, reasoning = await _llm_decision(
        substrate, ctx, decision, alternatives, confidence, reasoning)

    try:
        risks = substrate.world_model.risks_for([decision, greedy])
        if risks:
            penalty = min(MAX_RISK_CONFIDENCE_PENALTY,
                          sum(r["failure_rate"] for r in risks) / 10)
            confidence = round(max(0.05, confidence * (1 - penalty)), 4)
            reasoning += (f" | {len(risks)} known failure mode(s) for this course "
                          f"of action lower confidence by {round(penalty * 100)}%")
    except Exception:
        logger.exception("Risk-aware confidence adjustment failed")

    verdict = substrate.ethics.evaluate_action({
        "type": decision, "confidence": confidence, "modifies_self": False,
    })
    ctx.ethics_status = verdict["status"]
    ctx.ethics_score = verdict["score"]
    return decision, confidence, reasoning, None, None


async def _llm_decision(substrate, ctx, decision, alternatives, confidence, reasoning):
    """The legacy free-choice path, still under a lease (§M4.3)."""
    lease = (substrate.acquire("curiosity_explore")
             if substrate._is_llm_tick()
             and not substrate._regulation_directives.get("skip_llm")
             else None)
    if lease is None:
        return decision, confidence, reasoning

    options = [decision] + alternatives
    ctx.mark_external("decide")
    result = await substrate.llm.make_decision(options, {
        "focus": substrate.goals.get_current_focus(),
        "tick": substrate.tick_count,
        "goals_summary": {g.name: g.progress for g in substrate.goals.goals
                          if g.status == "active" and g.level != "axiom"},
        "mood": substrate.emotions.mood,
        "consciousness_mode": substrate.consciousness.mode,
    }, lease=lease)
    substrate.settle(lease, value=1.0 if result.get("success") else 0.0)

    if result["success"] and isinstance(result.get("parsed"), dict):
        parsed = result["parsed"]
        chosen_index = _coerce_int(parsed.get("chosen", 1), 1) - 1
        if 0 <= chosen_index < len(options):
            decision = options[chosen_index]
        confidence = _coerce_float(parsed.get("confidence", confidence), confidence)
        reasoning = str(parsed.get("reasoning", reasoning))
        await substrate.event_bus.publish(Event(
            source=Layer.GOAL_ENGINE, target=None, event_type="llm_decision",
            payload={"decision": decision, "reasoning": reasoning[:100]}))
    return decision, confidence, reasoning


# ── the pieces every path shares ─────────────────────────────────────

def _value_context(substrate) -> dict:
    total = substrate.health.successful_ticks + substrate.health.failed_ticks
    return {
        "tick": substrate.tick_count,
        "energy": substrate.emotions.energy,
        "error_rate": substrate.health.error_count / max(total, 1) if total else 0.0,
        "curiosity": substrate.goals.curiosity_level,
    }


def _generate_meta_goals(substrate) -> None:
    if substrate.tick_count % 30 != 0:
        return
    try:
        total = substrate.health.successful_ticks + substrate.health.failed_ticks
        for proposal in substrate.meta_goals.generate_goals({
            "memory_total": substrate.memory.status()["total_memories"],
            "mood_valence": substrate.emotions.valence,
            "learning_sessions": substrate.external_learning.learning_sessions,
            "avg_tick_ms": substrate.last_tick_duration * 1000,
            "tick": substrate.tick_count,
            "error_rate": substrate.health.error_count / max(total, 1) if total else 0,
            "active_agents": sum(1 for a in substrate.agent_system.agents
                                 if a.status in ("active", "deployed")),
        }):
            substrate.autobiography.log_event(
                "meta_goal", proposal["description"][:60], 0.5)
    except Exception:
        logger.exception("Meta-goal generation failed")


def _open_experience(substrate, ctx, focus, decision, plan) -> None:
    """Credit the choice, and open the experience it will be judged by."""
    try:
        # The planner chose, so the planner's pick is what reward must be
        # credited to — the value table's own argmax is no longer what happened.
        substrate.goal_intelligence.commit(decision, _value_context(substrate))
    except Exception:
        logger.exception("Recording the value-driven choice failed")

    try:
        focus_name = focus["name"] if focus else "idle_exploration"
        total = substrate.health.successful_ticks + substrate.health.failed_ticks
        error_rate = substrate.health.error_count / max(total, 1) if total else 0.0
        substrate._pending_experiences["decide"] = substrate.feedback_loop.record_situation(
            situation=f"focus={focus_name} mode={substrate.consciousness.mode} "
                      f"energy={substrate.emotions.energy:.2f} err={error_rate:.2f}",
            decision=decision,
            context={"tick": substrate.tick_count,
                     "value": plan.expected_value if plan else None,
                     "action": plan.action if plan else None},
        )
    except Exception:
        logger.exception("Opening the experience failed")


def _record_prediction(substrate, ctx) -> None:
    """The forecast, keyed on the ACTION rather than the objective.

    What the world model predicts about is what will actually be done; an
    objective is a wish and has no outcome of its own.
    """
    if ctx.state is None:
        return
    subject = ctx.action or ctx.decision
    if not subject:
        return
    try:
        ctx.prediction = substrate.world_model.make_prediction(
            ctx.state, subject, substrate.tick_count)
    except Exception:
        logger.exception("Recording the prediction failed")


def _build_causal_chain(substrate, focus) -> None:
    if substrate.tick_count % max(1, WORLD_MODEL_EVERY_N_TICKS) != 0 or not focus:
        return
    try:
        constraints = [g.name for g in substrate.goals.goals if g.level == "axiom"]
        chain = substrate.world_model.build_chain(focus["name"], constraints)
        if chain["plan"]:
            substrate.autobiography.log_event(
                "world_model",
                f"Chain for {focus['name']}: {len(chain['plan'])} steps, "
                f"conf={chain['confidence']}",
                0.4)
    except Exception:
        logger.exception("Higher-systems DECIDE hook failed")
