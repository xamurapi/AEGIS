"""DECIDE phase — objective selection, risk-aware confidence, ethics gate.

Extracted verbatim from ``Substrate._decide`` (spec §3.9). The substrate had
grown to 1763 lines with seven more systems still to land; each phase now lives
where it can be read and tested on its own. Behaviour is unchanged — the bodies
were moved, not rewritten.
"""
import logging

from aegis.config import MAX_RISK_CONFIDENCE_PENALTY, WORLD_MODEL_EVERY_N_TICKS
from aegis.event_bus import Event, Layer
from aegis.layers.phases.common import _coerce_float, _coerce_int

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "decide"
    new_goals = substrate.goals.generate_goals({
        "tick": substrate.tick_count,
        "memory_size": substrate.memory.status()["total_memories"],
    })
    focus = substrate.goals.get_current_focus()
    decision = focus["name"] if focus else "idle_exploration"
    alternatives = ["optimize_memory", "self_inspect", "explore_topic", "rest"]

    # Archetype-influenced decision
    if substrate.active_archetype and substrate.consciousness.mode in substrate.active_archetype.strategies:
        alternatives.append(f"archetype_{substrate.active_archetype.name}")

    # Real confidence from system state
    confidence = substrate._compute_confidence()
    reasoning = f"Selected based on priority and progress. Tick #{substrate.tick_count}"

    # Meta-goal generation (every 30 ticks)
    if substrate.tick_count % 30 == 0:
        total_ticks = substrate.health.successful_ticks + substrate.health.failed_ticks
        meta_ctx = {
            "memory_total": substrate.memory.status()["total_memories"],
            "mood_valence": substrate.emotions.valence,
            "learning_sessions": substrate.external_learning.learning_sessions,
            "avg_tick_ms": substrate.last_tick_duration * 1000,
            "tick": substrate.tick_count,
            "error_rate": substrate.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0,
            "active_agents": sum(1 for a in substrate.agent_system.agents if a.status in ("active", "deployed")),
        }
        new_meta = substrate.meta_goals.generate_goals(meta_ctx)
        for mg in new_meta:
            substrate.autobiography.log_event("meta_goal", mg["description"][:60], 0.5)

    # ── System 4: value-driven selection ─────────────────────────
    # Learned utility picks the objective BEFORE the decision is final.
    # This used to run after ethics had already judged `decision`, so the
    # choice was recorded as a number and steered nothing: the system
    # learned a utility it never acted on. It runs ahead of the LLM block
    # so a hosted model can still override, and it stands alone on the
    # ticks where no LLM call is made. Guarded — motivation must never be
    # able to abort a tick.
    _tt = substrate.health.successful_ticks + substrate.health.failed_ticks
    error_rate = substrate.health.error_count / max(_tt, 1) if _tt > 0 else 0.0
    gi_ctx = {
        "tick": substrate.tick_count,
        "energy": substrate.emotions.energy,
        "error_rate": error_rate,
        "curiosity": substrate.goals.curiosity_level,
    }
    gi_choice = None
    try:
        gi_choice = substrate.goal_intelligence.choose([decision] + alternatives, gi_ctx)
        if gi_choice and gi_choice.get("objective"):
            decision = gi_choice["objective"]
    except Exception:
        logger.exception("Value-driven selection failed — keeping heuristic decision")

    # LLM-powered decision making
    if substrate._is_llm_tick() and not substrate._regulation_directives.get("skip_llm"):
        options = [decision] + alternatives
        ctx.mark_external("decide")
        result = await substrate.llm.make_decision(options, {
            "focus": focus,
            "tick": substrate.tick_count,
            "goals_summary": {g.name: g.progress for g in substrate.goals.goals
                              if g.status == "active" and g.level != "axiom"},
            "mood": substrate.emotions.mood,
            "consciousness_mode": substrate.consciousness.mode,
        })
        if result["success"] and "parsed" in result and isinstance(result["parsed"], dict):
            parsed = result["parsed"]
            # Coerce defensively — "chosen"/"confidence" may arrive as
            # strings or non-numbers (audit M5).
            chosen_idx = _coerce_int(parsed.get("chosen", 1), 1) - 1
            if 0 <= chosen_idx < len(options):
                decision = options[chosen_idx]
            confidence = _coerce_float(parsed.get("confidence", confidence), confidence)
            reasoning = str(parsed.get("reasoning", reasoning))

            await substrate.event_bus.publish(Event(
                source=Layer.GOAL_ENGINE, target=None,
                event_type="llm_decision",
                payload={"decision": decision, "reasoning": reasoning[:100]}
            ))

    # ── System 1: known failure history costs confidence ─────────
    # The World Model observed which causes tend to fail. That memory was
    # written every tick and read by nothing, so a course of action with a
    # proven failure history was proposed with the same confidence as an
    # untried one. Confidence feeds both the introspection trace and the
    # ethics gate below, so this is where the causal memory earns its keep.
    try:
        focus_for_risk = focus["name"] if focus else "idle_exploration"
        risks = substrate.world_model.risks_for([decision, focus_for_risk])
        if risks:
            penalty = min(MAX_RISK_CONFIDENCE_PENALTY,
                          sum(r["failure_rate"] for r in risks) / 10)
            confidence = round(max(0.05, confidence * (1 - penalty)), 4)
            reasoning += (f" | {len(risks)} known failure mode(s) for this course "
                          f"of action lower confidence by {round(penalty * 100)}%")
    except Exception:
        logger.exception("Risk-aware confidence adjustment failed")

    trace = substrate.introspection.trace_decision(
        decision, alternatives, reasoning, confidence
    )

    eth_result = substrate.ethics.evaluate_action({
        "type": decision,
        "confidence": confidence,
        "modifies_self": False,
    })

    substrate.memory.add_working({
        "phase": "decide",
        "decision": decision,
        "reasoning": reasoning,
        "ethical_score": eth_result["score"],
        "ethical_status": eth_result["status"],
    })

    await substrate.event_bus.publish(Event(
        source=Layer.GOAL_ENGINE, target=Layer.ETHICS_CORE,
        event_type="decision", payload={"decision": decision, "ethics": eth_result}
    ))

    # ── Higher systems (5 & 1): experience opening and a causal chain for
    #    the objective. Deterministic and cheap; guarded so a failure here
    #    cannot abort the tick. (System 4 now runs before the decision is
    #    final — see the value-driven selection block above.) ──
    try:
        focus_name = focus["name"] if focus else "idle_exploration"

        # System 5: open an experience for this decision — it will be closed
        # in REFLECT with the tick's realized reward and inferred cause.
        substrate._pending_experiences["decide"] = substrate.feedback_loop.record_situation(
            situation=f"focus={focus_name} mode={substrate.consciousness.mode} "
                      f"energy={substrate.emotions.energy:.2f} err={error_rate:.2f}",
            decision=decision,
            context={"tick": substrate.tick_count,
                     "value": gi_choice["expected_value"] if gi_choice else None},
        )

        # System 1: build a causal chain (objective -> constraints -> risks
        # -> plan -> expected result) for the current focus.
        if substrate.tick_count % max(1, WORLD_MODEL_EVERY_N_TICKS) == 0 and focus:
            constraints = [g.name for g in substrate.goals.goals if g.level == "axiom"]
            chain = substrate.world_model.build_chain(focus_name, constraints)
            if chain["plan"]:
                substrate.autobiography.log_event(
                    "world_model",
                    f"Chain for {focus_name}: {len(chain['plan'])} steps, "
                    f"conf={chain['confidence']}",
                    0.4,
                )
    except Exception:
        logger.exception("Higher-systems DECIDE hook failed")
