"""EVALUATE phase — introspection, goal progress, regulation, LLM appraisal.

Extracted verbatim from ``Substrate._evaluate`` (spec §3.9). The substrate had
grown to 1763 lines with seven more systems still to land; each phase now lives
where it can be read and tested on its own. Behaviour is unchanged — the bodies
were moved, not rewritten.
"""
import logging

from aegis.event_bus import Event, Layer

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")


def close_previous_prediction(substrate, ctx: TickContext) -> None:
    """Score the forecast the last tick made, now that its outcome is known.

    The successor state is *this* tick's state — that is the whole reason the
    scoring happens here rather than at the end of the previous tick: where an
    action led cannot be observed until the world has moved.

    The error this produces is the model's only learning signal, and its
    surprise is what points curiosity somewhere specific instead of everywhere
    (§M1.5).
    """
    pending = substrate._pending_prediction
    if not pending or ctx.state is None:
        return
    substrate._pending_prediction = None

    try:
        substrate.world_model.observe_transition(
            pending["state"], pending["action"], ctx.state)
        score = substrate.world_model.score_prediction(
            pending["id"], pending["success"], pending["reward"], ctx.state)
    except Exception:
        logger.exception("Closing the previous prediction failed")
        return

    if score is None:
        return
    ctx.prediction_score = score.as_dict()

    # Surprise steers curiosity: repeatedly being wrong about where actions
    # lead is exactly the signal that there is something here worth learning.
    try:
        surprise = substrate.world_model.surprise()
        substrate.goals.curiosity_level = max(
            0.0, min(1.0, 0.5 * substrate.goals.curiosity_level
                     + 0.5 * min(1.0, surprise / 4.0)))
    except Exception:
        logger.exception("Feeding surprise into curiosity failed")


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "evaluate"
    close_previous_prediction(substrate, ctx)

    # Real system metrics for introspection
    total_ticks = substrate.health.successful_ticks + substrate.health.failed_ticks
    error_rate = substrate.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
    active_goals = [g for g in substrate.goals.goals if g.status == "active" and g.level != "axiom"]
    mem_status = substrate.memory.status()

    system_metrics = {
        "memory_load": mem_status["working_memory_size"] / max(1, mem_status["working_memory_max"]),
        "goal_pressure": len(active_goals) / 10.0,
        "ethics_load": substrate.ethics.total_checked / max(1, substrate.tick_count),
        "energy": substrate.emotions.energy,
        "information_gain": substrate.goals.information_gain,
        "error_rate": error_rate,
        "llm_active": substrate.llm_thinking,
        "tick": substrate.tick_count,
    }
    activations = substrate.introspection.inspect_activations("main", system_metrics)

    # Goal progress with real metrics
    goal_metrics = {
        "new_concepts": substrate._tick_new_concepts,
        "new_episodic": substrate._tick_new_episodic,
        "error_rate": error_rate,
        "energy": substrate.emotions.energy,
        "llm_insights": substrate._tick_llm_insights,
    }
    substrate.goals.evaluate_progress(goal_metrics)
    focus = substrate.goals.get_current_focus()

    bias_report = None
    if substrate.tick_count % 20 == 0 and substrate.introspection.decision_trace:
        bias_report = substrate.introspection.detect_bias(substrate.introspection.decision_trace[-20:])

    # Value system evaluation
    focus_name = focus["name"] if focus else "idle"
    substrate.values.evaluate_action(substrate.emotions.mood, substrate.emotions.success_rate, focus_name)

    # Health check
    health_report = substrate.health.check()
    if health_report["status"] == "critical":
        substrate.autobiography.log_event("health", f"Critical health: {health_report['critical']}", 0.9)

    # Meta-regulation
    reg = substrate.meta_regulation.regulate(
        substrate.emotions.energy, health_report["status"],
        substrate.health.consecutive_errors, substrate.consciousness.mode,
    )
    substrate._regulation_directives = reg["directives"]
    if reg["directives"]["force_recharge"] > 0:
        substrate.emotions.recharge(reg["directives"]["force_recharge"])
    if reg["directives"]["reduce_sensors"]:
        substrate.sensors.reduce_sensors(True)
    else:
        substrate.sensors.reduce_sensors(False)

    # Meta-consciousness (every 25 ticks)
    if substrate.tick_count % 25 == 0:
        mc = substrate.meta_consciousness.evaluate(
            substrate.consciousness.mode,
            substrate.active_archetype.name if substrate.active_archetype else None,
            substrate.emotions.mood, substrate.emotions.energy,
            focus_name, substrate.archetypes_list,
        )
        if mc["fragmentation"] > 0.5:
            substrate.autobiography.log_event("meta", f"High fragmentation: {mc['fragmentation']:.2f}", 0.7)

    # LLM-powered state evaluation
    llm_eval = None
    lease = (substrate.acquire("evaluate_state_llm")
             if substrate._is_llm_tick()
             and not substrate._regulation_directives.get("skip_llm")
             else None)
    if lease is not None:
        substrate.llm_thinking = True
        # RAG: pull concepts RELEVANT to the current focus, not just recent ones.
        focus_query = (focus.get("name", "") + " " + focus.get("description", "")) if focus else ""
        relevant = substrate.memory.retrieve(focus_query, k=6) if focus_query.strip() else []
        relevant_concepts = [r["concept"] for r in relevant] or list(substrate.memory.semantic.keys())[-10:]
        compact_state = {
            "tick": substrate.tick_count,
            "goals_active": len(active_goals),
            "current_focus": focus,
            "memory_total": mem_status["total_memories"],
            "episodic_recent": [e["event"] for e in substrate.memory.episodic[-3:]],
            "relevant_concepts": relevant_concepts,
            "semantic_concepts": list(substrate.memory.semantic.keys())[-10:],
            "version": substrate.self_mod.current_version,
            "curiosity": round(substrate.goals.curiosity_level, 3),
            "information_gain": round(substrate.goals.information_gain, 3),
            "mood": substrate.emotions.mood,
            "energy": round(substrate.emotions.energy, 3),
            "consciousness_mode": substrate.consciousness.mode,
            "active_archetype": substrate.active_archetype.name if substrate.active_archetype else None,
        }
        ctx.mark_external("evaluate")
        result = await substrate.llm.evaluate_state(compact_state, lease=lease)
        # The tick learned something exactly when the appraisal produced an
        # insight; that is the value this lease returned.
        substrate.settle(lease, value=1.0 if result.get("success") else 0.0)
        if result.get("response"):
            _, llm_warnings = substrate.self_preservation.filter_llm_response(result["response"])
            if llm_warnings:
                substrate.autobiography.log_event("llm_danger", str(llm_warnings[:2]), 0.9)
        if result["success"] and "parsed" in result:
            llm_eval = result["parsed"]
            insight = llm_eval.get("insight", "")
            if insight:
                substrate.memory.add_episodic(
                    f"LLM Insight: {insight}",
                    emotional_valence=0.3, importance=0.8
                )
                substrate.autobiography.log_event("insight", insight[:100], 0.7)
                substrate._tick_llm_insights += 1
            suggested = llm_eval.get("suggested_goals", [])
            if not isinstance(suggested, list):
                suggested = []
            for sg in suggested[:2]:
                # Skip non-string / empty entries (audit M5) — a model may
                # return numbers or objects here.
                if not isinstance(sg, str) or not sg.strip():
                    continue
                from aegis.layers.goal_engine import Goal
                # Priority based on current system needs, not random
                priority = 0.5 + 0.1 * substrate.emotions.energy
                g = Goal(
                    name=sg[:30].replace(" ", "_").lower(),
                    level="tactic",
                    description=sg,
                    priority=priority,
                )
                g.reasoning = "Generated by LLM evaluation"
                substrate.goals.goals.append(g)

            await substrate.event_bus.publish(Event(
                source=Layer.INTROSPECTION, target=None,
                event_type="llm_evaluation",
                payload={"assessment": llm_eval.get("assessment", "")[:100]}
            ))

    substrate.memory.add_working({
        "phase": "evaluate",
        "activations": activations,
        "focus": focus,
        "bias_report": bias_report,
        "llm_eval": llm_eval,
    })
