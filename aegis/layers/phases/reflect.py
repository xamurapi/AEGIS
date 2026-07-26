"""REFLECT phase — meta-reflection, experience closure, consolidation.

Extracted verbatim from ``Substrate._reflect`` (spec §3.9). The substrate had
grown to 1763 lines with seven more systems still to land; each phase now lives
where it can be read and tested on its own. Behaviour is unchanged — the bodies
were moved, not rewritten.
"""
import logging

from aegis.config import CHECKPOINT_EVERY_N_TICKS, COGNITIVE_GRAPH_EVERY_N_TICKS
from aegis.event_bus import Event, Layer
from aegis.layers.phases.common import _CONCEPT_SEEDS, _META_DOMAINS

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "reflect"

    # Meta-reflection (every 20 ticks)
    if substrate.tick_count % 20 == 0:
        total_goals = len(substrate.goals.goals)
        completed = sum(1 for g in substrate.goals.goals if g.status == "completed")
        total_ticks = substrate.health.successful_ticks + substrate.health.failed_ticks
        error_rate = substrate.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
        recent = [e["event"] for e in substrate.memory.episodic[-8:]]
        mr = substrate.meta_reflection.reflect(
            substrate.tick_count, substrate.emotions.energy, substrate.emotions.mood,
            substrate.emotions.valence, error_rate, completed, total_goals,
            substrate.consciousness.mode, recent,
        )
        for insight in mr.get("insights", [])[:2]:
            substrate.memory.add_episodic(f"MetaInsight: {insight[:80]}", emotional_valence=0.2, importance=0.75)

    # LLM-powered reflection
    llm_reflection = None
    if substrate._is_llm_tick() and not substrate._regulation_directives.get("skip_llm"):
        recent_events = [e["event"] for e in substrate.memory.episodic[-5:]]
        episode = {
            "tick": substrate.tick_count,
            "recent_events": recent_events,
            "goals_completed": sum(1 for g in substrate.goals.goals if g.status == "completed"),
            "information_gain": round(substrate.goals.information_gain, 3),
            "version": substrate.self_mod.current_version,
            "mood": substrate.emotions.mood,
            "consciousness_mode": substrate.consciousness.mode,
        }
        ctx.mark_external("reflect")
        result = await substrate.llm.reflect(episode)
        if result["success"] and "parsed" in result and isinstance(result["parsed"], dict):
            llm_reflection = result["parsed"]
            learning = str(llm_reflection.get("learning", "") or "")
            if learning:
                substrate.memory.add_episodic(
                    f"Reflection: {learning}",
                    emotional_valence=0.2, importance=0.85
                )
                substrate.autobiography.log_event("reflection", learning[:100], 0.6)
                substrate._tick_llm_insights += 1
            # "knowledge" may be a string or missing instead of an object
            # (audit M5) — guard before calling .get on it.
            knowledge = llm_reflection.get("knowledge", {})
            if isinstance(knowledge, dict) and isinstance(knowledge.get("concept"), str) \
                    and knowledge["concept"].strip():
                substrate.memory.add_semantic(knowledge["concept"], {
                    "definition": str(knowledge.get("definition", "")),
                    "type": "learned_concept",
                    "source": "self_reflection",
                    "confidence": 0.75,
                })
                substrate.memory.update_meta(
                    knowledge["concept"], True, 0.75
                )
                substrate._tick_new_concepts += 1

            await substrate.event_bus.publish(Event(
                source=Layer.INTROSPECTION, target=None,
                event_type="llm_reflection",
                payload={"learning": learning[:100]}
            ))

        substrate.llm_thinking = False

    # Dream generation (every 50 ticks when energy is low or reflective)
    if substrate.tick_count % 50 == 0 and not substrate._regulation_directives.get("skip_dreams") and \
            (substrate.emotions.energy < 0.4 or substrate.consciousness.mode == "reflective"):
        recent = [e["event"] for e in substrate.memory.episodic[-10:]]
        concepts = list(substrate.memory.semantic.keys())[-15:]
        dream = substrate.dreams.generate_dream(substrate.emotions.mood, recent, concepts)
        substrate.autobiography.log_event("dream", dream["narrative"][:80], 0.4)
        substrate.memory.add_episodic(f"Dream: {dream['narrative'][:80]}", emotional_valence=0.1, importance=0.5)

    # Energy recharge on rest ticks
    if substrate.tick_count % 20 == 0:
        substrate.emotions.recharge(0.05)

    # Event summary with computed importance
    event_summary = f"Tick {substrate.tick_count}: cycle completed"
    if llm_reflection:
        # Coerce — "learning" may be a non-string (audit M5).
        _learned = str(llm_reflection.get("learning", "") or "")
        event_summary += f" | Learned: {_learned[:60]}"
    importance = substrate._compute_importance()
    # Valence from actual emotional state
    valence = substrate.emotions.valence - 0.5  # center around 0
    substrate.memory.add_episodic(event_summary, emotional_valence=valence, importance=importance)
    substrate._tick_new_episodic += 1

    # Meta-knowledge update — round-robin through domains
    if substrate.tick_count % 10 == 0:
        domain = _META_DOMAINS[substrate._meta_domain_idx % len(_META_DOMAINS)]
        substrate._meta_domain_idx += 1
        # Confidence based on actual success in that domain
        domain_confidence = 0.5 + 0.3 * substrate.emotions.success_rate + 0.2 * substrate.emotions.energy
        substrate.memory.update_meta(domain, True, min(0.95, domain_confidence))

    # Concept seeding — round-robin
    if substrate.tick_count % 25 == 0:
        concept = _CONCEPT_SEEDS[substrate._concept_seed_idx % len(_CONCEPT_SEEDS)]
        substrate._concept_seed_idx += 1
        substrate.memory.add_semantic(f"{concept}_{substrate.tick_count}", {
            "type": concept, "tick": substrate.tick_count,
            "confidence": 0.5 + 0.3 * substrate.emotions.success_rate,
        })

    if substrate.tick_count % 30 == 0:
        substrate.memory.apply_forgetting()

    # ── Higher systems (5, 4, 2): close the experience loop, credit the
    #    realized reward to the chosen objective's value, and grow the
    #    cognitive graph from this tick's memory. Guarded — never aborts. ──
    try:
        realized = substrate._compute_reward()
        exp_id = substrate._pending_experiences.pop("decide", None)
        if exp_id is not None:
            # Success = this tick produced knowledge/insight and stayed healthy.
            success = (substrate._tick_new_concepts > 0 or substrate._tick_llm_insights > 0) \
                and substrate.health.consecutive_errors == 0
            experience = substrate.feedback_loop.record_result(
                exp_id, success=success, metric=realized,
                expected="knowledge gain / healthy tick",
            )
            # System 4: credit realized reward to the chosen objective.
            substrate.goal_intelligence.reward(realized)
            # System 1: the decision's outcome is causal data too.
            if experience is not None:
                substrate.world_model.observe(
                    f"decision:{experience['decision'][:40]}",
                    "productive" if success else "unproductive",
                    success=success,
                )

        # System 2: ingest recent memory into the typed cognitive graph.
        if substrate.tick_count % max(1, COGNITIVE_GRAPH_EVERY_N_TICKS) == 0:
            substrate.cognitive_graph.ingest_memory(substrate.memory)
    except Exception:
        logger.exception("Higher-systems REFLECT hook failed")

    if substrate.tick_count % CHECKPOINT_EVERY_N_TICKS == 0:
        substrate._save_checkpoint()
        if substrate.tick_count % (CHECKPOINT_EVERY_N_TICKS * 5) == 0:
            substrate.state_backup.save_state(substrate.full_status(), "scheduled")

    await substrate.event_bus.publish(Event(
        source=Layer.SUBSTRATE, target=None,
        event_type="tick_complete",
        payload={"tick": substrate.tick_count}
    ))
