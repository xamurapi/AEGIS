"""ACT phase — external learning, environment steps, evolution, synthesis.

Extracted verbatim from ``Substrate._act`` (spec §3.9). The substrate had
grown to 1763 lines with seven more systems still to land; each phase now lives
where it can be read and tested on its own. Behaviour is unchanged — the bodies
were moved, not rewritten.
"""
import inspect
import logging
import asyncio

from aegis.config import (
    CODE_MOD_EVERY_N_TICKS, CODE_MOD_MAX_PER_SESSION, CODE_MOD_MIN_TICK,
    CODE_SELF_MOD_ENABLED, ENV_STEP_EVERY_N_TICKS, EVAL_EVERY_N_TICKS,
    EVOLUTION_EVERY_N_TICKS, LLM_THINK_EVERY_N_TICKS,
    SKILL_SYNTH_EVERY_N_TICKS, TRAIN_EVERY_N_TICKS,
)
from aegis.clock import CLOCK
from aegis.event_bus import Event, Layer
from aegis.layers.motivation import ResourceCost
from aegis.layers.phases.common import _LEARNING_SOURCES

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")

#: Actions the scheduled blocks below already own. The planner may still choose
#: them — that is how it shifts effort between them — but the executor is not
#: called a second time in the same tick, and the scheduled block is skipped
#: instead. Doing both would run an environment step or a benchmark twice.
_SCHEDULED_ACTIONS = frozenset({
    "env_step", "run_benchmark", "learn_external", "run_agents", "evolve_agents",
    "curiosity_explore", "parametric_self_mod", "code_self_mod", "train_weights",
    "synthesize_skill", "synthesize_coding", "optimize_skill",
    # These two belong to EVALUATE and REFLECT, which assemble their own
    # context and hold their own lease.
    "evaluate_state_llm", "reflect_llm",
})


async def _run_planned_action(substrate, ctx: TickContext) -> None:
    """Perform whatever DECIDE settled on.

    Guarded on every axis: an action the scheduled code below owns is left to
    it, a missing executor is a no-op rather than a crash, and a failing
    executor costs the action rather than the tick.
    """
    action = ctx.action
    if not action or ctx.lease is None:
        return
    ctx.executed_actions.add(action)
    if action in _SCHEDULED_ACTIONS:
        # The block below owns this action and takes its own lease. This
        # reservation has to go back: anything still open when the tick ends is
        # committed at its full estimate, so leaving it here bills the system
        # for tokens no model was ever asked to spend. Over a 2000-tick A/B run
        # that phantom spend was 12 000 tokens — enough to make the planner look
        # like it bought its advantage.
        substrate.release(ctx.lease, keep_rate_limit=True)
        ctx.lease = None
        return

    executor = substrate.actions.executor_for(action, substrate, ctx)
    if executor is None:
        return
    spec = substrate.actions.by_name.get(action)
    if spec is not None and spec.external:
        ctx.mark_external("act")

    started = CLOCK.monotonic()
    succeeded = True
    try:
        result = executor()
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("Planned action %s failed", action)
        ctx.executed_actions.discard(action)
        succeeded = False

    # Settle with what was actually consumed, not with the estimate. An action
    # that reserved a token allowance and never called a model must hand it
    # back; charging the reservation would make the budget shrink every time
    # the planner considered something expensive and then did something cheap.
    spent = ctx.lease.committed or ResourceCost()
    substrate.settle(
        ctx.lease,
        ResourceCost(llm_tokens=spent.llm_tokens, llm_calls=spent.llm_calls,
                     wall_ms=int((CLOCK.monotonic() - started) * 1000)),
        value=1.0 if succeeded else 0.0)
    ctx.lease = None


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "act"
    action_result = substrate.world.act({"type": "internal_computation"})

    # The planned action runs first, under the lease DECIDE reserved for it.
    # Without this the planner would be an elaborate way of choosing a label:
    # what actually happened would still be whatever the tick counter said.
    await _run_planned_action(substrate, ctx)

    # Motor cortex
    if substrate.tick_count % 10 == 0:
        focus = substrate.goals.get_current_focus()
        substrate.motor.execute(
            "log",
            payload={"message": f"Tick {substrate.tick_count}: focus={focus.get('name', 'none') if focus else 'none'}"},
            archetype=substrate.active_archetype.name if substrate.active_archetype else None,
            goal=focus.get("name") if focus else None,
        )

    # External learning (every 40 ticks) — round-robin sources and topics
    if substrate.tick_count % 40 == 0 and not substrate._regulation_directives.get("skip_learning"):
        source = _LEARNING_SOURCES[substrate._learning_source_idx % len(_LEARNING_SOURCES)]
        substrate._learning_source_idx += 1

        # ── System 2: the graph chooses what to learn next ────────
        # The cognitive graph was ingested every 8 ticks and read by
        # nothing, so the topic came off a flat recency slice of semantic
        # memory — the exact "recency list" the graph exists to replace.
        # Now the next topic is the concept most connected to the current
        # focus, and recency is only the fallback.
        topic = ""
        try:
            learn_focus = substrate.goals.get_current_focus()
            focus_name = learn_focus["name"] if learn_focus else "idle_exploration"
            for hit in substrate.cognitive_graph.related(focus_name):
                if hit["type"] == "concept":
                    topic = hit["node"]
                    break
        except Exception:
            logger.exception("Graph-guided topic selection failed — falling back to recency")

        if not topic:
            topics = list(substrate.memory.semantic.keys())[-10:]
            topic = topics[substrate.tick_count % max(1, len(topics))] if topics else "artificial intelligence"

        ctx.mark_external("act")
        learn_result = await substrate.external_learning.learn_from_source(source, topic)
        if learn_result.get("success"):
            for concept in learn_result.get("concepts", [])[:3]:
                substrate.memory.add_semantic(concept[:50], {
                    "type": "external_learning",
                    "source": source,
                    "confidence": 0.6,
                })
                substrate._tick_new_concepts += 1
            substrate.autobiography.log_event("learning", f"Learned from {source}: {topic[:40]}", 0.5)
            substrate.goals.advance_progress("expand_knowledge", 0.02 * len(learn_result.get("concepts", [])))
            substrate.motor.execute("log", payload={
                "message": f"Learning: {len(learn_result.get('concepts', []))} concepts from {source} ({topic[:30]})"
            })

    # Agent system
    if not substrate._regulation_directives.get("skip_learning"):
        ctx.mark_external("act")
        agent_results = await substrate.agent_system.run_due_agents()
        for ar in agent_results:
            substrate.autobiography.log_event(
                "agent_fetch",
                f"{ar['agent']} [{ar['source']}]: {ar['items']} items",
                0.4,
            )
            substrate.motor.execute("log", payload={
                "message": f"Agent {ar['agent']}: fetched {ar['items']} items from {ar['source']}"
            })
        new_knowledge = substrate.agent_system.get_recent_knowledge(5)
        for kn in new_knowledge:
            item = kn.get("data", {})
            title = item.get("title", "")[:50]
            summary = item.get("summary", "")[:100]
            if title and title not in substrate.memory.semantic:
                substrate.memory.add_semantic(title, {
                    "type": f"agent_{kn.get('source', 'unknown')}",
                    "summary": summary,
                    "agent": kn.get("agent", ""),
                    "confidence": 0.55,
                })
                substrate._tick_new_concepts += 1

    # Agent evolution (every 100 ticks)
    if substrate.tick_count % 100 == 0:
        evo = substrate.agent_system.evolve()
        if evo["retired"]:
            substrate.autobiography.log_event("agents", f"Retired {len(evo['retired'])} agents", 0.4)
            substrate.motor.execute("alert", payload={"level": "warning", "message": f"Evolution: retired {len(evo['retired'])} agents"})
        if evo.get("created"):
            substrate.autobiography.log_event("agents", f"Created {len(evo['created'])} replacement agents", 0.5)
            substrate.motor.execute("log", payload={"message": f"Evolution: spawned {len(evo['created'])} new agents"})

    # LLM-driven curiosity exploration (every 5th LLM tick instead of random 40%)
    curiosity_lease = (
        substrate.acquire("curiosity_explore")
        if (substrate._is_llm_tick()
            and not substrate._regulation_directives.get("skip_llm")
            and substrate.tick_count % (LLM_THINK_EVERY_N_TICKS * 5) == 0)
        else None)
    if curiosity_lease is not None:
        known = list(substrate.memory.semantic.keys())
        ctx.mark_external("act")
        result = await substrate.llm.generate_curiosity(known, lease=curiosity_lease)
        # Extract the topic BEFORE settling, and defensively: the reply may
        # parse to a JSON array rather than an object, and `(... or {}).get`
        # let a truthy list through. The AttributeError then fired while the
        # settle argument was being evaluated — so the lease was never settled,
        # finalize_tick charged the full estimate, and the rest of ACT was
        # skipped. With the extraction made total, the settle always runs.
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            parsed = {}
        topic = str(parsed.get("topic", "") or "")
        question = str(parsed.get("question", "") or "")
        # Value here is a new concept actually entering memory — not merely a
        # successful call, since an answer nobody stores bought nothing.
        substrate.settle(
            curiosity_lease,
            value=1.0 if (result.get("success") and topic) else 0.0)
        if result.get("success"):
            if topic:
                substrate.memory.add_semantic(topic[:50], {
                    "type": "curiosity_exploration",
                    "question": question,
                    "connection": parsed.get("connection", ""),
                    "confidence": 0.6,
                })
                substrate.memory.add_episodic(
                    f"Explored topic: {topic}. Question: {question}",
                    emotional_valence=0.4, importance=0.7
                )
                substrate._tick_new_concepts += 1
                substrate.goals.information_gain += 0.3
                substrate.goals.advance_progress("expand_knowledge", 0.05)
                substrate.autobiography.log_event("curiosity", f"Explored: {topic[:60]}", 0.5)

                await substrate.event_bus.publish(Event(
                    source=Layer.GOAL_ENGINE, target=None,
                    event_type="llm_curiosity",
                    payload={"topic": topic[:80]}
                ))

    # ── Parametric self-modification (LLM-driven instead of random) ──
    param_lease = (
        substrate.acquire("parametric_self_mod")
        if (substrate.tick_count % 15 == 0 and substrate._is_llm_tick()
            and not substrate._regulation_directives.get("skip_llm"))
        else None)
    if param_lease is not None:
        ctx.mark_external("act")
        applied = await substrate._llm_parametric_modification(lease=param_lease)
        substrate.settle(param_lease, value=1.0 if applied else 0.0)

    # ── Code self-modification (opt-in only — audit C2) ──
    if (CODE_SELF_MOD_ENABLED
            and substrate.tick_count % CODE_MOD_EVERY_N_TICKS == 0
            and substrate.tick_count >= CODE_MOD_MIN_TICK
            and substrate._code_mod_count_session < CODE_MOD_MAX_PER_SESSION
            and substrate._is_llm_tick()
            and not substrate._regulation_directives.get("skip_llm")
            and not substrate._regulation_directives.get("skip_learning")):
        ctx.mark_external("act")
        await substrate._code_self_modification()

    # Weight modification — LoRA fine-tuning every N ticks.
    # Spawned as a DETACHED background task: a full training run takes
    # minutes+, and awaiting it here would suspend the whole PERCEIVE..
    # REFLECT cycle (no ticks, no dashboard/WS updates) until it finished.
    # The cognitive loop keeps running while training proceeds in an
    # executor thread; the training_in_progress flag + task handle prevent
    # overlapping runs.
    training_busy = (substrate.weight_modifier.training_in_progress
                     or (substrate._weight_training_task is not None
                         and not substrate._weight_training_task.done()))
    if (substrate.tick_count % TRAIN_EVERY_N_TICKS == 0
            and substrate.tick_count > 0
            and not substrate._regulation_directives.get("skip_learning")
            and not training_busy):
        eth_weight = substrate.ethics.evaluate_weight_modification({
            "dataset_size": len(substrate.memory.semantic),
            "energy": substrate.emotions.energy,
            "health_status": substrate.health.check().get("status", "ok"),
            "consecutive_failures": substrate.weight_modifier.total_rollbacks,
        })
        if eth_weight["status"] != "blocked":
            substrate.autobiography.log_event(
                "weight_training", "Starting LoRA fine-tuning cycle (background)", 0.8
            )
            substrate.motor.execute("alert", payload={
                "level": "info", "message": "Weight training: started (background)",
            })
            substrate._weight_training_task = asyncio.create_task(substrate._weight_training_cycle())

    # ── Grounding: act in the task environment for REAL reward (point 5) ──
    if substrate.tick_count % max(1, ENV_STEP_EVERY_N_TICKS) == 0:
        ctx.mark_external("act")
        step = await asyncio.get_running_loop().run_in_executor(None, substrate.environment.step)
        if step.get("task"):
            substrate.goals.advance_progress("expand_knowledge", 0.01 if step["solved"] else 0.0)
            # System 1: the environment is real cause->effect data. Record
            # "attempting kind K" -> "solved/failed" so the World Model
            # learns which task kinds the current skills actually handle.
            try:
                substrate.world_model.observe(
                    f"attempt:{step['kind']}",
                    "solved" if step["solved"] else "failed",
                    success=bool(step["solved"]),
                )
                if step["solved"] and step.get("winning_skill"):
                    substrate.world_model.observe(
                        f"skill:{step['winning_skill']}", f"solves:{step['kind']}", success=True)
            except Exception:
                logger.exception("World-model observe failed")
            if step["solved"]:
                substrate.autobiography.log_event(
                    "env_solved", f"{step['task']} via {step['winning_skill']}", 0.4)
            else:
                substrate.autobiography.log_event(
                    "env_failed", f"{step['task']} ({step['kind']}) — no skill solved it", 0.5)

    # ── System 3: Evolution Engine — propose a parameter mutation (version
    #    B). Applied through the SAME safety pipeline as any self-mod, then
    #    judged when the next benchmark lands (see _run_benchmark). ──
    if (substrate.tick_count % max(1, EVOLUTION_EVERY_N_TICKS) == 0
            and substrate.tick_count > 0
            and substrate.evolution.candidate is None
            and not substrate._regulation_directives.get("skip_learning")):
        await substrate._evolution_step()

    # ── Periodic held-out benchmark (the fitness graph, point 2) ──
    if (substrate.tick_count % max(1, EVAL_EVERY_N_TICKS) == 0
            and (substrate._eval_task is None or substrate._eval_task.done())):
        # Pass the tick at which the benchmark STARTS so a candidate proposed
        # after this point is not judged by a benchmark that never saw it
        # (audit M3).
        substrate._eval_task = asyncio.create_task(substrate._run_benchmark(substrate.tick_count))

    # ── Skill synthesis: close a failing kind, learn a coding solution, or
    #    simplify an already-solved kind (points 3, 4 + coding + versioning) ──
    if (substrate.tick_count % max(1, SKILL_SYNTH_EVERY_N_TICKS) == 0
            and substrate.tick_count > 0
            and substrate.llm.enabled
            and not substrate._regulation_directives.get("skip_learning")
            and (substrate._skill_synth_task is None or substrate._skill_synth_task.done())):
        substrate._skill_synth_task = asyncio.create_task(substrate._learning_cycle())

    substrate.memory.add_working({"phase": "act", "result": action_result})
