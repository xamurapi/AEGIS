"""Turning a declared action into something callable (spec M2.3, Appendix A).

Appendix A names each action's executor as a path — ``StateBackup.save_state``,
``IntrospectionEngine.detect_bias`` — and that is the right thing to declare:
it says which subsystem owns the work. It is not, however, the right thing to
*call*. Most of those methods need arguments assembled from the current tick,
and a registry that pointed straight at them produced a planner that could
choose an action it was unable to perform.

So each such action gets an adapter here: a function of ``(substrate, ctx)``
that gathers what the underlying call needs and makes it. Actions whose
executor genuinely takes no arguments — ``health.check``, ``_save_checkpoint``
— need no entry and are called through the declared path directly.

Adapters live here rather than on the substrate so that the registry stays a
table of data and the substrate does not grow a method per action.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("aegis.executors")


def backup_state(substrate, ctx):
    """A full state snapshot, labelled as planned rather than scheduled."""
    return substrate.state_backup.save_state(substrate.full_status(), "planned")


def self_inspect(substrate, ctx):
    """Look for bias in the recent decision trace."""
    return substrate.introspection.detect_bias(
        substrate.introspection.decision_trace[-20:])


def consolidate_memory(substrate, ctx):
    """Forget what has decayed, then fold what is left into the graph.

    Both halves, because consolidation that only forgets loses knowledge and
    consolidation that only ingests never reclaims anything.
    """
    substrate.memory.apply_forgetting()
    return substrate.cognitive_graph.ingest_memory(substrate.memory)


def dream(substrate, ctx):
    """Recombine recent experience while the system is idle or low on energy."""
    recent = [entry["event"] for entry in substrate.memory.episodic[-10:]]
    concepts = list(substrate.memory.semantic.keys())[-15:]
    return substrate.dreams.generate_dream(substrate.emotions.mood, recent, concepts)


def rest(substrate, ctx):
    """Recover a little energy."""
    return substrate.emotions.recharge(0.05)


def mine_rules(substrate, ctx):
    """Look for new behaviour rules in the experience log (M3.4).

    The safety-critical set is passed in rather than looked up inside the
    policy: what must never be suppressed is a property of the action registry,
    and a policy that decided this for itself could decide otherwise.
    """
    protected = [spec.name for spec in substrate.actions.safety_critical()]
    return substrate.policy.mine(substrate.tick_count, safety_critical=protected)


def review_rules(substrate, ctx):
    """Conclude finished trials and re-judge active rules (M3.5)."""
    return substrate.policy.review(substrate.tick_count)


def evolve_generation(substrate, ctx):
    """Start a generation, detached. Never run one inside a tick.

    A generation evaluates ten variants, each of which runs a benchmark and a
    short rollout in another process — minutes of work against a 20 ms ACT
    budget (§3.4). Called inline it does not merely overrun the budget, it
    stops the cognitive cycle for the duration: no ticks, no dashboard, no
    health checks. So the action's job is to *schedule* it, exactly as the
    benchmark and the training cycle are scheduled.

    The return value is deliberately NOT the task. ACT's generic executor
    plumbing awaits any awaitable an executor returns, so handing the task
    back re-attached the very work this function exists to detach — the tick
    then sat inside the generation for minutes. The task lives on
    ``substrate._evolution_task`` (the action registry's precondition reads it
    there); the caller gets a plain descriptor it cannot accidentally await.
    """
    if substrate.evolution.generation_running:
        return None
    task = getattr(substrate, "_evolution_task", None)
    if task is not None and not task.done():
        return None

    async def _run():
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, substrate.evolution.run_generation, substrate.tick_count)
        except Exception:
            logger.exception("A detached evolution generation failed")
            return None

    substrate._evolution_task = asyncio.create_task(_run())
    return {"scheduled": True, "tick": substrate.tick_count}


#: action name -> adapter. Only actions whose declared executor needs arguments
#: appear here; everything else is called through its declared path.
ADAPTERS = {
    "backup_state": backup_state,
    "self_inspect": self_inspect,
    "consolidate_memory": consolidate_memory,
    "dream": dream,
    "rest": rest,
    "mine_rules": mine_rules,
    "review_rules": review_rules,
    "evolve_generation": evolve_generation,
}


def adapter_for(name: str):
    return ADAPTERS.get(str(name))
