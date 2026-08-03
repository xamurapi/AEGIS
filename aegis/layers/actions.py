"""The action registry (spec M2.3, Appendix A).

Everything the system can do is declared here, once, as data. Nothing outside
this table can be executed: the planner selects from it, the resource manager
prices from it, and the behaviour policy suppresses from it. Adding a capability
is one record plus an executor — the planner is never edited.

Replacing the old ``tick % N`` arithmetic with a registry changes what a
schedule *means*. ``min_interval`` is now a rate limit — "not more often than
this" — and no longer a trigger. What actually causes an action to run is that
it won on priority and got a lease. That is the difference between a system that
does things because the clock said so and one that does them because it decided
to.

Two fields carry safety:

* ``safety_critical`` — the behaviour policy may not suppress it, evolution may
  not disable it, and it may spend the reserved resource floor. Health checks,
  checkpoints and perception are not negotiable.
* ``executor`` — a dotted path resolved against the live substrate. An action
  whose subsystem is not attached is *unavailable*, not broken: that is how a
  contour still under construction, or one that failed to load, is prevented
  from being scheduled rather than crashing when selected.
"""
from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Callable

import aegis.config as cfg
from aegis.layers.motivation.resources import ResourceCost

logger = logging.getLogger("aegis.actions")


@dataclass(frozen=True)
class ActionSpec:
    """One thing the system can do."""

    name: str
    drive: str                       # competence | knowledge | coherence | stability
    cost: ResourceCost
    executor: str                    # dotted attribute path from the substrate
    preconditions: tuple[str, ...] = ()
    min_interval: int = 1            # rate limit in ticks, NOT a trigger
    reversible: bool = True
    safety_critical: bool = False
    #: Whether the wall time is spent waiting on something outside this process
    #: — a network call, a hosted model, a subprocess, a detached task.
    external: bool = False
    #: Development stage that delivers the executor. Actions whose stage has
    #: not landed resolve to nothing and are simply unavailable.
    stage: int = 2

    def reservation_cost(self) -> ResourceCost:
        """What is actually held against the per-tick budget.

        The declared cost keeps Appendix A's numbers, which are what ROI and
        reporting are computed from. But wall time spent waiting on a hosted
        model is not wall time the cognitive cycle spent thinking, and charging
        it to a 2500 ms tick budget would make every LLM action permanently
        unaffordable. This is the same rule the phase budgets already use
        (§3.4): external work is excluded, so the two measurements agree
        instead of contradicting each other.
        """
        if not self.external:
            return self.cost
        return ResourceCost(**{**self.cost.as_dict(), "wall_ms": 0})

    def describe(self) -> dict:
        return {
            "name": self.name,
            "drive": self.drive,
            "cost": self.cost.as_dict(),
            "reserved": self.reservation_cost().as_dict(),
            "executor": self.executor,
            "preconditions": list(self.preconditions),
            "min_interval": self.min_interval,
            "reversible": self.reversible,
            "safety_critical": self.safety_critical,
            "external": self.external,
            "stage": self.stage,
        }


def _cost(tok=0, ms=0, proc=0, net=0, train=0) -> ResourceCost:
    """Appendix A writes costs as tok / ms / proc / net / train."""
    return ResourceCost(llm_tokens=tok, wall_ms=ms, subprocess_slots=proc,
                        net_calls=net, training_slots=train,
                        llm_calls=1 if tok else 0)


# ── the registry (Appendix A, in table order) ────────────────────────

ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec("perceive_world", "coherence", _cost(ms=5), "world.perceive",
               min_interval=1, safety_critical=True, stage=2),
    ActionSpec("health_check", "stability", _cost(ms=3), "health.check",
               min_interval=1, safety_critical=True, stage=2),
    ActionSpec("checkpoint", "stability", _cost(ms=40), "_save_checkpoint",
               preconditions=("checkpoint_due",), min_interval=10,
               safety_critical=True, stage=2),
    ActionSpec("backup_state", "stability", _cost(ms=80), "state_backup.save_state",
               preconditions=("backup_due",), min_interval=50,
               safety_critical=True, stage=2),
    ActionSpec("capacity_regulate", "stability", _cost(ms=5), "regulate_capacity",
               preconditions=("has_latency_samples",), min_interval=50,
               safety_critical=True, stage=2),
    ActionSpec("env_step", "competence", _cost(ms=300, proc=1), "environment.step",
               preconditions=("skills_available",), min_interval=2, external=True, stage=2),
    ActionSpec("run_benchmark", "competence", _cost(ms=4000, proc=1), "evaluator.run",
               preconditions=("no_benchmark_running",), min_interval=50, external=True, stage=2),
    ActionSpec("synthesize_skill", "competence", _cost(tok=2500, ms=6000, proc=1),
               "_skill_synthesis",
               preconditions=("has_failing_kind", "role_code_available"),
               min_interval=200, external=True, stage=2),
    ActionSpec("synthesize_coding", "competence", _cost(tok=3000, ms=8000, proc=1),
               "_coding_synthesis", preconditions=("has_unsolved_coding",),
               min_interval=200, external=True, stage=2),
    ActionSpec("optimize_skill", "competence", _cost(tok=2000, ms=5000, proc=1),
               "_skill_optimization", preconditions=("no_failing_kind",),
               min_interval=200, external=True, stage=2),
    ActionSpec("reason_task", "competence", _cost(tok=4000, ms=8000, proc=1),
               "reasoning.solve", preconditions=("has_queued_task",),
               min_interval=1, external=True, stage=8),
    ActionSpec("evolve_generation", "competence", _cost(tok=800, ms=20000, proc=4),
               "evolution.run_generation",
               preconditions=("no_active_generation", "evolution_allowed"),
               min_interval=250, external=True, stage=7),
    ActionSpec("mine_rules", "coherence", _cost(ms=400), "policy.mine",
               preconditions=("enough_experiences",), min_interval=200, stage=5),
    ActionSpec("review_rules", "coherence", _cost(ms=150), "policy.review",
               preconditions=("has_active_rules",), min_interval=1000, stage=5),
    ActionSpec("scan_weakness", "coherence", _cost(ms=600), "reasoning.scan_weakness",
               preconditions=("enough_results",), min_interval=300, stage=9),
    ActionSpec("synthesize_strategy", "competence", _cost(tok=2000, ms=5000),
               "reasoning.propose_strategy_async", preconditions=("has_weakness",),
               min_interval=300, external=True, stage=9),
    ActionSpec("evaluate_strategy", "competence", _cost(tok=1500, ms=9000, proc=1),
               "reasoning.evaluate_candidate",
               preconditions=("has_candidate_strategy",), min_interval=300, external=True, stage=9),
    ActionSpec("scan_hypotheses", "knowledge", _cost(ms=1500), "discovery.scan",
               preconditions=("enough_telemetry",), min_interval=1000, stage=10),
    ActionSpec("fit_model", "knowledge", _cost(ms=3000, proc=1), "discovery.fit",
               preconditions=("has_unmodelled_hypothesis",), min_interval=1000,
               external=True, stage=10),
    ActionSpec("run_experiment", "knowledge", _cost(ms=200), "discovery.step_experiment",
               preconditions=("has_prereg", "health_ok"), min_interval=1, stage=10),
    ActionSpec("learn_external", "knowledge", _cost(ms=1200, net=1),
               "external_learning.learn_from_source",
               preconditions=("network_allowed", "not_skip_learning"),
               min_interval=40, external=True, stage=2),
    ActionSpec("run_agents", "knowledge", _cost(ms=2000, net=3),
               "agent_system.run_due_agents", preconditions=("not_skip_learning",),
               min_interval=5, external=True, stage=2),
    ActionSpec("evolve_agents", "knowledge", _cost(ms=300), "agent_system.evolve",
               min_interval=100, stage=2),
    ActionSpec("curiosity_explore", "knowledge", _cost(tok=900, ms=3000),
               "llm.generate_curiosity", preconditions=("role_deep_available",),
               min_interval=15, external=True, stage=2),
    ActionSpec("evaluate_state_llm", "coherence", _cost(tok=1200, ms=3000),
               "llm.evaluate_state", preconditions=("role_fast_available",),
               min_interval=3, external=True, stage=2),
    ActionSpec("reflect_llm", "coherence", _cost(tok=1200, ms=3000), "llm.reflect",
               preconditions=("role_fast_available",), min_interval=3, external=True, stage=2),
    ActionSpec("self_inspect", "coherence", _cost(ms=60), "introspection.detect_bias",
               preconditions=("has_decision_trace",), min_interval=20, stage=2),
    ActionSpec("consolidate_memory", "coherence", _cost(ms=250), "memory.apply_forgetting",
               min_interval=8, stage=2),
    ActionSpec("parametric_self_mod", "competence", _cost(tok=1000, ms=2000),
               "_llm_parametric_modification",
               preconditions=("role_deep_available", "health_not_critical"),
               min_interval=15, reversible=True, external=True, stage=2),
    ActionSpec("train_weights", "competence", _cost(train=1), "_weight_training_cycle",
               preconditions=("dataset_large_enough", "training_ethics_ok"),
               min_interval=1000, reversible=True, external=True, stage=2),
    ActionSpec("code_self_mod", "competence", _cost(tok=4000, ms=6000),
               "_code_self_modification", preconditions=("code_self_mod_enabled",),
               min_interval=500, reversible=False, external=True, stage=2),
    ActionSpec("dream", "stability", _cost(ms=100), "dreams.generate_dream",
               preconditions=("low_energy_or_reflective",), min_interval=50, stage=2),
    ActionSpec("rest", "stability", _cost(ms=10), "emotions.recharge",
               min_interval=1, stage=2),
    # M11: metacognition. Neither is safety_critical — the policy may suppress
    # them, the resource floor owes them nothing, and the system is required to
    # work with the whole contour switched off (M11.7.2).
    ActionSpec("attribute_strategy", "coherence", _cost(tok=900, ms=300, proc=1),
               "metacognition.attribute",
               preconditions=("has_unexplained_strategy",),
               min_interval=300, external=True, stage=11),
    ActionSpec("invent_strategy", "competence", _cost(tok=2000, ms=5000),
               "metacognition.invent",
               preconditions=("has_weakness", "far_quota_open"),
               min_interval=300, external=True, stage=11),
)

ACTIONS_BY_NAME: dict[str, ActionSpec] = {spec.name: spec for spec in ACTIONS}

#: The last development stage whose executors exist. Actions declared for a
#: later stage resolve to nothing and are simply unavailable — which is how a
#: contour under construction is kept out of the schedule instead of crashing
#: when it is selected. Bumped by the stage that delivers the executors.
DELIVERED_STAGE = 11


# ── preconditions ────────────────────────────────────────────────────
# Named predicates over the live substrate. Declaring them by name rather than
# inline keeps the registry data rather than code, and makes the whole set of
# conditions enumerable — which is what lets a test prove none of them is a
# typo that silently reads as False forever.

def _failing_kinds(substrate) -> list[str]:
    """What the last benchmark found failing.

    The cached answer, deliberately: ``failing_kinds()`` re-runs the whole
    sandboxed benchmark, which is seconds of subprocess work. A precondition
    evaluated on every candidate on every tick cannot afford to re-measure —
    and a decision should act on the last measurement anyway.
    """
    try:
        return substrate.evaluator.failing_kinds_cached()
    except Exception:
        logger.debug("failing_kinds probe failed", exc_info=True)
        return []


def _task_finished(task) -> bool:
    """Whether a detached task is absent or already over."""
    return task is None or task.done()


def _unsolved_coding(substrate) -> list:
    try:
        return substrate.evaluator.unsolved_coding_cached()
    except Exception:
        logger.debug("unsolved_coding probe failed", exc_info=True)
        return []


PREDICATES: dict[str, Callable[[object, object], bool]] = {
    "checkpoint_due":
        lambda s, ctx: s.tick_count % max(1, cfg.CHECKPOINT_EVERY_N_TICKS) == 0,
    "backup_due":
        lambda s, ctx: s.tick_count % max(1, cfg.CHECKPOINT_EVERY_N_TICKS * 5) == 0,
    "has_latency_samples":
        lambda s, ctx: bool(s.health.tick_durations),
    "skills_available":
        lambda s, ctx: bool(s.skill_library.skills),
    "no_benchmark_running":
        lambda s, ctx: s._eval_task is None or s._eval_task.done(),
    "has_failing_kind":
        lambda s, ctx: bool(_failing_kinds(s)),
    "no_failing_kind":
        lambda s, ctx: not _failing_kinds(s),
    "has_unsolved_coding":
        lambda s, ctx: bool(_unsolved_coding(s)),
    "has_queued_task":
        lambda s, ctx: bool(getattr(s, "reasoning", None)
                            and s.reasoning.has_queued_task()),
    "no_active_generation":
        lambda s, ctx: (not getattr(s.evolution, "generation_running", False)
                        and _task_finished(getattr(s, "_evolution_task", None))),
    "evolution_allowed":
        lambda s, ctx: (cfg.EVO_ENABLED
                        and not s._regulation_directives.get("skip_learning")),
    "enough_experiences":
        lambda s, ctx: s.feedback_loop.resolved >= cfg.POLICY_MIN_SUPPORT * 5,
    "has_active_rules":
        lambda s, ctx: bool(getattr(s, "policy", None) and s.policy.active_rules()),
    "enough_results":
        lambda s, ctx: bool(getattr(s, "reasoning", None)
                            and s.reasoning.result_count() >= cfg.DISC_MIN_N // 2),
    "has_weakness":
        lambda s, ctx: bool(getattr(s, "reasoning", None) and s.reasoning.top_weakness()),
    "has_candidate_strategy":
        lambda s, ctx: bool(getattr(s, "reasoning", None)
                            and s.reasoning.pending_candidates()),
    "enough_telemetry":
        lambda s, ctx: bool(getattr(s, "discovery", None)
                            and s.discovery.data_points() >= cfg.DISC_MIN_N),
    "has_unmodelled_hypothesis":
        lambda s, ctx: bool(getattr(s, "discovery", None)
                            and s.discovery.unmodelled_hypotheses()),
    "has_prereg":
        lambda s, ctx: bool(getattr(s, "discovery", None)
                            and s.discovery.active_preregistrations()),
    "health_ok":
        lambda s, ctx: s.health.last_status == "healthy",
    "health_not_critical":
        lambda s, ctx: s.health.last_status != "critical",
    "network_allowed":
        lambda s, ctx: bool(s.world.permissions.get("network", True)),
    "not_skip_learning":
        lambda s, ctx: not s._regulation_directives.get("skip_learning"),
    "role_fast_available":
        lambda s, ctx: s.llm.cortex.role_available("fast") or s.llm.enabled,
    "role_deep_available":
        lambda s, ctx: s.llm.cortex.role_available("deep") or s.llm.enabled,
    "role_code_available":
        lambda s, ctx: s.llm.cortex.role_available("code") or s.llm.enabled,
    "has_decision_trace":
        lambda s, ctx: len(s.introspection.decision_trace) >= 5,
    "dataset_large_enough":
        lambda s, ctx: len(s.memory.semantic) >= cfg.TRAIN_MIN_DATASET_SIZE,
    "training_ethics_ok":
        lambda s, ctx: not s.ethics.kill_switch_active,
    "code_self_mod_enabled":
        lambda s, ctx: cfg.CODE_SELF_MOD_ENABLED,
    "low_energy_or_reflective":
        lambda s, ctx: (s.emotions.energy < 0.4
                        or s.consciousness.mode == "reflective"),
    # M11: an accepted strategy nobody has explained yet, and room left in the
    # far-candidate quota. Both read the enabled flag through the contour, so
    # META_ENABLED=False makes both actions simply unavailable.
    "has_unexplained_strategy":
        lambda s, ctx: bool(getattr(s, "metacognition", None)
                            and s.metacognition.enabled
                            and s.metacognition.pending_attributions()),
    "far_quota_open":
        lambda s, ctx: bool(getattr(s, "metacognition", None)
                            and s.metacognition.enabled
                            and s.metacognition.quota_open()),
}

_COMPARATORS = {">=": operator.ge, "<=": operator.le, "==": operator.eq,
                "!=": operator.ne, ">": operator.gt, "<": operator.lt}


def evaluate_precondition(name: str, substrate, ctx=None) -> bool:
    """Whether one precondition holds. Unknown names are False, loudly.

    A typo in a precondition must not read as "always allowed" — silently
    granting an action its licence is the worse failure by far.
    """
    predicate = PREDICATES.get(name)
    if predicate is not None:
        try:
            return bool(predicate(substrate, ctx))
        except Exception:
            logger.debug("Precondition %s raised; treating as unmet", name,
                         exc_info=True)
            return False

    comparison = _parse_comparison(name)
    if comparison is not None:
        attribute, compare, threshold = comparison
        value = _resolve(substrate, attribute)
        try:
            return bool(compare(float(value), threshold))
        except (TypeError, ValueError):
            return False

    logger.warning("Unknown precondition %r — treating as unmet", name)
    return False


def _parse_comparison(text: str):
    """Parse ``energy>0.2`` into (path, comparator, threshold)."""
    for symbol in (">=", "<=", "==", "!=", ">", "<"):
        if symbol in text:
            left, _, right = text.partition(symbol)
            try:
                return left.strip(), _COMPARATORS[symbol], float(right.strip())
            except ValueError:
                return None
    return None


def _resolve(root, path: str):
    """Walk a dotted attribute path, or return None if it does not exist."""
    current = root
    for part in str(path).split("."):
        if not part:
            return None
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


class ActionSpace:
    """The set of actions, and which of them are possible right now."""

    def __init__(self, specs: tuple[ActionSpec, ...] = ACTIONS):
        self.specs = tuple(specs)
        self.by_name = {spec.name: spec for spec in self.specs}
        #: name -> tick it last ran, for the rate limit
        self.last_run: dict[str, int] = {}
        self.blocked_counts: dict[str, int] = {}

    # ── resolution ───────────────────────────────────────────────────

    def executor_for(self, spec: ActionSpec | str, substrate, ctx=None):
        """A zero-argument callable that performs an action, or None.

        An adapter is used when the declared executor needs arguments assembled
        from the tick (see :mod:`aegis.layers.executors`); otherwise the
        declared path is called directly. Either way the caller gets something
        it can simply invoke — a planner that could choose an action it was
        unable to perform would be choosing labels.
        """
        from aegis.layers.executors import adapter_for

        spec = self.by_name[spec] if isinstance(spec, str) else spec
        adapter = adapter_for(spec.name)
        if adapter is not None:
            # The adapter still needs its subsystem to exist, or an action from
            # a contour that has not landed would look performable.
            if _resolve(substrate, spec.executor) is None:
                return None
            return lambda: adapter(substrate, ctx)
        target = _resolve(substrate, spec.executor)
        return target if callable(target) else None

    def is_wired(self, spec: ActionSpec | str, substrate) -> bool:
        return self.executor_for(spec, substrate) is not None

    def unwired(self, substrate) -> list[str]:
        return sorted(spec.name for spec in self.specs
                      if not self.is_wired(spec, substrate))

    # ── availability ─────────────────────────────────────────────────

    def rate_limited(self, spec: ActionSpec, tick: int) -> bool:
        last = self.last_run.get(spec.name)
        return last is not None and tick - last < spec.min_interval

    def available(self, substrate, ctx=None) -> list[ActionSpec]:
        """Actions that could run this tick, in a deterministic order.

        Sorted by name rather than by declaration order or by score: the
        planner is what ranks, and an ordering that depended on dict iteration
        would make two identical runs diverge (§3.1).
        """
        tick = getattr(substrate, "tick_count", 0)
        out = []
        for spec in self.specs:
            if not self.is_wired(spec, substrate):
                continue
            if self.rate_limited(spec, tick):
                continue
            if not all(evaluate_precondition(p, substrate, ctx)
                       for p in spec.preconditions):
                continue
            out.append(spec)
        return sorted(out, key=lambda s: s.name)

    def mark_run(self, spec: ActionSpec | str, tick: int) -> None:
        name = spec if isinstance(spec, str) else spec.name
        self.last_run[name] = int(tick)

    def note_blocked(self, reason: str) -> None:
        self.blocked_counts[reason] = self.blocked_counts.get(reason, 0) + 1

    # ── reporting ────────────────────────────────────────────────────

    def safety_critical(self) -> list[ActionSpec]:
        return [spec for spec in self.specs if spec.safety_critical]

    def by_drive(self, drive: str) -> list[ActionSpec]:
        return sorted((s for s in self.specs if s.drive == drive),
                      key=lambda s: s.name)

    def status(self, substrate=None) -> dict:
        return {
            "total": len(self.specs),
            "safety_critical": [s.name for s in self.safety_critical()],
            "by_drive": {drive: [s.name for s in self.by_drive(drive)]
                         for drive in ("competence", "knowledge",
                                       "coherence", "stability")},
            "unwired": self.unwired(substrate) if substrate is not None else [],
            "blocked": dict(sorted(self.blocked_counts.items())),
        }
