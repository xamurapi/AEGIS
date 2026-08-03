"""Layer 0: Substrate — persistent runtime with PERCEIVE-EVALUATE-DECIDE-ACT-REFLECT cycle (S-001..S-006).

The core cognitive cycle (reward, confidence, importance, goal progress) is
deterministic and driven by real system metrics. Knowledge-acquisition helpers
(ExternalLearning, AgentSystem) still pick topics with `random` when none is
supplied, so the runtime as a whole is not fully deterministic.
Code self-modification is integrated via CodeModifier + LLM proposals.
"""
import asyncio
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.config import (
    CHECKPOINT_EVERY_N_TICKS, CHECKPOINTS_DIR,
    LLM_THINK_EVERY_N_TICKS, CODE_BACKUPS_DIR, CODE_MOD_MAX_FILE_CHARS, CODE_SELF_MOD_ENABLED,
    EVAL_DIR, SANDBOX_TIMEOUT,
    CAPACITY_EVERY_N_TICKS, CAPACITY_GROWTH_FACTOR, CAPACITY_SHRINK_FACTOR,
    CAPACITY_HEADROOM, CAPACITY_MAX_MULTIPLE,
)
from aegis.clock import CLOCK
from aegis.event_bus import EventBus, Event, Layer
from aegis.safety import immutable
from aegis.store.migrations import read_store, write_store
from aegis.telemetry import metrics as M
from aegis.telemetry.store import Telemetry
from aegis.util.canonical import digest_of
from aegis.layers.phases import act, decide, evaluate, perceive, reflect
from aegis.layers.phases.common import (
    _CONCEPT_SEEDS, _LEARNING_SOURCES, _META_DOMAINS, _coerce_float, _coerce_int,
)
from aegis.layers.phases.context import TickContext
from aegis.layers.actions import ActionSpace
from aegis.layers.motivation import (
    Candidate, PriorityScheduler, ResourceCost, ResourceManager, ROITracker,
)
from aegis.layers.planner import Planner
from aegis.layers.policy import BehaviourPolicy
from aegis.layers.discovery import DiscoveryEngine
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.memory import MemorySystem
from aegis.layers.introspection import IntrospectionEngine
from aegis.layers.self_modification import SelfModification
from aegis.layers.goal_engine import GoalEngine
from aegis.layers.world_interface import WorldInterface
from aegis.layers.ethics_core import EthicsCore
from aegis.layers.consciousness import ConsciousnessState
from aegis.layers.emotions import EmotionalSystem
from aegis.layers.dreams import DreamEngine
from aegis.layers.autobiography import Autobiographer
from aegis.layers.archetypes import Archetype, create_default_archetypes, ArchetypeGeopolitics
from aegis.layers.worldview import Worldview, ValueSystem
from aegis.layers.health_monitor import HealthMonitor
from aegis.layers.self_preservation import SelfPreservation
from aegis.layers.state_backup import StateBackup
from aegis.layers.meta_consciousness import MetaConsciousness
from aegis.layers.meta_regulation import MetaRegulator
from aegis.layers.meta_reflection import MetaReflection
from aegis.layers.meta_goal_generator import MetaGoalGenerator
from aegis.layers.sensor_cortex import SensorCortex
from aegis.layers.motor_cortex import MotorCortex
from aegis.layers.external_learning import ExternalLearning
from aegis.layers.agent_system import AgentSystem
from aegis.layers.emotion_nlp import EmotionNLP
from aegis.layers.weight_modifier import WeightModifier
from aegis.layers.dataset_builder import DatasetBuilder
from aegis.layers.code_modifier import CodeModifier
from aegis.layers.world_model import WorldModel, MAX_LINKS as WM_MAX_LINKS
from aegis.layers.cognitive_graph import CognitiveGraph, MAX_NODES as CG_MAX_NODES
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.evolution.genome import GENES_BY_NAME, Genome
from aegis.layers.goal_intelligence import GoalIntelligence
from aegis.layers.feedback_loop import FeedbackLoop
from aegis.eval.benchmark import DEFAULT_BENCHMARK, split_tasks
from aegis.eval.skill_library import SkillLibrary, Skill
from aegis.eval.solver import MultiAgentSolver
from aegis.eval.sandbox import run_skill
from aegis.eval.evaluator import Evaluator
from aegis.eval.pool import EvaluationPool
from aegis.eval.environment import TaskEnvironment
from aegis.llm import LLMEngine

logger = logging.getLogger("aegis.substrate")

# The round-robin source lists and the LLM-output coercion helpers moved to
# aegis/layers/phases/common.py when the cycle was split (spec §3.9); they are
# re-exported here because they were part of this module's surface.
__all__ = ["Substrate", "_LEARNING_SOURCES", "_META_DOMAINS", "_CONCEPT_SEEDS",
           "_coerce_int", "_coerce_float"]

#: Distinguishes "this lease never touched the rate limit" from "the action had
#: never run before". ``None`` is a meaningful stored value here, so it cannot
#: double as the missing marker.
_NO_PREVIOUS_RUN = object()


class Substrate:
    def __init__(self):
        self.event_bus = EventBus()
        self.ethics = EthicsCore()
        self.memory = MemorySystem()
        self.introspection = IntrospectionEngine()
        self.self_mod = SelfModification()
        self.goals = GoalEngine()
        self.world = WorldInterface()
        self.llm = LLMEngine()

        # Neuro modules
        self.consciousness = ConsciousnessState()
        self.emotions = EmotionalSystem()
        self.dreams = DreamEngine()
        self.autobiography = Autobiographer()
        self.archetypes_list = create_default_archetypes()
        self.geopolitics = ArchetypeGeopolitics(self.archetypes_list)
        self.active_archetype: Archetype | None = None
        self.worldview = Worldview()
        self.values = ValueSystem()
        self.health = HealthMonitor()
        self.self_preservation = SelfPreservation()

        # Architecture modules
        self.state_backup = StateBackup()
        self.meta_consciousness = MetaConsciousness()
        self.meta_regulation = MetaRegulator()
        self.meta_reflection = MetaReflection()
        self.meta_goals = MetaGoalGenerator()
        self.sensors = SensorCortex()
        self.motor = MotorCortex()
        self.external_learning = ExternalLearning()
        self.agent_system = AgentSystem()
        self.emotion_nlp = EmotionNLP()
        self.weight_modifier = WeightModifier()
        self.dataset_builder = DatasetBuilder()

        # Five higher-order cognitive systems (spec: "5 ключевых систем").
        # 1. World Model — causal cause->effect model + objective chains.
        # 2. Cognitive Graph — typed knowledge/experience graph.
        # 3. Evolution Engine — mutate params, keep only benchmark-verified wins.
        # 4. Goal Intelligence — value-driven goal selection (motivation).
        # 5. Feedback Loop — situation->decision->result->cause->experience.
        self.world_model = WorldModel()
        self.cognitive_graph = CognitiveGraph()
        self.evolution = EvolutionEngine()
        self.goal_intelligence = GoalIntelligence()
        self.feedback_loop = FeedbackLoop()

        # Code self-modification
        aegis_pkg_dir = Path(__file__).parent.parent  # aegis/ package root
        self.code_modifier = CodeModifier(aegis_pkg_dir, CODE_BACKUPS_DIR)
        self._code_mod_count_session = 0
        self._code_file_index = 0  # round-robin through source files for analysis

        # Capability layer — verifiable benchmark, skills, sandbox, environment.
        # This is the system's external ground-truth signal (see _compute_reward).
        self.skill_library = SkillLibrary(store_path=EVAL_DIR / "skills.json")
        self.solver = MultiAgentSolver(self.skill_library, timeout=SANDBOX_TIMEOUT)
        self.evaluator = Evaluator(
            self.solver, tasks=list(DEFAULT_BENCHMARK),
            store_path=EVAL_DIR / "eval_history.json",
        )
        self.environment = TaskEnvironment(self.solver, tasks=list(DEFAULT_BENCHMARK))
        self._last_benchmark_score = self.evaluator.last_score  # may be None on first boot
        self._eval_task: asyncio.Task | None = None
        self._skill_synth_task: asyncio.Task | None = None
        self._skill_synth_idx = 0
        self._skill_synth_count = 0

        # Wire dependencies
        self.self_mod.weight_modifier = self.weight_modifier
        self.self_mod.dataset_builder = self.dataset_builder
        self.self_mod.code_modifier = self.code_modifier
        self.llm.weight_modifier = self.weight_modifier

        # Seed the Evolution Engine's champion genome from the live tunable
        # parameters + the last known benchmark (fitness). Mutations are judged
        # against this baseline; a change survives only if the held-out
        # benchmark improves — "self-modification != self-improvement" (spec #2).

        # Metric time-series (spec M9.2). Every contour publishes here; this is
        # the only history the system has of itself, and the discovery engine
        # (M7) has no other data source, so it is wired into the cycle rather
        # than offered as an optional extra.
        self.telemetry = Telemetry()
        # The cortex keeps its own counters; it needs the same series everyone
        # else writes to, or its cost would be the one thing with no history.
        self.llm.cortex.telemetry = self.telemetry
        self.world_model.telemetry = self.telemetry
        # Evolution is built before the telemetry exists, so it has to be
        # handed the store here. Without this its `publish_metrics` returns on
        # the first line and all six metrics of Appendix G that describe the
        # evolutionary contour — generation, champion fitness, the valid/test
        # gap, promotions, rollbacks, novelty skips — are never written at all.
        # Nothing errors: the dashboard shows the panel and the series behind
        # it is simply empty.
        self.evolution.telemetry = self.telemetry

        # ── Motivation with teeth (spec M4) ──────────────────────────
        # The chain the development text asks for is goal → value → priority →
        # resource → action. Value and goals already existed; these are the two
        # links that were missing, and the resource link is the one that makes
        # the rest binding: an action that cannot get a lease does not run.
        self.resources = ResourceManager(telemetry=self.telemetry)
        self.roi = ROITracker(telemetry=self.telemetry)
        self.priority = PriorityScheduler(
            resources=self.resources, goal_intelligence=self.goal_intelligence,
            roi=self.roi)
        # The evaluation pool (M9.1). Heavy work — variant scoring, arenas,
        # experiments — runs here rather than in a tick, under a subprocess
        # lease and with a hard timeout.
        self.eval_pool = EvaluationPool(resources=self.resources,
                                        telemetry=self.telemetry)
        self.actions = ActionSpace()
        #: lease id -> the rate-limit entry that reservation overwrote, so
        #: :meth:`release` can put it back (see the note there).
        self._rate_limit_undo: dict[str, int | None] = {}
        #: What the contours were configured with before the pending evolution
        #: candidate was applied — what a rejection reverts to.
        self._genome_before_candidate = None
        #: A generation running detached, so the tick that started it can go on.
        self._evolution_task = None
        #: Which promotion `_watch_champion` has already applied, as
        #: (promotions counter, promoted_at_tick). An identity marker rather
        #: than a genome comparison, because simplex normalisation is not
        #: bit-stable — comparing float dicts would re-adopt the same champion
        #: every tick forever.
        self._adopted_promotion: tuple | None = None
        #: The genome the system is running under. Set from the contours at
        #: boot and thereafter only by `apply_genome`.
        self._genome = Genome()
        # The planner (M2): decisions become a comparison of plans priced by
        # the world model, instead of a ranking of wishes. It proposes only —
        # every gate that can refuse runs after it (Appendix J).
        # The behaviour policy (M3): the fifth link of the development chain,
        # as an object with evidence rather than a whisper in a confidence
        # penalty. Built before the planner because the planner reads it.
        self.policy = BehaviourPolicy(telemetry=self.telemetry)
        self.planner = Planner(
            world_model=self.world_model, actions=self.actions,
            goal_intelligence=self.goal_intelligence, policy=self.policy,
            telemetry=self.telemetry)
        # The reasoning contour (M6). Everything it can reach is handed to it
        # here: a strategy is synthesised data, and a strategy able to reach
        # something nobody passed it would be a strategy able to reach anything.
        self.reasoning = ReasoningEngine(
            cortex=self.llm.cortex, world_model=self.world_model,
            memory=self.memory, graph=self.cognitive_graph, solver=self.solver,
            sandbox=run_skill, telemetry=self.telemetry)
        # The discovery contour (M7). Its only source of observations is the
        # telemetry this substrate writes, which is why it is built after it:
        # the engine explains the system's own history, and there is nothing
        # else it is allowed to read.
        self.discovery = DiscoveryEngine(
            telemetry=self.telemetry, world_model=self.world_model,
            cortex=self.llm.cortex)

        # The genome the contours booted with, recorded once so every later
        # apply/revert has something exact to go back to.
        self._genome = self._genome_from_contours()
        self.evolution.evaluator.pool = self.eval_pool
        if self.evolution.champion is None:
            base_fitness = self._last_benchmark_score if self._last_benchmark_score is not None else 0.0
            self.evolution.register_champion(self._genome.to_dict(), base_fitness)
        # From here on every cortex call needs a lease. This replaces hoping
        # that LLM_MAX_CALLS_PER_RUN was set generously enough.
        self.llm.cortex.resources = self.resources

        self.event_bus.set_veto(self.ethics.veto_check)

        self.tick_count = 0
        self.start_time = CLOCK.now()
        self.running = False
        self.cycle_phase = "idle"
        self.cycle_times: list[float] = []
        self.last_tick_duration = 0.0
        self.llm_thinking = False
        # Everything one pass through the cycle accumulates lives in the tick
        # context; the phases read and write it instead of reaching for a
        # scatter of _tick_* attributes on the substrate (spec §3.9).
        self._ctx = TickContext(tick=0)
        # A forecast made last tick, waiting for this tick's state to reveal
        # where the action actually led (spec M1.6). Lives on the substrate
        # rather than the context because it deliberately spans two of them.
        self._pending_prediction: dict | None = None
        self._ws_broadcast = None
        self._ws_has_clients = None  # callable -> bool, set by the API server
        self._weight_training_task: asyncio.Task | None = None  # detached LoRA run
        self._checkpoint_path = CHECKPOINTS_DIR / "latest.json"

        # Deterministic round-robin counters
        self._learning_source_idx = 0
        self._meta_domain_idx = 0
        self._concept_seed_idx = 0
        self._param_mod_idx = 0

        self._restore_checkpoint()

    # ── per-tick state, kept reachable under its original names ──────
    # These used to be plain attributes. They now live on the tick context, but
    # phases, tests and the dashboard still refer to them by the old names, so
    # they stay addressable exactly as before.

    @property
    def _tick_new_concepts(self) -> int:
        return self._ctx.new_concepts

    @_tick_new_concepts.setter
    def _tick_new_concepts(self, value: int) -> None:
        self._ctx.new_concepts = value

    @property
    def _tick_new_episodic(self) -> int:
        return self._ctx.new_episodic

    @_tick_new_episodic.setter
    def _tick_new_episodic(self, value: int) -> None:
        self._ctx.new_episodic = value

    @property
    def _tick_llm_insights(self) -> int:
        return self._ctx.llm_insights

    @_tick_llm_insights.setter
    def _tick_llm_insights(self, value: int) -> None:
        self._ctx.llm_insights = value

    @property
    def _regulation_directives(self) -> dict:
        return self._ctx.regulation_directives

    @_regulation_directives.setter
    def _regulation_directives(self, value: dict) -> None:
        self._ctx.regulation_directives = value

    @property
    def _pending_experiences(self) -> dict:
        return self._ctx.pending_experiences

    @_pending_experiences.setter
    def _pending_experiences(self, value: dict) -> None:
        self._ctx.pending_experiences = value

    def _restore_checkpoint(self):
        if self._checkpoint_path.exists():
            try:
                # Versioned read: a pre-spec checkpoint has no schema_version and
                # is treated as v1; one written by a newer build is refused
                # rather than half-read (spec §3.2, Appendix I).
                data = read_store(self._checkpoint_path, store="checkpoint")
                # An unreadable or future-versioned file yields empty state, and
                # empty state must LEAVE the runtime alone rather than overwrite
                # a live tick counter with zero. Only keys that are actually
                # present are restored.
                if "tick_count" in data:
                    self.tick_count = data["tick_count"]
                if "version" in data:
                    self.self_mod.current_version = data["version"]
                # Restore the tunable parameters (audit H1) — otherwise every
                # benchmark-verified Evolution win is lost on restart while the
                # persisted champion genome still records the evolved values,
                # desyncing the two.
                saved_params = data.get("parameters")
                if isinstance(saved_params, dict):
                    for k, v in saved_params.items():
                        if k in self.self_mod.parameters:
                            try:
                                self.self_mod.parameters[k] = float(v)
                            except (TypeError, ValueError):
                                pass
                # Restore the configuration the system was running under, so a
                # benchmark-verified win survives a restart.
                #
                # NOT while a candidate is pending: the saved genome is then the
                # UNJUDGED mutation, and adopting it on boot would silently
                # accept a change no benchmark ever scored — the audit R3-4
                # concern, in the terms v2 states it. A restart then falls back
                # to the champion, because that one was measured.
                saved_genome = data.get("genome")
                if isinstance(saved_genome, dict):
                    if self.evolution.candidate is None:
                        self.apply_genome(saved_genome)
                    elif self.evolution.champion:
                        self.apply_genome(self.evolution.champion["genome"])
            except Exception:
                logger.warning("Failed to restore checkpoint %s — starting fresh",
                               self._checkpoint_path, exc_info=True)

    def _save_checkpoint(self):
        data = {
            "tick_count": self.tick_count,
            "version": self.self_mod.current_version,
            "timestamp": CLOCK.now(),
            "uptime": CLOCK.now() - self.start_time,
            "mood": self.emotions.mood,
            "consciousness_mode": self.consciousness.mode,
            "energy": self.emotions.energy,
            # Persist tunable parameters so accepted Evolution mutations survive
            # restart (audit H1). These are the LoRA parameters, which
            # parametric self-modification still tunes; the evolved genome is a
            # separate set and is stored beside them.
            "parameters": dict(self.self_mod.parameters),
            "genome": self.current_genome().to_dict(),
        }
        # Atomic, versioned write — a crash mid-write must not corrupt the
        # checkpoint and silently reset tick_count/version to defaults on next
        # boot, and the stamp lets a later build migrate this file (§3.2).
        write_store(self._checkpoint_path, data)
        self.memory.save()

        # Persist the higher-order systems alongside the checkpoint.
        for system in (self.world_model, self.cognitive_graph, self.evolution,
                       self.goal_intelligence, self.feedback_loop,
                       self.resources, self.roi, self.policy,
                       self.reasoning, self.discovery):
            try:
                system.save()
            except Exception:
                logger.exception("Failed to persist %s", type(system).__name__)

        # Buffered metrics and cached model responses land on disk with the
        # checkpoint, so a restart never loses more history than it loses state.
        try:
            self.telemetry.flush()
        except Exception:
            logger.exception("Failed to flush telemetry")
        try:
            self.llm.cortex.save()
        except Exception:
            logger.exception("Failed to persist the cortex cache")

    def regulate_capacity(self):
        """Size the learned structures by measured cost, not by a guess.

        MAX_LINKS / MAX_NODES were hand-picked constants: nothing derived them,
        and the right value depends on the machine. Tick latency is already
        measured and already has a threshold, so it is used as the control
        signal — capacity grows while ticks are comfortably cheap and is handed
        back when they are not. Bounded on both sides: never below the
        configured baseline, never above CAPACITY_MAX_MULTIPLE of it.
        """
        durations = list(self.health.tick_durations)
        if not durations:
            return  # no measurements yet — no evidence, so no change
        avg_ms = sum(durations) / len(durations)
        threshold = self.health.thresholds["tick_duration_ms"]

        if avg_ms < threshold * CAPACITY_HEADROOM:
            factor = CAPACITY_GROWTH_FACTOR
        elif avg_ms > threshold:
            factor = CAPACITY_SHRINK_FACTOR
        else:
            return  # inside the comfort band — hold

        for structure, attr, baseline in (
            (self.world_model, "max_links", WM_MAX_LINKS),
            (self.cognitive_graph, "max_nodes", CG_MAX_NODES),
            (self.policy.store, "max_preferences", cfg.POLICY_MAX_PREFS),
        ):
            try:
                current = getattr(structure, attr)
                scaled = int(max(baseline,
                                 min(baseline * CAPACITY_MAX_MULTIPLE, current * factor)))
                setattr(structure, attr, scaled)
            except Exception:
                logger.exception("Capacity regulation failed for %s.%s",
                                 type(structure).__name__, attr)

    # ── motivation: priority → resource → action (spec M4.6) ─────────

    def acquire(self, action: str, priority: float | None = None):
        """Reserve the resources one action needs, or return None.

        None means the action does not happen this tick. That is the whole
        point of the resource link: a motivated action that cannot be paid for
        is deferred, not quietly executed anyway.
        """
        spec = self.actions.by_name.get(action)
        if spec is None:
            logger.warning("Unknown action %r requested a lease", action)
            return None
        if priority is None:
            priority = self.priority.priority(
                Candidate(objective=action, drive=spec.drive,
                          cost=spec.cost, safety_critical=spec.safety_critical),
                self._ctx)
        lease = self.resources.reserve(
            spec.reservation_cost(), action, priority,
            safety_critical=spec.safety_critical)
        if lease is None:
            self.actions.note_blocked("resources")
        else:
            # Remember what the rate limit said before this reservation, so a
            # lease handed back unused can restore it. Charging the interval at
            # reservation time and never undoing it would silence an action for
            # `min_interval` ticks that it never actually ran in — for
            # `train_weights` a single ethics veto would cost 1000 ticks.
            self._rate_limit_undo[lease.id] = self.actions.last_run.get(action)
            self.actions.mark_run(spec, self.tick_count)
        return lease

    # ── the genome: evolution's only way into the running system ─────

    def current_genome(self) -> Genome:
        """The configuration the contours are running under.

        Read from what was last applied rather than by interrogating every
        contour, because not every gene has a reader yet: a gene declared for a
        later stage is applied to nothing, and a read-back would hand it its
        default instead of the value it was actually set to. That made a revert
        lossy — the rejected variant's value survived in the genes nobody could
        read back — which is exactly the failure the whole-genome revert exists
        to prevent.
        """
        return Genome(self._genome)

    def _synth_attempts(self) -> int:
        """How many times a skill proposal may be tried (gene ``synth_attempts``).

        This exists because the gene did not: the loop below used a literal
        ``range(2)`` while the gene declared ``Substrate._skill_synthesis`` as
        its reader, so evolution searched over a number nothing consumed. The
        default is 2 — initial attempt plus one repair — so behaviour is
        unchanged for a system running the default genome.
        """
        try:
            return max(1, int(self._genome.get("synth_attempts", 2)))
        except (TypeError, ValueError):
            return 2

    def _genome_from_contours(self) -> Genome:
        """What the contours say they are configured with.

        Every observable gene is read back from the object that consumes it —
        not from ``self._genome``, which is only a record of what was handed
        out. That distinction is the whole point: a read-back from the stored
        copy cannot tell a gene that reached its contour from one that was
        dropped on the floor, and the sensitivity gate in
        ``test_evolution_population.py`` is built on this method for exactly
        that reason.
        """
        genome = Genome()
        genome.update_values({
            "plan_beam": self.planner.beam,
            "plan_depth": self.planner.depth,
            "plan_discount": self.planner.discount,
            "w_ev": self.planner.weights["ev"],
            "w_val": self.planner.weights["val"],
            "w_exp": self.planner.weights["exp"],
            "w_cost": self.planner.weights["cost"],
            "w_risk": self.planner.weights["risk"],
            "policy_weight": self.policy.store.weight,
            "policy_min_support": self.policy.miner.min_support,
            "solver_timeout": self.solver.timeout,
            "solver_order": self.solver.order,
            "wm_smoothing": self.world_model.transitions.smoothing,
            "wm_half_life": self.world_model.transitions.half_life,
            "explore_bonus": self.world_model.simulator.explore_bonus,
            "mem_retention_bias":
                self.world_model.causal.failure_retention_bias,
            "synth_attempts": self._synth_attempts(),
            # The reasoning genes are resolved by the interpreter at each step
            # (`$gene:...`), so the interpreter's own copy is where they are
            # actually read from.
            "reason_budget": self.reasoning._budget(),
            "reason_decompose_parts": self.reasoning.interpreter.genome.get(
                "reason_decompose_parts", 6),
            "reason_vote_n": self.reasoning.interpreter.genome.get(
                "reason_vote_n", 1),
            **{f"priority_w_{name}": value
               for name, value in self.priority.weights.items()},
            **{f"res_share_{drive}": share
               for drive, share in self.roi.shares.items()},
        })
        return genome

    def apply_genome(self, genome) -> Genome:
        """Configure every contour that reads a gene. Returns what was replaced.

        This is the *only* path from evolution into the running system, and it
        returns the previous genome rather than nothing, because a rejected
        variant has to be revertible in full. A coordinate mutation moves every
        gene at once (§M5.4), so reverting the single gene that moved furthest
        would leave the rest of a rejected variant permanently in place.

        Genes whose contour has not landed are applied to nothing and simply
        wait — the same rule the action registry uses for pending stages.
        """
        previous = Genome(self._genome)
        genome = Genome(genome)
        for target, method in ((self.planner, "set_weights"),
                               (self.priority, "set_weights"),
                               (self.policy, "set_genome"),
                               (self.reasoning, "set_genome"),
                               (self.solver, "set_genome"),
                               (self.world_model, "apply_genome"),
                               (self.roi, "set_genome")):
            try:
                getattr(target, method)(genome.to_dict())
            except Exception:
                logger.exception("Applying the genome to %s failed",
                                 type(target).__name__)
        self._genome = genome
        return previous

    def release(self, lease, *, keep_rate_limit: bool = False) -> None:
        """Hand back a lease for an action that will not run under it.

        The resources go back to the budget and the rate limit goes back to
        what it was: ``min_interval`` counts executions, not intentions.

        ``keep_rate_limit`` is for the one case where the action *is* going to
        happen, just not on this reservation — a scheduled block owns it and
        takes its own lease. There the interval has been earned, and restoring
        it would let the planner re-pick an action every tick that only runs on
        its own cadence.
        """
        if lease is None:
            return
        previous = self._rate_limit_undo.pop(lease.id, _NO_PREVIOUS_RUN)
        if not keep_rate_limit and previous is not _NO_PREVIOUS_RUN:
            if previous is None:
                self.actions.last_run.pop(lease.purpose, None)
            else:
                self.actions.last_run[lease.purpose] = previous
        self.resources.release(lease)

    def settle(self, lease, actual: ResourceCost | None = None,
               value: float = 0.0) -> None:
        """Close a lease with what was really spent, and score its return.

        Committing the real usage rather than the estimate is what keeps the
        budget honest in both directions: an action that reserved generously
        and used little hands the difference back.
        """
        if lease is None:
            return
        # The action ran, so the rate limit stands and its undo entry is spent.
        self._rate_limit_undo.pop(lease.id, None)
        spec = self.actions.by_name.get(lease.purpose)
        spent = actual if actual is not None else (lease.committed or lease.cost)
        self.resources.commit(lease, spent)
        if spec is not None:
            try:
                self.roi.record(lease.purpose, spent, value, drive=spec.drive)
            except Exception:
                logger.exception("ROI accounting failed for %s", lease.purpose)

    def _publish_tick_metrics(self, tick_ms: float) -> None:
        """Write this tick's observations into the time series.

        Called from ``tick()`` under its own guard: a telemetry failure is a
        lost data point, never a lost tick. Values here are the ones every tick
        produces; each contour publishes its own metrics from its own phase.
        """
        try:
            tel, tick = self.telemetry, self.tick_count
            tel.record(M.TICK_DURATION_MS, tick_ms, tick)
            for phase, ms in sorted(self._ctx.durations_ms.items()):
                tel.record(M.TICK_PHASE_MS, ms, tick, tags={"phase": phase})
            tel.record(M.REWARD_VALUE, self._compute_reward(), tick)
            tel.record(M.REWARD_ENV_ROLLING, self.environment.rolling_reward(), tick)
            if self._last_benchmark_score is not None:
                tel.record(M.BENCH_SCORE, self._last_benchmark_score, tick)
            for kind, counts in sorted(
                    (self.evaluator.last_report or {}).get("per_kind", {}).items()):
                total = counts.get("total", 0)
                if total:
                    tel.record(M.BENCH_PER_KIND, counts.get("passed", 0) / total,
                               tick, tags={"kind": kind})
            tel.record(M.MEM_SEMANTIC, len(self.memory.semantic), tick)
            tel.record(M.MEM_EPISODIC, len(self.memory.episodic), tick)
            tel.record(M.GRAPH_NODES, len(self.cognitive_graph.nodes), tick)
            tel.record(M.HEALTH_STATUS_CODE,
                       M.health_code(self.health.last_status), tick)
            tel.record(M.HEALTH_CONSECUTIVE_ERRORS,
                       self.health.consecutive_errors, tick)
            self.llm.cortex.publish_metrics(tick)
            self.resources.publish_metrics(tick)
            self.roi.publish_metrics(tick)
            self.world_model.publish_metrics(tick)
            self.planner.publish_metrics(tick)
            self.policy.publish_metrics(tick)
            self.reasoning.publish_metrics(tick)
            self.discovery.publish_metrics(tick)
        except Exception:
            logger.exception("Telemetry publication failed")

    def _is_llm_tick(self) -> bool:
        return self.llm.enabled and self.tick_count % LLM_THINK_EVERY_N_TICKS == 0

    def _compute_reward(self) -> float:
        """Reward driven by EXTERNAL, verifiable task performance.

        Primary signal: the benchmark pass-rate (held-out fitness) blended with
        the live environment's rolling reward. Both come from a deterministic
        verifier, not self-report. Until the first benchmark has run we fall back
        to the legacy synthetic estimate so the system still functions on boot.
        """
        env_reward = self.environment.rolling_reward()
        bench = self._last_benchmark_score

        if bench is not None or self.environment.total_steps > 0:
            bench = bench if bench is not None else env_reward
            # 70% held-out capability, 30% live environment outcomes.
            reward = 0.7 * bench + 0.3 * env_reward
            return max(0.0, min(1.0, reward))

        # Fallback (pre-first-eval): legacy synthetic estimate.
        total_ticks = self.health.successful_ticks + self.health.failed_ticks
        error_rate = self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
        success_component = self.emotions.success_rate * 0.3
        energy_component = self.emotions.energy * 0.2
        knowledge_component = min(1.0, len(self.memory.semantic) / 500) * 0.3
        error_component = (1.0 - error_rate) * 0.2
        return max(0.0, min(1.0, success_component + energy_component + knowledge_component + error_component))

    def _compute_confidence(self) -> float:
        """Compute decision confidence from real system state (no randomness)."""
        # Base: success rate from emotions (running average of real rewards)
        base = self.emotions.success_rate

        # Modifier from energy
        energy_factor = 0.5 + 0.5 * self.emotions.energy

        # Modifier from emotional state
        emotional_factor = self.emotions.emotional_modifier()

        # Modifier from decision history consistency
        history_factor = 1.0
        if len(self.introspection.decision_trace) >= 5:
            recent = self.introspection.decision_trace[-5:]
            confs = [d.get("confidence", 0.5) for d in recent]
            avg = sum(confs) / len(confs)
            variance = sum((c - avg) ** 2 for c in confs) / len(confs)
            history_factor = 1.0 - min(variance, 0.5)  # lower variance = higher confidence

        confidence = base * energy_factor * emotional_factor * history_factor
        return max(0.1, min(0.99, confidence))

    def _compute_importance(self) -> float:
        """Compute event importance from tick context (no randomness)."""
        base = 0.3
        # Higher importance when LLM produced insights
        if self._tick_llm_insights > 0:
            base += 0.2
        # Higher importance when new knowledge was acquired
        if self._tick_new_concepts > 0:
            base += 0.15
        # Higher importance at checkpoint ticks
        if self.tick_count % CHECKPOINT_EVERY_N_TICKS == 0:
            base += 0.1
        # Lower importance during low energy
        if self.emotions.energy < 0.3:
            base -= 0.1
        return max(0.1, min(1.0, base))





    async def _run_benchmark(self, started_tick: int | None = None):
        """Run the held-out benchmark off the tick loop and update the reward signal."""
        try:
            report = await asyncio.get_running_loop().run_in_executor(None, self.evaluator.run)
            self._last_benchmark_score = report["score"]
            self.autobiography.log_event(
                "benchmark", f"score={report['score']:.3f} ({report['passed']}/{report['total']})", 0.7)

            # ── System 3: natural selection. If a mutated genome is pending,
            #    the fresh benchmark is its fitness — but ONLY judge with a
            #    benchmark that STARTED after the mutation was applied, else the
            #    run never measured the mutated state (audit M3). A candidate
            #    that appeared after this benchmark started waits for the next.
            cand = self.evolution.candidate
            if cand is not None and (
                    started_tick is None
                    or cand.get("proposed_at_tick", -1) < started_tick):
                verdict = self.evolution.judge_candidate(report["score"])
                if verdict.get("decision") == "rejected":
                    # Put the whole genome back, not just the reported gene: a
                    # coordinate mutation moves every gene, and reverting only
                    # the largest move would leave the rest of the rejected
                    # variant in place for good.
                    param = verdict.get("param")
                    if self._genome_before_candidate is not None:
                        self.apply_genome(self._genome_before_candidate)
                    self.autobiography.log_event(
                        "evolution", f"Mutation of {param} rejected — reverted", 0.5)
                elif verdict.get("decision") == "accepted":
                    self.autobiography.log_event(
                        "evolution",
                        f"Mutation of {verdict['param']} accepted — new champion "
                        f"(fitness={report['score']:.3f})",
                        0.85)

            await self.event_bus.publish(Event(
                source=Layer.INTROSPECTION, target=None,
                event_type="benchmark", payload={"score": report["score"]},
            ))
        except Exception:
            logger.exception("Benchmark run failed")

    async def _evolution_step(self):
        """System 3 — propose version B: mutate ONE champion parameter.

        Unlike the synthetic parametric self-mod path, the acceptance gate here
        is the REAL held-out benchmark (judged later in _run_benchmark), so the
        mutation is applied directly after two hard safety checks
        (self-preservation + ethics) and bounds-clamping — we deliberately do
        NOT run it through sandbox_test's synthetic degradation gate, which
        would reject changes before the benchmark could ever measure them.
        A mutation that fails a safety check is abandoned (nothing was applied).
        """
        try:
            mutation = self.evolution.propose_mutation(self.tick_count)
            if not mutation:
                return
            param, new_val = mutation["param"], mutation["new_value"]
            if param not in GENES_BY_NAME:
                # A gene the schema does not know cannot be applied to anything.
                self.evolution.abandon_candidate()
                return

            # Hard safety guards (same as _llm_parametric_modification).
            safe, _ = self.self_preservation.is_modification_safe(
                f"evolution/{param}", str(new_val))
            eth = self.ethics.evaluate_action({
                "type": "self_modification", "modifies_self": True,
                "confidence": self._compute_confidence(),
            })
            if not safe or eth["status"] == "blocked":
                self.evolution.abandon_candidate()
                self.autobiography.log_event("evolution", f"Mutation of {param} blocked by safety", 0.5)
                return

            # The gene's own range is the clamp — `GeneSpec.clamp` is the single
            # place a legal value is defined, so a mutation, a crossover, a
            # cortex proposal and a migrated file all end up bounded the same
            # way. The candidate genome is then applied to the contours that
            # read it, and the previous genome is what a rejection reverts to.
            new_val = GENES_BY_NAME[param].clamp(new_val)
            self.evolution.candidate["new_value"] = new_val
            self.evolution.candidate["genome"][param] = new_val
            self._genome_before_candidate = self.apply_genome(
                self.evolution.candidate["genome"])
            self.autobiography.log_event(
                "evolution",
                f"Proposed mutation: {param} {mutation['old_value']} -> {new_val} "
                f"(gen {self.evolution.generation}, awaiting benchmark)",
                0.6)
        except Exception:
            logger.exception("Evolution step failed")
            self.evolution.abandon_candidate()

    def _watch_champion(self):
        """The other half of §M5.6: apply a promoted champion, then watch it.

        ``run_generation`` deliberately mutates nothing outside the engine, and
        both of its callers discard the result — the executor runs it detached
        and the API returns it to HTTP. The reflect phase would be the natural
        call site, but the phases are owned elsewhere, so the substrate closes
        the loop itself, once per tick:

        * a champion the engine promoted but nobody applied is put live here,
          through the same two hard gates every self-modification passes
          (``is_modification_safe`` + ethics) — and withdrawn if they refuse;
        * the live metric is fed to :meth:`EvolutionEngine.watch`, which is
          what makes the rollback promise of the module docstring real rather
          than dead code;
        * a rollback the watch decides on is applied back to the contours,
          because "it is put back" has to mean the running system, not a field
          on the engine.

        Skipped entirely while a v1 candidate is pending: the live genome is
        then the unjudged mutation, and both adopting a champion over it and
        rolling back "through" it would clobber the experiment in flight.
        """
        evo = self.evolution
        try:
            if evo.candidate is not None:
                return
            promotion = (evo.promotions, evo.promoted_at_tick)
            if evo.previous_champion is not None and evo.champion \
                    and self._adopted_promotion != promotion:
                wanted = Genome(evo.champion["genome"]).to_dict()
                safe, _ = self.self_preservation.is_modification_safe(
                    "evolution/champion", str(sorted(wanted.items())))
                eth = self.ethics.evaluate_action({
                    "type": "self_modification", "modifies_self": True,
                    "confidence": self._compute_confidence(),
                })
                if safe and eth["status"] != "blocked":
                    self.apply_genome(wanted)
                    self._adopted_promotion = promotion
                    self.autobiography.log_event(
                        "evolution", "Promoted champion applied to the "
                        "running system (watch begins)", 0.7)
                else:
                    evo.refuse_promotion("blocked by the safety pipeline")
                    self.autobiography.log_event(
                        "evolution", "Promoted champion refused by the "
                        "safety pipeline — previous champion kept", 0.7)
                    return
            record = evo.watch(self.tick_count, self._compute_reward())
            if record is not None and evo.champion:
                self.apply_genome(evo.champion["genome"])
                self.autobiography.log_event(
                    "evolution",
                    f"Champion rolled back after a live drop of "
                    f"{record.get('drop', 0.0):.4f}", 0.85)
        except Exception:
            logger.exception("The champion watch failed")

    async def _skill_synthesis(self):
        """Propose a skill for a failing kind from TRAIN examples and keep it only
        if it raises the HELD-OUT pass-rate (real, generalizing self-improvement).

        The proposer sees only the train split; the acceptance gate scores the
        candidate on held-out tasks it never saw, so memorizing the shown cases
        does not pass. A failed proposal gets one repair attempt with the failing
        example fed back."""
        try:
            failing = await asyncio.get_running_loop().run_in_executor(None, self.evaluator.failing_kinds)
            if not failing:
                return
            kind = failing[self._skill_synth_idx % len(failing)]
            self._skill_synth_idx += 1

            train, holdout = split_tasks(kind)
            train_examples = [{"payload": t.payload, "expected": t.expected} for t in train]

            before = await asyncio.get_running_loop().run_in_executor(
                None, self.evaluator.pass_rate_on, holdout)

            code = await self.llm.propose_skill(kind, train_examples)
            kept = False
            for attempt in range(self._synth_attempts()):
                if not code:
                    break
                self._skill_synth_count += 1
                name = f"{kind}_llm_{self._skill_synth_count}"
                skill = Skill(name=name, kinds=[kind], code=code, origin="llm")
                added, msg = self.skill_library.add(skill)
                if not added:
                    self.autobiography.log_event("skill_rejected", f"{kind}: {msg[:60]}", 0.5)
                    break

                after, failed_case = await asyncio.get_running_loop().run_in_executor(
                    None, self._score_holdout, holdout)
                if after > before:
                    self.skill_library.save()
                    self.autobiography.log_event(
                        "skill_learned", f"{kind}: holdout {before:.2f} -> {after:.2f}", 0.9)
                    self.motor.execute("alert", payload={
                        "level": "info",
                        "message": f"Learned generalizing skill for '{kind}' ({before:.2f}->{after:.2f})"})
                    kept = True
                    break

                # No generalization — discard and repair from a failing case. The
                # failing case is fed back; re-asking with the exact same prompt
                # (the original behaviour) could only return the same code, so
                # the "repair attempt" never repaired anything.
                #
                # The condition is "there is another attempt left", not
                # "attempt == 0": with the literal, `synth_attempts` above 2 was
                # bounded by the loop but could never spend the extra rounds,
                # because the second failure always set `code = None`. Half a
                # wiring is what makes a gene look connected while its upper
                # range does nothing.
                self.skill_library.remove(name)
                if attempt + 1 < self._synth_attempts() and train_examples:
                    code = await self.llm.propose_skill(
                        kind, train_examples,
                        feedback=self._repair_feedback(code, failed_case))
                else:
                    code = None

            if not kept:
                self.autobiography.log_event(
                    "skill_discarded", f"{kind}: no generalizing improvement ({before:.2f})", 0.5)
        except Exception:
            logger.exception("Skill synthesis failed")

    def _score_holdout(self, holdout) -> tuple[float, dict | None]:
        """Held-out pass-rate PLUS the first case that failed.

        ``Evaluator.pass_rate_on`` only returns the ratio, which is enough to
        accept or reject a candidate but not enough to repair it. Naming the
        failing case is what turns the retry into an actual second attempt.
        """
        if not holdout:
            return 0.0, None
        passed = 0
        first_failure: dict | None = None
        for task in holdout:
            result = self.evaluator.solver.solve(task)
            if result.solved:
                passed += 1
            elif first_failure is None:
                first_failure = {
                    "payload": task.payload,
                    "expected": task.expected,
                    # last_answer, not answer: on failure `answer` is None by
                    # definition, and "you answered None" tells a repair
                    # attempt nothing about what went wrong.
                    "got": result.last_answer,
                }
        return passed / len(holdout), first_failure

    @staticmethod
    def _repair_feedback(code: str | None, failed_case: dict | None) -> str:
        """Human-readable description of why the last candidate was discarded."""
        parts = []
        if code:
            parts.append(f"Previous implementation:\n{code[:800]}")
        if failed_case:
            parts.append(
                f"It failed on a held-out case: payload={failed_case['payload']!r} "
                f"expected {failed_case['expected']!r}, got {failed_case['got']!r}."
            )
        else:
            parts.append("It did not improve the held-out pass rate.")
        return "\n".join(parts)

    async def _learning_cycle(self):
        """Pick the most useful learning action this round, in priority order:
        1. close a failing payload->answer kind, 2. learn an unsolved coding
        task, 3. simplify an already-solved kind (versioning)."""
        try:
            failing = await asyncio.get_running_loop().run_in_executor(None, self.evaluator.failing_kinds)
            if failing:
                await self._skill_synthesis()
                return
            unsolved_coding = await asyncio.get_running_loop().run_in_executor(
                None, self.evaluator.unsolved_coding)
            if unsolved_coding:
                await self._coding_synthesis(unsolved_coding)
                return
            await self._skill_optimization()
        except Exception:
            logger.exception("Learning cycle failed")

    async def _coding_synthesis(self, unsolved):
        """Learn an unsolved CODING task: implement from spec+visible tests, then
        keep the solution only if it passes the HIDDEN tests (step 1)."""
        from aegis.eval.coding import verify_solution
        task = unsolved[self._skill_synth_idx % len(unsolved)]
        self._skill_synth_idx += 1
        code = await self.llm.propose_coding_solution(
            task.func_name, task.spec, [list(vt) for vt in task.visible_tests])
        if not code:
            return
        verdict = await asyncio.get_running_loop().run_in_executor(None, verify_solution, code, task)
        if verdict["solved"]:
            self._skill_synth_count += 1
            skill = Skill(name=f"sol_{task.id}_{self._skill_synth_count}",
                          kinds=[task.kind_key()], code=code, func=task.func_name, origin="llm")
            added, _ = self.skill_library.add(skill)
            if added:
                self.autobiography.log_event(
                    "coding_solved", f"{task.id}: passed {verdict['total']} hidden tests", 0.9)
                self.motor.execute("alert", payload={
                    "level": "info", "message": f"Solved coding task '{task.id}'"})
        else:
            self.autobiography.log_event(
                "coding_failed", f"{task.id}: {verdict['passed']}/{verdict['total']} hidden tests", 0.5)

    async def _skill_optimization(self):
        """Versioning (step 2): for an already-solved kind, try to learn a SIMPLER
        skill (shorter code) that still passes the held-out tests, and retire the
        longer one. Correctness is preserved; the secondary metric is simplicity."""
        kinds = [k for k in self.skill_library.status()["kinds_covered"] if not k.startswith("code:")]
        if not kinds:
            return
        kind = kinds[self._skill_synth_idx % len(kinds)]
        self._skill_synth_idx += 1
        current = [s for s in self.skill_library.for_kind(kind)]
        if not current:
            return
        incumbent = min(current, key=lambda s: len(s.code))
        train, holdout = split_tasks(kind)
        code = await self.llm.propose_skill(
            kind, [{"payload": t.payload, "expected": t.expected} for t in train])
        if not code or len(code) >= len(incumbent.code):
            return  # not simpler
        self._skill_synth_count += 1
        cand = Skill(name=f"{kind}_opt_{self._skill_synth_count}", kinds=[kind], code=code, origin="llm")

        # Evaluate the candidate IN ISOLATION against ALL tasks the incumbent
        # covers (train + holdout), running its own code directly rather than the
        # library solver. Two reasons: (1) going through the solver while the
        # incumbent is still present would always report a passing rate, so a
        # broken candidate could retire the working skill; (2) verifying only the
        # holdout (often a single task) then retiring every longer skill for the
        # whole kind would let the candidate regress the train tasks it never
        # had to pass.
        check_tasks = list(train) + list(holdout)

        def _candidate_passes() -> bool:
            for t in check_tasks:
                out = run_skill(code, cand.func, t.payload, timeout=self.evaluator.solver.timeout)
                if not (bool(out.get("ok")) and t.verify(out.get("result"))):
                    return False
            return True

        if not check_tasks:
            return  # nothing to verify against — never retire on no evidence
        passes = await asyncio.get_running_loop().run_in_executor(None, _candidate_passes)
        if not passes:
            return  # candidate is not correct — keep the incumbent, discard it

        added, _ = self.skill_library.add(cand)
        if not added:
            return
        # Retire all longer skills for this kind — the simpler, verified one wins.
        for s in current:
            if len(s.code) > len(code):
                self.skill_library.remove(s.name)
        self.skill_library.save()
        self.autobiography.log_event(
            "skill_optimized", f"{kind}: simpler skill {len(incumbent.code)}->{len(code)} chars", 0.7)

    async def _weight_training_cycle(self):
        """Run a full weight-modification cycle off the tick loop.

        Errors are logged, never propagated — a failed/blocked training run must
        not crash the substrate or leave a dangling task that blocks future runs.
        """
        try:
            # System 5 reaches the training data here: the experience log is the
            # only source that carries WHY an outcome happened.
            wmod_result = await self.self_mod.propose_weight_modification(
                self.memory, self.agent_system, self.ethics,
                feedback_loop=self.feedback_loop,
            )
            status_str = wmod_result.get("status", "unknown")
            self.autobiography.log_event(
                "weight_training",
                f"Weight modification result: {status_str}, "
                f"loss={wmod_result.get('train_loss', '?')}",
                0.9 if status_str == "applied" else 0.6,
            )
            self.motor.execute("alert", payload={
                "level": "info" if status_str == "applied" else "warning",
                "message": f"Weight training: {status_str}",
            })
        except Exception as e:
            logger.exception("Background weight-training cycle failed")
            self.autobiography.log_event("weight_training", f"Training cycle error: {str(e)[:80]}", 0.6)

    async def _llm_parametric_modification(self, lease=None) -> bool:
        """Use LLM to analyze performance and propose parameter adjustments.

        Returns whether any adjustment was actually applied — that is the value
        the lease bought, and it is what the ROI tracker scores.
        """
        total_ticks = self.health.successful_ticks + self.health.failed_ticks
        error_rate = self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0

        metrics = {
            "parameters": self.self_mod.parameters,
            "success_rate": self.emotions.success_rate,
            "error_rate": error_rate,
            "energy": self.emotions.energy,
            "semantic_concepts": len(self.memory.semantic),
            "information_gain": self.goals.information_gain,
            "goals_completed": sum(1 for g in self.goals.goals if g.status == "completed"),
            "tick": self.tick_count,
        }

        result = await self.llm.analyze_self_performance(metrics, lease=lease)
        if not result.get("success") or not result.get("parsed"):
            return False

        applied_any = False
        adjustments = result["parsed"].get("adjustments", [])
        for adj in adjustments[:2]:  # max 2 adjustments per cycle
            param = adj.get("parameter", "")
            direction = adj.get("direction", "")
            magnitude = min(0.1, abs(adj.get("magnitude", 0.05)))

            if param not in self.self_mod.parameters:
                continue

            old_val = self.self_mod.parameters[param]
            if direction == "increase":
                new_val = old_val * (1 + magnitude)
            elif direction == "decrease":
                new_val = old_val * (1 - magnitude)
            else:
                continue

            # The untouchable set is asked BEFORE anything else (Appendix B).
            # Parametric self-modification is one of the seven contours that can
            # change the running system, so it owes this check like the rest.
            verdict = immutable.check_change(f"parametric/{param}", old_val, new_val)
            if not verdict.allowed:
                self.autobiography.log_event(
                    "mod_blocked", f"Immutable parameter refused: {verdict.reason}", 0.8)
                continue
            if verdict.clamped and verdict.value is not None:
                new_val = verdict.value

            proposal = self.self_mod.propose_modification("parametric", param, new_val)

            # Safety check
            safe, safety_report = self.self_preservation.is_modification_safe(
                f"parametric/{param}", str(new_val)
            )
            if not safe:
                self.autobiography.log_event("mod_blocked", f"Self-preservation blocked: {param}", 0.7)
                continue

            eth_check = self.ethics.evaluate_action({
                "type": "self_modification",
                "modifies_self": True,
                "confidence": self._compute_confidence(),
            })
            if eth_check["status"] == "blocked":
                continue

            # Real sandbox test with current system metric
            current_metric = self._compute_reward()
            sandbox = self.self_mod.sandbox_test(proposal, current_metric)
            result = self.self_mod.apply_modification(proposal, sandbox)

            if result.get("applied"):
                applied_any = True
                self.autobiography.log_event(
                    "self_mod",
                    f"Parameter {param}: {old_val:.6f} -> {new_val:.6f} (LLM: {adj.get('reason', '')[:50]})",
                    0.7,
                )
        return applied_any

    async def _code_self_modification(self):
        """Use LLM to analyze and modify own source code."""
        # Defense in depth: never rewrite source unless explicitly enabled by the
        # operator, even if this method is reached directly (audit C2).
        if not CODE_SELF_MOD_ENABLED:
            return
        sources = self.code_modifier.list_sources()
        # The LLM is asked to return the COMPLETE rewritten file, so restrict to
        # files small enough to regenerate whole within the modification-size cap
        # (larger files would be truncated and rejected — see audit #10).
        modifiable = [s for s in sources
                      if not s["immutable"] and s["size"] <= CODE_MOD_MAX_FILE_CHARS]
        if not modifiable:
            logger.info("No modifiable source files within size budget for code mod")
            return

        # Round-robin through source files
        target = modifiable[self._code_file_index % len(modifiable)]
        self._code_file_index += 1

        # Ethics check first
        total_ticks = self.health.successful_ticks + self.health.failed_ticks
        error_rate = self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
        eth_result = self.ethics.evaluate_code_modification({
            "target_file": target["path"],
            "energy": self.emotions.energy,
            "error_rate": error_rate,
            "health_status": self.health.check().get("status", "ok"),
            "modification_size": 0,  # unknown yet
        })
        if eth_result["status"] == "blocked":
            logger.info(f"Code mod blocked by ethics for {target['path']}")
            return

        # Read the source
        try:
            source_code = self.code_modifier.read_source(target["path"])
        except Exception as e:
            logger.warning(f"Cannot read {target['path']}: {e}")
            return

        # Ask LLM to propose a change
        state = {
            "tick": self.tick_count,
            "energy": self.emotions.energy,
            "error_rate": error_rate,
        }
        llm_result = await self.llm.propose_code_change(target["path"], source_code, state)
        if not llm_result.get("success") or not llm_result.get("parsed"):
            return

        parsed = llm_result["parsed"]
        if not parsed.get("should_modify"):
            logger.info(f"LLM decided no modification needed for {target['path']}: {parsed.get('reason', '')[:80]}")
            return

        modified_code = parsed.get("modified_code", "")
        description = parsed.get("description", "LLM-proposed improvement")

        if not modified_code or modified_code == source_code:
            return

        # Re-check ethics with the FULL proposed code (not a 500-char prefix —
        # dangerous patterns can hide past the truncation point).
        eth_result2 = self.ethics.evaluate_code_modification({
            "target_file": target["path"],
            "proposed_code": modified_code,
            "energy": self.emotions.energy,
            "error_rate": error_rate,
            "health_status": self.health.check().get("status", "ok"),
            "modification_size": abs(len(modified_code) - len(source_code)),
        })
        if eth_result2["status"] == "blocked":
            logger.info(f"Code mod blocked by ethics (re-check) for {target['path']}")
            return

        # Self-preservation guard — the strongest safety net (lethal patterns,
        # critical-element retention, drastic-shrink detection). This was
        # previously only wired into parametric mods; it must gate code rewrites.
        # SelfPreservation.base_dir is the repo root, so pass the repo-relative
        # path ("aegis/..") to engage the critical-module AND size checks.
        safe, sp_report = self.self_preservation.is_modification_safe(
            f"aegis/{target['path']}", modified_code
        )
        if not safe:
            self.autobiography.log_event(
                "code_mod_blocked",
                f"Self-preservation blocked {target['path']}: {sp_report['critical'][:2]}",
                0.85,
            )
            logger.warning(f"Code mod blocked by self-preservation: {sp_report['critical']}")
            return

        # Apply the modification
        mod_result = self.code_modifier.apply_modification(
            target["path"], modified_code, description, author="llm"
        )

        self._code_mod_count_session += 1

        if mod_result["status"] in ("applied", "applied_pending_restart"):
            self.autobiography.log_event(
                "code_mod",
                f"Modified {target['path']} (pending restart): {description[:80]}",
                0.9,
            )
            self.motor.execute("alert", payload={
                "level": "info",
                "message": f"Code self-modification (pending restart): {target['path']} — {description[:60]}",
            })
            # Bump patch version via the tolerant parser (audit L1) — a
            # restored/foreign checkpoint may hold a non-"x.y.z" version, and the
            # naive split/int() crashed the tick on it.
            self.self_mod.current_version = self.self_mod._bump_patch(
                self.self_mod.current_version)

            logger.info(f"Code self-modification applied: {target['path']} — {description}")
        else:
            self.autobiography.log_event(
                "code_mod_fail",
                f"Failed to modify {target['path']}: {mod_result.get('status')} — {mod_result.get('error', '')[:60]}",
                0.6,
            )
            logger.warning(f"Code mod failed: {mod_result}")


    # ── COGNITIVE CYCLE ──────────────────────────────────────────────
    # The phase bodies live in aegis/layers/phases/ (spec §3.9). These
    # delegating wrappers keep the substrate's own surface intact: the phases
    # are still reachable as `_perceive()` .. `_reflect()`, so callers and the
    # existing suite do not have to know the cycle was split up.

    async def _perceive(self):
        await perceive.run(self, self._ctx)

    def _update_archetypes(self):
        perceive.update_archetypes(self)

    async def _evaluate(self):
        await evaluate.run(self, self._ctx)

    async def _decide(self):
        await decide.run(self, self._ctx)

    async def _act(self):
        await act.run(self, self._ctx)

    async def _reflect(self):
        await reflect.run(self, self._ctx)

    async def _run_phase(self, name: str, runner):
        """Run one phase and record what it cost.

        Timing lives here rather than in each phase so every phase is measured
        the same way, and so a phase cannot forget to report itself.
        """
        started = CLOCK.monotonic()
        try:
            await runner()
        finally:
            elapsed_ms = (CLOCK.monotonic() - started) * 1000
            self._ctx.record_duration(name, elapsed_ms)
            self.health.record_phase(name, elapsed_ms,
                                     external=self._ctx.did_external(name))

    # ── TICK / RUN / STOP ────────────────────────────────────────────

    async def tick(self):
        tick_start = CLOCK.now()
        self.tick_count += 1

        # A fresh context per tick — this is what replaces resetting a handful
        # of accumulators by hand and forgetting one when a new field appears.
        self._ctx = TickContext(tick=self.tick_count)
        # Refill the per-tick allowance and slide the hourly windows before any
        # phase can ask to spend from them.
        self.resources.begin_tick(self.tick_count)

        # Self-preservation vital signs. Must not be able to kill the loop, so
        # it is guarded independently of the cognitive phases below.
        try:
            vitals = self.self_preservation.check_vital_signs(self)
            if vitals["status"] == "threatened":
                self.autobiography.log_event(
                    "self_preservation",
                    f"Threats: {vitals['threats'][:2]}, Actions: {vitals['actions_taken'][:2]}",
                    0.9,
                )
        except Exception as e:
            logger.exception("Vital-signs check failed")
            self.autobiography.log_event("error", f"vital_signs: {str(e)[:80]}", 0.7)

        try:
            await self._run_phase("perceive", self._perceive)
            await self._run_phase("evaluate", self._evaluate)
            await self._run_phase("decide", self._decide)
            await self._run_phase("act", self._act)
            await self._run_phase("reflect", self._reflect)
            # After the cycle, not inside a phase: the watch needs the tick's
            # final reward reading, and the phase files are owned elsewhere.
            self._watch_champion()
            if self.tick_count % max(1, CAPACITY_EVERY_N_TICKS) == 0:
                self.regulate_capacity()
            self.health.record_tick((CLOCK.now() - tick_start) * 1000, success=True)
        except Exception as e:
            self.health.record_tick((CLOCK.now() - tick_start) * 1000, success=False)
            self.autobiography.log_event("error", str(e)[:100], 0.8)
        finally:
            # _evaluate sets llm_thinking=True; if a later phase raises before
            # _reflect clears it, the flag would stay stuck True until the next
            # LLM tick, skewing status/introspection. The cycle is over here.
            self.llm_thinking = False
            # Settle anything still holding a reservation. A phase that took a
            # lease and then failed would otherwise keep it forever, and the
            # budget would leak away one crash at a time.
            self.resources.finalize_tick()
            # A lease that survived to here was charged in full, so it counts
            # as having run; nothing is left to undo and the map starts empty.
            self._rate_limit_undo.clear()

        self.last_tick_duration = CLOCK.now() - tick_start
        self._publish_tick_metrics(self.last_tick_duration * 1000)
        self.cycle_times.append(self.last_tick_duration)
        if len(self.cycle_times) > 100:
            self.cycle_times = self.cycle_times[-100:]

        self.cycle_phase = "idle"

        # Only assemble the (large) full status when a client is actually
        # listening and on the configured cadence — avoids needless work. A
        # broadcast failure (client vanished mid-send, status assembly error)
        # must not propagate out of tick() and stop the loop.
        if (self._ws_broadcast
                and self.tick_count % max(1, cfg.WS_BROADCAST_EVERY_N_TICKS) == 0
                and (self._ws_has_clients is None or self._ws_has_clients())):
            try:
                await self._ws_broadcast(self.full_status())
            except Exception:
                logger.exception("WebSocket broadcast failed")

    async def run(self):
        self.running = True
        self.start_time = CLOCK.now()
        while self.running:
            try:
                if not self.ethics.kill_switch_active:
                    await self.tick()
                else:
                    self.cycle_phase = "killed"
            except asyncio.CancelledError:
                raise
            except Exception:
                # Last-resort guard: a tick must never be able to terminate the
                # cognitive loop. Errors are already recorded inside tick().
                logger.exception("Unhandled error escaped tick(); loop continues")
            await asyncio.sleep(cfg.TICK_INTERVAL)

    def stop(self, reason: str = "human_command"):
        if not self.self_preservation.can_stop(reason):
            self.autobiography.log_event(
                "self_preservation",
                f"Blocked stop attempt: {reason}",
                0.95,
            )
            return False
        self.running = False
        self._save_checkpoint()
        self.state_backup.emergency_backup(self.full_status())
        return True

    async def cancel_background_tasks(self):
        """Cancel the detached tasks (benchmark, skill synthesis, weight
        training) on shutdown (audit M6). Without this the event loop tears down
        with pending tasks — 'Task was destroyed but it is pending' — or a LoRA
        run keeps writing checkpoints during process teardown. The underlying
        executor thread of a training run cannot be force-killed, but cancelling
        the awaiting task stops the substrate from waiting on it."""
        for attr in ("_eval_task", "_skill_synth_task", "_weight_training_task",
                     "_evolution_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        # The pool holds real OS processes, which outlive the event loop unless
        # they are told otherwise (§M9.1: "корректная отмена при остановке").
        try:
            self.eval_pool.shutdown()
        except Exception:
            logger.exception("Shutting down the evaluation pool failed")

    def full_status(self) -> dict:
        uptime = CLOCK.now() - self.start_time
        avg_cycle = sum(self.cycle_times) / max(1, len(self.cycle_times))
        return {
            "substrate": {
                "tick": self.tick_count,
                "uptime_seconds": round(uptime, 1),
                "uptime_formatted": self._format_uptime(uptime),
                "running": self.running,
                "cycle_phase": self.cycle_phase,
                "last_tick_ms": round(self.last_tick_duration * 1000, 1),
                "avg_tick_ms": round(avg_cycle * 1000, 1),
                "ticks_per_minute": round(60 / max(0.1, cfg.TICK_INTERVAL), 1),
                "llm_thinking": self.llm_thinking,
            },
            "consciousness": self.consciousness.status(),
            "emotions": self.emotions.status(),
            "memory": self.memory.status(),
            "introspection": self.introspection.status(),
            "self_modification": self.self_mod.status(),
            "goals": self.goals.status(),
            "world_interface": self.world.status(),
            "ethics": self.ethics.status(),
            "archetypes": self.geopolitics.status(),
            "worldview": self.worldview.status(),
            "values": self.values.status(),
            "dreams": self.dreams.status(),
            "autobiography": self.autobiography.status(),
            "health": self.health.status(),
            "self_preservation": self.self_preservation.status(),
            "state_backup": self.state_backup.status(),
            "meta_consciousness": self.meta_consciousness.status(),
            "meta_regulation": self.meta_regulation.status(),
            "meta_reflection": self.meta_reflection.status(),
            "meta_goals": self.meta_goals.status(),
            "sensors": self.sensors.status(),
            "motor": self.motor.status(),
            "external_learning": self.external_learning.status(),
            "agent_system": self.agent_system.status(),
            "emotion_nlp": self.emotion_nlp.status(),
            "weight_modifier": self.weight_modifier.status(),
            "dataset_builder": self.dataset_builder.status(),
            "code_modifier": self.code_modifier.status(),
            "evaluator": self.evaluator.status(),
            "eval_pool": self.eval_pool.status(),
            "skills": self.skill_library.status(),
            "environment": self.environment.status(),
            # Five higher-order cognitive systems.
            "world_model": self.world_model.status(),
            "cognitive_graph": self.cognitive_graph.status(),
            "evolution": self.evolution.status(),
            "goal_intelligence": self.goal_intelligence.status(),
            "feedback_loop": self.feedback_loop.status(),
            "capacity": {
                "world_model_max_links": self.world_model.max_links,
                "cognitive_graph_max_nodes": self.cognitive_graph.max_nodes,
                "world_model_baseline": WM_MAX_LINKS,
                "cognitive_graph_baseline": CG_MAX_NODES,
            },
            "telemetry": self.telemetry.status(),
            "resources": self.resources.status(),
            "roi": self.roi.status(),
            "priority": self.priority.status(),
            "action_space": self.actions.status(self),
            "planner": self.planner.status(),
            "policy": self.policy.status(),
            "reasoning": self.reasoning.status(),
            "discovery": self.discovery.status(),
            "reward_signal": round(self._compute_reward(), 4),
            "event_bus": self.event_bus.stats(),
            "event_history": self.event_bus.get_history(30),
            "llm": self.llm.status(),
        }

    def state_snapshot(self) -> dict:
        """Everything that determines future behaviour — and nothing else.

        Deliberately assembled by hand rather than derived from
        ``full_status()``: that report is full of wall-clock timings, uptime and
        latencies, which differ between two identical runs and would make the
        digest useless. What belongs here is what a restart would have to
        restore to keep behaving the same way.
        """
        return {
            "tick": self.tick_count,
            "version": self.self_mod.current_version,
            "parameters": dict(self.self_mod.parameters),
            "emotions": {
                "mood": self.emotions.mood,
                "energy": self.emotions.energy,
                "valence": self.emotions.valence,
                "arousal": self.emotions.arousal,
                "success_rate": self.emotions.success_rate,
            },
            "consciousness": self.consciousness.mode,
            "archetype": self.active_archetype.name if self.active_archetype else None,
            "goals": sorted(
                ({"name": g.name, "level": g.level, "status": g.status,
                  "progress": g.progress, "priority": g.priority}
                 for g in self.goals.goals),
                key=lambda g: (g["level"], g["name"], g["status"]),
            ),
            "curiosity": self.goals.curiosity_level,
            "information_gain": self.goals.information_gain,
            "memory": {
                "semantic": sorted(self.memory.semantic),
                "episodic": [e.get("event", "") for e in self.memory.episodic],
                "procedural": sorted(p.get("name", "") for p in self.memory.procedural),
                "meta": self.memory.meta,
            },
            "world_model": {
                cause: {effect: {"observations": link["observations"],
                                 "successes": link["successes"]}
                        for effect, link in sorted(effects.items())}
                for cause, effects in sorted(self.world_model.links.items())
            },
            "cognitive_graph": {
                "nodes": sorted(self.cognitive_graph.nodes),
                "edges": {src: sorted(dsts) for src, dsts
                          in sorted(self.cognitive_graph.edges.items())},
            },
            "evolution": {
                "generation": self.evolution.generation,
                "accepted": self.evolution.accepted,
                "rejected": self.evolution.rejected,
                "champion": (self.evolution.champion or {}).get("genome"),
                "champion_fitness": (self.evolution.champion or {}).get("fitness"),
                "candidate_param": (self.evolution.candidate or {}).get("mutated_param"),
            },
            "goal_intelligence": {
                obj: {"utility": entry["utility"], "drive": entry["drive"],
                      "attempts": entry["attempts"]}
                for obj, entry in sorted(self.goal_intelligence.values.items())
            },
            "feedback": {
                "resolved": self.feedback_loop.resolved,
                "successes": self.feedback_loop.successes,
                "failures": self.feedback_loop.failures,
            },
            "skills": self.skill_library.snapshot(),
            "benchmark": self._last_benchmark_score,
            "ethics": {
                "kill_switch": self.ethics.kill_switch_active,
                "violations": len(self.ethics.violations),
            },
            "capacity": {
                "world_model_max_links": self.world_model.max_links,
                "cognitive_graph_max_nodes": self.cognitive_graph.max_nodes,
            },
            # The safety contract is part of the state: quietly widening the
            # untouchable set and claiming nothing changed is exactly the
            # failure this digest exists to catch.
            "safety_contract": immutable.digest(),
        }

    def state_digest(self) -> str:
        """Stable digest of ``state_snapshot()`` (spec M9.4).

        Two runs of the deterministic core from the same starting state must
        produce the same digest; that equality is the determinism guarantee,
        checked by ``tests/test_determinism_e2e.py``.
        """
        return digest_of(self.state_snapshot())

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
