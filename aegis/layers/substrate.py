"""Layer 0: Substrate — persistent runtime with PERCEIVE-EVALUATE-DECIDE-ACT-REFLECT cycle (S-001..S-006).

The core cognitive cycle (reward, confidence, importance, goal progress) is
deterministic and driven by real system metrics. Knowledge-acquisition helpers
(ExternalLearning, AgentSystem) still pick topics with `random` when none is
supplied, so the runtime as a whole is not fully deterministic.
Code self-modification is integrated via CodeModifier + LLM proposals.
"""
import asyncio
import json
import time
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.config import (
    CHECKPOINT_EVERY_N_TICKS, CHECKPOINTS_DIR,
    LLM_THINK_EVERY_N_TICKS, TRAIN_EVERY_N_TICKS,
    CODE_BACKUPS_DIR, CODE_MOD_EVERY_N_TICKS, CODE_MOD_MIN_TICK, CODE_MOD_MAX_PER_SESSION,
    CODE_MOD_MAX_FILE_CHARS, CODE_SELF_MOD_ENABLED,
    EVAL_DIR, EVAL_EVERY_N_TICKS, ENV_STEP_EVERY_N_TICKS, SKILL_SYNTH_EVERY_N_TICKS,
    SANDBOX_TIMEOUT,
    WORLD_MODEL_EVERY_N_TICKS, COGNITIVE_GRAPH_EVERY_N_TICKS, EVOLUTION_EVERY_N_TICKS,
)
from aegis.event_bus import EventBus, Event, Layer
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
from aegis.layers.world_model import WorldModel
from aegis.layers.cognitive_graph import CognitiveGraph
from aegis.layers.evolution_engine import EvolutionEngine
from aegis.layers.goal_intelligence import GoalIntelligence
from aegis.layers.feedback_loop import FeedbackLoop
from aegis.eval.benchmark import DEFAULT_BENCHMARK, tasks_for_kind, split_tasks
from aegis.eval.coding import CODING_BENCHMARK
from aegis.eval.composite import COMPOSITE_BENCHMARK
from aegis.eval.skill_library import SkillLibrary, Skill
from aegis.eval.solver import MultiAgentSolver
from aegis.eval.sandbox import run_skill
from aegis.eval.evaluator import Evaluator
from aegis.eval.environment import TaskEnvironment
from aegis.llm import LLMEngine

logger = logging.getLogger("aegis.substrate")

# External learning sources — cycled in order, not random
_LEARNING_SOURCES = ["wikipedia", "arxiv", "quotes"]

# Meta-knowledge domains — cycled in order
_META_DOMAINS = ["reasoning", "memory", "ethics", "planning", "creativity"]

# Concept seeds — cycled in order
_CONCEPT_SEEDS = ["pattern", "cycle", "adaptation", "learning", "stability"]


# ── LLM-output coercion (audit M5) ───────────────────────────────────
# Models routinely return "almost right" JSON — numbers as strings, an object
# where a scalar was asked for, a scalar where an object was asked for. Using
# those values raw (``parsed["chosen"] - 1``, ``knowledge.get(...)`` on a str)
# raises inside a cognitive phase and drops the rest of the tick. These helpers
# coerce defensively so a malformed field falls back instead of crashing.

def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        self._pending_experiences: dict = {}  # phase-open experience ids by kind

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
        if self.evolution.champion is None:
            base_fitness = self._last_benchmark_score if self._last_benchmark_score is not None else 0.0
            self.evolution.register_champion(dict(self.self_mod.parameters), base_fitness)

        self.event_bus.set_veto(self.ethics.veto_check)

        self.tick_count = 0
        self.start_time = time.time()
        self.running = False
        self.cycle_phase = "idle"
        self.cycle_times: list[float] = []
        self.last_tick_duration = 0.0
        self.llm_thinking = False
        self._regulation_directives: dict = {}
        self._ws_broadcast = None
        self._ws_has_clients = None  # callable -> bool, set by the API server
        self._weight_training_task: asyncio.Task | None = None  # detached LoRA run
        self._checkpoint_path = CHECKPOINTS_DIR / "latest.json"

        # Deterministic round-robin counters
        self._learning_source_idx = 0
        self._meta_domain_idx = 0
        self._concept_seed_idx = 0
        self._param_mod_idx = 0

        # Per-tick metric accumulators (reset each tick)
        self._tick_new_concepts = 0
        self._tick_new_episodic = 0
        self._tick_llm_insights = 0

        self._restore_checkpoint()

    def _restore_checkpoint(self):
        if self._checkpoint_path.exists():
            try:
                data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                self.tick_count = data.get("tick_count", 0)
                self.self_mod.current_version = data.get("version", "1.0.0")
            except Exception:
                logger.warning("Failed to restore checkpoint %s — starting fresh",
                               self._checkpoint_path, exc_info=True)

    def _save_checkpoint(self):
        data = {
            "tick_count": self.tick_count,
            "version": self.self_mod.current_version,
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "mood": self.emotions.mood,
            "consciousness_mode": self.consciousness.mode,
            "energy": self.emotions.energy,
        }
        # Atomic write — a crash mid-write must not corrupt the checkpoint and
        # silently reset tick_count/version to defaults on next boot.
        tmp = self._checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self._checkpoint_path)
        self.memory.save()

        # Persist the five higher-order systems alongside the checkpoint.
        for system in (self.world_model, self.cognitive_graph, self.evolution,
                       self.goal_intelligence, self.feedback_loop):
            try:
                system.save()
            except Exception:
                logger.exception("Failed to persist %s", type(system).__name__)

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

    # ── PERCEIVE ─────────────────────────────────────────────────────

    async def _perceive(self):
        self.cycle_phase = "perceive"
        perception = self.world.perceive()

        # Sensor cortex
        sensor_data = self.sensors.read_all()
        perception["sensors"] = sensor_data

        # Emotional perception — reward from REAL system metrics
        reward = self._compute_reward()
        context = {
            "tick": self.tick_count,
            "new_knowledge": self._tick_new_concepts > 0,
            "error": self.health.consecutive_errors > 0,
            "unexpected": self.health.consecutive_errors > 3,
            "repetitive": self.emotions.mood_duration > 15,
        }
        self.emotions.update(reward, context)

        # Update consciousness based on emotion
        self.consciousness.update_mode(self.emotions.mood, self.emotions.energy, self.emotions.arousal)

        # Archetype activation
        self._update_archetypes()

        self.memory.add_working({"phase": "perceive", "data": perception, "mood": self.emotions.mood})
        await self.event_bus.publish(Event(
            source=Layer.SUBSTRATE, target=Layer.MEMORY,
            event_type="perception", payload=perception
        ))

    def _update_archetypes(self):
        for arch in self.archetypes_list:
            if arch.should_activate(self.emotions.mood, self.emotions.energy):
                self.active_archetype = arch
                break
        else:
            if self.archetypes_list:
                self.active_archetype = self.archetypes_list[0]

        if self.active_archetype:
            action_desc = self.active_archetype.act(
                self.consciousness.mode,
                self.goals.get_current_focus().get("name", "idle") if self.goals.get_current_focus() else "idle"
            )
            self.active_archetype.log_experience(
                self.tick_count, self.emotions.mood,
                self.emotions.success_rate, action_desc
            )

        if self.tick_count % 10 == 0:
            self.geopolitics.update_influence()

    # ── EVALUATE ─────────────────────────────────────────────────────

    async def _evaluate(self):
        self.cycle_phase = "evaluate"

        # Real system metrics for introspection
        total_ticks = self.health.successful_ticks + self.health.failed_ticks
        error_rate = self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
        active_goals = [g for g in self.goals.goals if g.status == "active" and g.level != "axiom"]
        mem_status = self.memory.status()

        system_metrics = {
            "memory_load": mem_status["working_memory_size"] / max(1, mem_status["working_memory_max"]),
            "goal_pressure": len(active_goals) / 10.0,
            "ethics_load": self.ethics.total_checked / max(1, self.tick_count),
            "energy": self.emotions.energy,
            "information_gain": self.goals.information_gain,
            "error_rate": error_rate,
            "llm_active": self.llm_thinking,
            "tick": self.tick_count,
        }
        activations = self.introspection.inspect_activations("main", system_metrics)

        # Goal progress with real metrics
        goal_metrics = {
            "new_concepts": self._tick_new_concepts,
            "new_episodic": self._tick_new_episodic,
            "error_rate": error_rate,
            "energy": self.emotions.energy,
            "llm_insights": self._tick_llm_insights,
        }
        self.goals.evaluate_progress(goal_metrics)
        focus = self.goals.get_current_focus()

        bias_report = None
        if self.tick_count % 20 == 0 and self.introspection.decision_trace:
            bias_report = self.introspection.detect_bias(self.introspection.decision_trace[-20:])

        # Value system evaluation
        focus_name = focus["name"] if focus else "idle"
        self.values.evaluate_action(self.emotions.mood, self.emotions.success_rate, focus_name)

        # Health check
        health_report = self.health.check()
        if health_report["status"] == "critical":
            self.autobiography.log_event("health", f"Critical health: {health_report['critical']}", 0.9)

        # Meta-regulation
        reg = self.meta_regulation.regulate(
            self.emotions.energy, health_report["status"],
            self.health.consecutive_errors, self.consciousness.mode,
        )
        self._regulation_directives = reg["directives"]
        if reg["directives"]["force_recharge"] > 0:
            self.emotions.recharge(reg["directives"]["force_recharge"])
        if reg["directives"]["reduce_sensors"]:
            self.sensors.reduce_sensors(True)
        else:
            self.sensors.reduce_sensors(False)

        # Meta-consciousness (every 25 ticks)
        if self.tick_count % 25 == 0:
            mc = self.meta_consciousness.evaluate(
                self.consciousness.mode,
                self.active_archetype.name if self.active_archetype else None,
                self.emotions.mood, self.emotions.energy,
                focus_name, self.archetypes_list,
            )
            if mc["fragmentation"] > 0.5:
                self.autobiography.log_event("meta", f"High fragmentation: {mc['fragmentation']:.2f}", 0.7)

        # LLM-powered state evaluation
        llm_eval = None
        if self._is_llm_tick() and not self._regulation_directives.get("skip_llm"):
            self.llm_thinking = True
            # RAG: pull concepts RELEVANT to the current focus, not just recent ones.
            focus_query = (focus.get("name", "") + " " + focus.get("description", "")) if focus else ""
            relevant = self.memory.retrieve(focus_query, k=6) if focus_query.strip() else []
            relevant_concepts = [r["concept"] for r in relevant] or list(self.memory.semantic.keys())[-10:]
            compact_state = {
                "tick": self.tick_count,
                "goals_active": len(active_goals),
                "current_focus": focus,
                "memory_total": mem_status["total_memories"],
                "episodic_recent": [e["event"] for e in self.memory.episodic[-3:]],
                "relevant_concepts": relevant_concepts,
                "semantic_concepts": list(self.memory.semantic.keys())[-10:],
                "version": self.self_mod.current_version,
                "curiosity": round(self.goals.curiosity_level, 3),
                "information_gain": round(self.goals.information_gain, 3),
                "mood": self.emotions.mood,
                "energy": round(self.emotions.energy, 3),
                "consciousness_mode": self.consciousness.mode,
                "active_archetype": self.active_archetype.name if self.active_archetype else None,
            }
            result = await self.llm.evaluate_state(compact_state)
            if result.get("response"):
                _, llm_warnings = self.self_preservation.filter_llm_response(result["response"])
                if llm_warnings:
                    self.autobiography.log_event("llm_danger", str(llm_warnings[:2]), 0.9)
            if result["success"] and "parsed" in result:
                llm_eval = result["parsed"]
                insight = llm_eval.get("insight", "")
                if insight:
                    self.memory.add_episodic(
                        f"LLM Insight: {insight}",
                        emotional_valence=0.3, importance=0.8
                    )
                    self.autobiography.log_event("insight", insight[:100], 0.7)
                    self._tick_llm_insights += 1
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
                    priority = 0.5 + 0.1 * self.emotions.energy
                    g = Goal(
                        name=sg[:30].replace(" ", "_").lower(),
                        level="tactic",
                        description=sg,
                        priority=priority,
                    )
                    g.reasoning = "Generated by LLM evaluation"
                    self.goals.goals.append(g)

                await self.event_bus.publish(Event(
                    source=Layer.INTROSPECTION, target=None,
                    event_type="llm_evaluation",
                    payload={"assessment": llm_eval.get("assessment", "")[:100]}
                ))

        self.memory.add_working({
            "phase": "evaluate",
            "activations": activations,
            "focus": focus,
            "bias_report": bias_report,
            "llm_eval": llm_eval,
        })

    # ── DECIDE ───────────────────────────────────────────────────────

    async def _decide(self):
        self.cycle_phase = "decide"
        new_goals = self.goals.generate_goals({
            "tick": self.tick_count,
            "memory_size": self.memory.status()["total_memories"],
        })
        focus = self.goals.get_current_focus()
        decision = focus["name"] if focus else "idle_exploration"
        alternatives = ["optimize_memory", "self_inspect", "explore_topic", "rest"]

        # Archetype-influenced decision
        if self.active_archetype and self.consciousness.mode in self.active_archetype.strategies:
            alternatives.append(f"archetype_{self.active_archetype.name}")

        # Real confidence from system state
        confidence = self._compute_confidence()
        reasoning = f"Selected based on priority and progress. Tick #{self.tick_count}"

        # Meta-goal generation (every 30 ticks)
        if self.tick_count % 30 == 0:
            total_ticks = self.health.successful_ticks + self.health.failed_ticks
            meta_ctx = {
                "memory_total": self.memory.status()["total_memories"],
                "mood_valence": self.emotions.valence,
                "learning_sessions": self.external_learning.learning_sessions,
                "avg_tick_ms": self.last_tick_duration * 1000,
                "tick": self.tick_count,
                "error_rate": self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0,
                "active_agents": sum(1 for a in self.agent_system.agents if a.status in ("active", "deployed")),
            }
            new_meta = self.meta_goals.generate_goals(meta_ctx)
            for mg in new_meta:
                self.autobiography.log_event("meta_goal", mg["description"][:60], 0.5)

        # LLM-powered decision making
        if self._is_llm_tick() and not self._regulation_directives.get("skip_llm"):
            options = [decision] + alternatives
            result = await self.llm.make_decision(options, {
                "focus": focus,
                "tick": self.tick_count,
                "goals_summary": {g.name: g.progress for g in self.goals.goals
                                  if g.status == "active" and g.level != "axiom"},
                "mood": self.emotions.mood,
                "consciousness_mode": self.consciousness.mode,
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

                await self.event_bus.publish(Event(
                    source=Layer.GOAL_ENGINE, target=None,
                    event_type="llm_decision",
                    payload={"decision": decision, "reasoning": reasoning[:100]}
                ))

        trace = self.introspection.trace_decision(
            decision, alternatives, reasoning, confidence
        )

        eth_result = self.ethics.evaluate_action({
            "type": decision,
            "confidence": confidence,
            "modifies_self": False,
        })

        self.memory.add_working({
            "phase": "decide",
            "decision": decision,
            "reasoning": reasoning,
            "ethical_score": eth_result["score"],
            "ethical_status": eth_result["status"],
        })

        await self.event_bus.publish(Event(
            source=Layer.GOAL_ENGINE, target=Layer.ETHICS_CORE,
            event_type="decision", payload={"decision": decision, "ethics": eth_result}
        ))

        # ── Higher systems (4 & 5 & 1): value-driven selection, experience
        #    opening, and a causal chain for the objective. All deterministic
        #    and cheap; guarded so a failure here cannot abort the tick. ──
        try:
            focus_name = focus["name"] if focus else "idle_exploration"
            _tt = self.health.successful_ticks + self.health.failed_ticks
            error_rate = self.health.error_count / max(_tt, 1) if _tt > 0 else 0.0
            options = [decision] + alternatives
            gi_ctx = {
                "tick": self.tick_count,
                "energy": self.emotions.energy,
                "error_rate": error_rate,
                "curiosity": self.goals.curiosity_level,
            }
            gi_choice = self.goal_intelligence.choose(options, gi_ctx)

            # System 5: open an experience for this decision — it will be closed
            # in REFLECT with the tick's realized reward and inferred cause.
            self._pending_experiences["decide"] = self.feedback_loop.record_situation(
                situation=f"focus={focus_name} mode={self.consciousness.mode} "
                          f"energy={self.emotions.energy:.2f} err={error_rate:.2f}",
                decision=decision,
                context={"tick": self.tick_count,
                         "value": gi_choice["expected_value"] if gi_choice else None},
            )

            # System 1: build a causal chain (objective -> constraints -> risks
            # -> plan -> expected result) for the current focus.
            if self.tick_count % max(1, WORLD_MODEL_EVERY_N_TICKS) == 0 and focus:
                constraints = [g.name for g in self.goals.goals if g.level == "axiom"]
                chain = self.world_model.build_chain(focus_name, constraints)
                if chain["plan"]:
                    self.autobiography.log_event(
                        "world_model",
                        f"Chain for {focus_name}: {len(chain['plan'])} steps, "
                        f"conf={chain['confidence']}",
                        0.4,
                    )
        except Exception:
            logger.exception("Higher-systems DECIDE hook failed")

    # ── ACT ──────────────────────────────────────────────────────────

    async def _act(self):
        self.cycle_phase = "act"
        action_result = self.world.act({"type": "internal_computation"})

        # Motor cortex
        if self.tick_count % 10 == 0:
            focus = self.goals.get_current_focus()
            self.motor.execute(
                "log",
                payload={"message": f"Tick {self.tick_count}: focus={focus.get('name', 'none') if focus else 'none'}"},
                archetype=self.active_archetype.name if self.active_archetype else None,
                goal=focus.get("name") if focus else None,
            )

        # External learning (every 40 ticks) — round-robin sources and topics
        if self.tick_count % 40 == 0 and not self._regulation_directives.get("skip_learning"):
            source = _LEARNING_SOURCES[self._learning_source_idx % len(_LEARNING_SOURCES)]
            self._learning_source_idx += 1

            # Pick topic from most recent semantic concepts, or default
            topics = list(self.memory.semantic.keys())[-10:]
            topic = topics[self.tick_count % max(1, len(topics))] if topics else "artificial intelligence"

            learn_result = await self.external_learning.learn_from_source(source, topic)
            if learn_result.get("success"):
                for concept in learn_result.get("concepts", [])[:3]:
                    self.memory.add_semantic(concept[:50], {
                        "type": "external_learning",
                        "source": source,
                        "confidence": 0.6,
                    })
                    self._tick_new_concepts += 1
                self.autobiography.log_event("learning", f"Learned from {source}: {topic[:40]}", 0.5)
                self.goals.advance_progress("expand_knowledge", 0.02 * len(learn_result.get("concepts", [])))
                self.motor.execute("log", payload={
                    "message": f"Learning: {len(learn_result.get('concepts', []))} concepts from {source} ({topic[:30]})"
                })

        # Agent system
        if not self._regulation_directives.get("skip_learning"):
            agent_results = await self.agent_system.run_due_agents()
            for ar in agent_results:
                self.autobiography.log_event(
                    "agent_fetch",
                    f"{ar['agent']} [{ar['source']}]: {ar['items']} items",
                    0.4,
                )
                self.motor.execute("log", payload={
                    "message": f"Agent {ar['agent']}: fetched {ar['items']} items from {ar['source']}"
                })
            new_knowledge = self.agent_system.get_recent_knowledge(5)
            for kn in new_knowledge:
                item = kn.get("data", {})
                title = item.get("title", "")[:50]
                summary = item.get("summary", "")[:100]
                if title and title not in self.memory.semantic:
                    self.memory.add_semantic(title, {
                        "type": f"agent_{kn.get('source', 'unknown')}",
                        "summary": summary,
                        "agent": kn.get("agent", ""),
                        "confidence": 0.55,
                    })
                    self._tick_new_concepts += 1

        # Agent evolution (every 100 ticks)
        if self.tick_count % 100 == 0:
            evo = self.agent_system.evolve()
            if evo["retired"]:
                self.autobiography.log_event("agents", f"Retired {len(evo['retired'])} agents", 0.4)
                self.motor.execute("alert", payload={"level": "warning", "message": f"Evolution: retired {len(evo['retired'])} agents"})
            if evo.get("created"):
                self.autobiography.log_event("agents", f"Created {len(evo['created'])} replacement agents", 0.5)
                self.motor.execute("log", payload={"message": f"Evolution: spawned {len(evo['created'])} new agents"})

        # LLM-driven curiosity exploration (every 5th LLM tick instead of random 40%)
        if self._is_llm_tick() and not self._regulation_directives.get("skip_llm") and self.tick_count % (LLM_THINK_EVERY_N_TICKS * 5) == 0:
            known = list(self.memory.semantic.keys())
            result = await self.llm.generate_curiosity(known)
            if result["success"] and "parsed" in result:
                parsed = result["parsed"]
                topic = parsed.get("topic", "")
                question = parsed.get("question", "")
                if topic:
                    self.memory.add_semantic(topic[:50], {
                        "type": "curiosity_exploration",
                        "question": question,
                        "connection": parsed.get("connection", ""),
                        "confidence": 0.6,
                    })
                    self.memory.add_episodic(
                        f"Explored topic: {topic}. Question: {question}",
                        emotional_valence=0.4, importance=0.7
                    )
                    self._tick_new_concepts += 1
                    self.goals.information_gain += 0.3
                    self.goals.advance_progress("expand_knowledge", 0.05)
                    self.autobiography.log_event("curiosity", f"Explored: {topic[:60]}", 0.5)

                    await self.event_bus.publish(Event(
                        source=Layer.GOAL_ENGINE, target=None,
                        event_type="llm_curiosity",
                        payload={"topic": topic[:80]}
                    ))

        # ── Parametric self-modification (LLM-driven instead of random) ──
        if self.tick_count % 15 == 0 and self._is_llm_tick() and not self._regulation_directives.get("skip_llm"):
            await self._llm_parametric_modification()

        # ── Code self-modification (opt-in only — audit C2) ──
        if (CODE_SELF_MOD_ENABLED
                and self.tick_count % CODE_MOD_EVERY_N_TICKS == 0
                and self.tick_count >= CODE_MOD_MIN_TICK
                and self._code_mod_count_session < CODE_MOD_MAX_PER_SESSION
                and self._is_llm_tick()
                and not self._regulation_directives.get("skip_llm")
                and not self._regulation_directives.get("skip_learning")):
            await self._code_self_modification()

        # Weight modification — LoRA fine-tuning every N ticks.
        # Spawned as a DETACHED background task: a full training run takes
        # minutes+, and awaiting it here would suspend the whole PERCEIVE..
        # REFLECT cycle (no ticks, no dashboard/WS updates) until it finished.
        # The cognitive loop keeps running while training proceeds in an
        # executor thread; the training_in_progress flag + task handle prevent
        # overlapping runs.
        training_busy = (self.weight_modifier.training_in_progress
                         or (self._weight_training_task is not None
                             and not self._weight_training_task.done()))
        if (self.tick_count % TRAIN_EVERY_N_TICKS == 0
                and self.tick_count > 0
                and not self._regulation_directives.get("skip_learning")
                and not training_busy):
            eth_weight = self.ethics.evaluate_weight_modification({
                "dataset_size": len(self.memory.semantic),
                "energy": self.emotions.energy,
                "health_status": self.health.check().get("status", "ok"),
                "consecutive_failures": self.weight_modifier.total_rollbacks,
            })
            if eth_weight["status"] != "blocked":
                self.autobiography.log_event(
                    "weight_training", "Starting LoRA fine-tuning cycle (background)", 0.8
                )
                self.motor.execute("alert", payload={
                    "level": "info", "message": "Weight training: started (background)",
                })
                self._weight_training_task = asyncio.create_task(self._weight_training_cycle())

        # ── Grounding: act in the task environment for REAL reward (point 5) ──
        if self.tick_count % max(1, ENV_STEP_EVERY_N_TICKS) == 0:
            step = await asyncio.get_running_loop().run_in_executor(None, self.environment.step)
            if step.get("task"):
                self.goals.advance_progress("expand_knowledge", 0.01 if step["solved"] else 0.0)
                # System 1: the environment is real cause->effect data. Record
                # "attempting kind K" -> "solved/failed" so the World Model
                # learns which task kinds the current skills actually handle.
                try:
                    self.world_model.observe(
                        f"attempt:{step['kind']}",
                        "solved" if step["solved"] else "failed",
                        success=bool(step["solved"]),
                    )
                    if step["solved"] and step.get("winning_skill"):
                        self.world_model.observe(
                            f"skill:{step['winning_skill']}", f"solves:{step['kind']}", success=True)
                except Exception:
                    logger.exception("World-model observe failed")
                if step["solved"]:
                    self.autobiography.log_event(
                        "env_solved", f"{step['task']} via {step['winning_skill']}", 0.4)
                else:
                    self.autobiography.log_event(
                        "env_failed", f"{step['task']} ({step['kind']}) — no skill solved it", 0.5)

        # ── System 3: Evolution Engine — propose a parameter mutation (version
        #    B). Applied through the SAME safety pipeline as any self-mod, then
        #    judged when the next benchmark lands (see _run_benchmark). ──
        if (self.tick_count % max(1, EVOLUTION_EVERY_N_TICKS) == 0
                and self.tick_count > 0
                and self.evolution.candidate is None
                and not self._regulation_directives.get("skip_learning")):
            await self._evolution_step()

        # ── Periodic held-out benchmark (the fitness graph, point 2) ──
        if (self.tick_count % max(1, EVAL_EVERY_N_TICKS) == 0
                and (self._eval_task is None or self._eval_task.done())):
            self._eval_task = asyncio.create_task(self._run_benchmark())

        # ── Skill synthesis: close a failing kind, learn a coding solution, or
        #    simplify an already-solved kind (points 3, 4 + coding + versioning) ──
        if (self.tick_count % max(1, SKILL_SYNTH_EVERY_N_TICKS) == 0
                and self.tick_count > 0
                and self.llm.enabled
                and not self._regulation_directives.get("skip_learning")
                and (self._skill_synth_task is None or self._skill_synth_task.done())):
            self._skill_synth_task = asyncio.create_task(self._learning_cycle())

        self.memory.add_working({"phase": "act", "result": action_result})

    async def _run_benchmark(self):
        """Run the held-out benchmark off the tick loop and update the reward signal."""
        try:
            report = await asyncio.get_running_loop().run_in_executor(None, self.evaluator.run)
            self._last_benchmark_score = report["score"]
            self.autobiography.log_event(
                "benchmark", f"score={report['score']:.3f} ({report['passed']}/{report['total']})", 0.7)

            # ── System 3: natural selection. If a mutated genome is pending,
            #    the fresh benchmark is its fitness. Keep it only if it beat the
            #    champion; otherwise roll the parameter back to its old value. ──
            if self.evolution.candidate is not None:
                verdict = self.evolution.judge_candidate(report["score"])
                if verdict.get("decision") == "rejected" and verdict.get("revert_to") is not None:
                    param = verdict["param"]
                    if param in self.self_mod.parameters:
                        self.self_mod.parameters[param] = verdict["revert_to"]
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
            if param not in self.self_mod.parameters:
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

            # Clamp to the parameter's real bounds, then apply directly. The
            # champion/candidate genome and old_value let us revert on rejection.
            bounds = getattr(self.self_mod, "_param_bounds", {}).get(param)
            if bounds:
                lo, hi = bounds
                new_val = max(lo, min(hi, new_val))
                self.evolution.candidate["new_value"] = new_val
                self.evolution.candidate["genome"][param] = new_val
            self.self_mod.parameters[param] = new_val
            self.autobiography.log_event(
                "evolution",
                f"Proposed mutation: {param} {mutation['old_value']:.5f} -> {new_val:.5f} "
                f"(gen {self.evolution.generation}, awaiting benchmark)",
                0.6)
        except Exception:
            logger.exception("Evolution step failed")
            self.evolution.abandon_candidate()

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
            for attempt in range(2):  # initial + one repair
                if not code:
                    break
                self._skill_synth_count += 1
                name = f"{kind}_llm_{self._skill_synth_count}"
                skill = Skill(name=name, kinds=[kind], code=code, origin="llm")
                added, msg = self.skill_library.add(skill)
                if not added:
                    self.autobiography.log_event("skill_rejected", f"{kind}: {msg[:60]}", 0.5)
                    break

                after = await asyncio.get_running_loop().run_in_executor(
                    None, self.evaluator.pass_rate_on, holdout)
                if after > before:
                    self.skill_library.save()
                    self.autobiography.log_event(
                        "skill_learned", f"{kind}: holdout {before:.2f} -> {after:.2f}", 0.9)
                    self.motor.execute("alert", payload={
                        "level": "info",
                        "message": f"Learned generalizing skill for '{kind}' ({before:.2f}->{after:.2f})"})
                    kept = True
                    break

                # No generalization — discard and try one repair from a failing case.
                self.skill_library.remove(name)
                if attempt == 0 and train_examples:
                    code = await self.llm.propose_skill(kind, train_examples)
                else:
                    code = None

            if not kept:
                self.autobiography.log_event(
                    "skill_discarded", f"{kind}: no generalizing improvement ({before:.2f})", 0.5)
        except Exception:
            logger.exception("Skill synthesis failed")

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
            wmod_result = await self.self_mod.propose_weight_modification(
                self.memory, self.agent_system, self.ethics
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

    async def _llm_parametric_modification(self):
        """Use LLM to analyze performance and propose parameter adjustments."""
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

        result = await self.llm.analyze_self_performance(metrics)
        if not result.get("success") or not result.get("parsed"):
            return

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
                self.autobiography.log_event(
                    "self_mod",
                    f"Parameter {param}: {old_val:.6f} -> {new_val:.6f} (LLM: {adj.get('reason', '')[:50]})",
                    0.7,
                )

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

    # ── REFLECT ──────────────────────────────────────────────────────

    async def _reflect(self):
        self.cycle_phase = "reflect"

        # Meta-reflection (every 20 ticks)
        if self.tick_count % 20 == 0:
            total_goals = len(self.goals.goals)
            completed = sum(1 for g in self.goals.goals if g.status == "completed")
            total_ticks = self.health.successful_ticks + self.health.failed_ticks
            error_rate = self.health.error_count / max(total_ticks, 1) if total_ticks > 0 else 0
            recent = [e["event"] for e in self.memory.episodic[-8:]]
            mr = self.meta_reflection.reflect(
                self.tick_count, self.emotions.energy, self.emotions.mood,
                self.emotions.valence, error_rate, completed, total_goals,
                self.consciousness.mode, recent,
            )
            for insight in mr.get("insights", [])[:2]:
                self.memory.add_episodic(f"MetaInsight: {insight[:80]}", emotional_valence=0.2, importance=0.75)

        # LLM-powered reflection
        llm_reflection = None
        if self._is_llm_tick() and not self._regulation_directives.get("skip_llm"):
            recent_events = [e["event"] for e in self.memory.episodic[-5:]]
            episode = {
                "tick": self.tick_count,
                "recent_events": recent_events,
                "goals_completed": sum(1 for g in self.goals.goals if g.status == "completed"),
                "information_gain": round(self.goals.information_gain, 3),
                "version": self.self_mod.current_version,
                "mood": self.emotions.mood,
                "consciousness_mode": self.consciousness.mode,
            }
            result = await self.llm.reflect(episode)
            if result["success"] and "parsed" in result and isinstance(result["parsed"], dict):
                llm_reflection = result["parsed"]
                learning = str(llm_reflection.get("learning", "") or "")
                if learning:
                    self.memory.add_episodic(
                        f"Reflection: {learning}",
                        emotional_valence=0.2, importance=0.85
                    )
                    self.autobiography.log_event("reflection", learning[:100], 0.6)
                    self._tick_llm_insights += 1
                # "knowledge" may be a string or missing instead of an object
                # (audit M5) — guard before calling .get on it.
                knowledge = llm_reflection.get("knowledge", {})
                if isinstance(knowledge, dict) and isinstance(knowledge.get("concept"), str) \
                        and knowledge["concept"].strip():
                    self.memory.add_semantic(knowledge["concept"], {
                        "definition": str(knowledge.get("definition", "")),
                        "type": "learned_concept",
                        "source": "self_reflection",
                        "confidence": 0.75,
                    })
                    self.memory.update_meta(
                        knowledge["concept"], True, 0.75
                    )
                    self._tick_new_concepts += 1

                await self.event_bus.publish(Event(
                    source=Layer.INTROSPECTION, target=None,
                    event_type="llm_reflection",
                    payload={"learning": learning[:100]}
                ))

            self.llm_thinking = False

        # Dream generation (every 50 ticks when energy is low or reflective)
        if self.tick_count % 50 == 0 and not self._regulation_directives.get("skip_dreams") and \
                (self.emotions.energy < 0.4 or self.consciousness.mode == "reflective"):
            recent = [e["event"] for e in self.memory.episodic[-10:]]
            concepts = list(self.memory.semantic.keys())[-15:]
            dream = self.dreams.generate_dream(self.emotions.mood, recent, concepts)
            self.autobiography.log_event("dream", dream["narrative"][:80], 0.4)
            self.memory.add_episodic(f"Dream: {dream['narrative'][:80]}", emotional_valence=0.1, importance=0.5)

        # Energy recharge on rest ticks
        if self.tick_count % 20 == 0:
            self.emotions.recharge(0.05)

        # Event summary with computed importance
        event_summary = f"Tick {self.tick_count}: cycle completed"
        if llm_reflection:
            # Coerce — "learning" may be a non-string (audit M5).
            _learned = str(llm_reflection.get("learning", "") or "")
            event_summary += f" | Learned: {_learned[:60]}"
        importance = self._compute_importance()
        # Valence from actual emotional state
        valence = self.emotions.valence - 0.5  # center around 0
        self.memory.add_episodic(event_summary, emotional_valence=valence, importance=importance)
        self._tick_new_episodic += 1

        # Meta-knowledge update — round-robin through domains
        if self.tick_count % 10 == 0:
            domain = _META_DOMAINS[self._meta_domain_idx % len(_META_DOMAINS)]
            self._meta_domain_idx += 1
            # Confidence based on actual success in that domain
            domain_confidence = 0.5 + 0.3 * self.emotions.success_rate + 0.2 * self.emotions.energy
            self.memory.update_meta(domain, True, min(0.95, domain_confidence))

        # Concept seeding — round-robin
        if self.tick_count % 25 == 0:
            concept = _CONCEPT_SEEDS[self._concept_seed_idx % len(_CONCEPT_SEEDS)]
            self._concept_seed_idx += 1
            self.memory.add_semantic(f"{concept}_{self.tick_count}", {
                "type": concept, "tick": self.tick_count,
                "confidence": 0.5 + 0.3 * self.emotions.success_rate,
            })

        if self.tick_count % 30 == 0:
            self.memory.apply_forgetting()

        # ── Higher systems (5, 4, 2): close the experience loop, credit the
        #    realized reward to the chosen objective's value, and grow the
        #    cognitive graph from this tick's memory. Guarded — never aborts. ──
        try:
            realized = self._compute_reward()
            exp_id = self._pending_experiences.pop("decide", None)
            if exp_id is not None:
                # Success = this tick produced knowledge/insight and stayed healthy.
                success = (self._tick_new_concepts > 0 or self._tick_llm_insights > 0) \
                    and self.health.consecutive_errors == 0
                experience = self.feedback_loop.record_result(
                    exp_id, success=success, metric=realized,
                    expected="knowledge gain / healthy tick",
                )
                # System 4: credit realized reward to the chosen objective.
                self.goal_intelligence.reward(realized)
                # System 1: the decision's outcome is causal data too.
                if experience is not None:
                    self.world_model.observe(
                        f"decision:{experience['decision'][:40]}",
                        "productive" if success else "unproductive",
                        success=success,
                    )

            # System 2: ingest recent memory into the typed cognitive graph.
            if self.tick_count % max(1, COGNITIVE_GRAPH_EVERY_N_TICKS) == 0:
                self.cognitive_graph.ingest_memory(self.memory)
        except Exception:
            logger.exception("Higher-systems REFLECT hook failed")

        if self.tick_count % CHECKPOINT_EVERY_N_TICKS == 0:
            self._save_checkpoint()
            if self.tick_count % (CHECKPOINT_EVERY_N_TICKS * 5) == 0:
                self.state_backup.save_state(self.full_status(), "scheduled")

        await self.event_bus.publish(Event(
            source=Layer.SUBSTRATE, target=None,
            event_type="tick_complete",
            payload={"tick": self.tick_count}
        ))

    # ── TICK / RUN / STOP ────────────────────────────────────────────

    async def tick(self):
        tick_start = time.time()
        self.tick_count += 1

        # Reset per-tick accumulators
        self._tick_new_concepts = 0
        self._tick_new_episodic = 0
        self._tick_llm_insights = 0

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
            await self._perceive()
            await self._evaluate()
            await self._decide()
            await self._act()
            await self._reflect()
            self.health.record_tick((time.time() - tick_start) * 1000, success=True)
        except Exception as e:
            self.health.record_tick((time.time() - tick_start) * 1000, success=False)
            self.autobiography.log_event("error", str(e)[:100], 0.8)
        finally:
            # _evaluate sets llm_thinking=True; if a later phase raises before
            # _reflect clears it, the flag would stay stuck True until the next
            # LLM tick, skewing status/introspection. The cycle is over here.
            self.llm_thinking = False

        self.last_tick_duration = time.time() - tick_start
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
        self.start_time = time.time()
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
        for attr in ("_eval_task", "_skill_synth_task", "_weight_training_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    def full_status(self) -> dict:
        uptime = time.time() - self.start_time
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
            "skills": self.skill_library.status(),
            "environment": self.environment.status(),
            # Five higher-order cognitive systems.
            "world_model": self.world_model.status(),
            "cognitive_graph": self.cognitive_graph.status(),
            "evolution": self.evolution.status(),
            "goal_intelligence": self.goal_intelligence.status(),
            "feedback_loop": self.feedback_loop.status(),
            "reward_signal": round(self._compute_reward(), 4),
            "event_bus": self.event_bus.stats(),
            "event_history": self.event_bus.get_history(30),
            "llm": self.llm.status(),
        }

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
