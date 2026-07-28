"""Scoring a genome without letting it touch the running system (spec M5.5).

A variant is evaluated by building a **fresh** system from the genome in another
process, running it against a fixed scenario, and reading the result. Nothing
live is reachable from inside, which is what makes ten variants per generation
safe: the alternative — apply, measure, revert — trains the live skill library
with every measurement and tunes the very counters the solver ranks by.

The fitness has three terms (§M5.5):

    f = score_valid − κ_cost·cost_norm − κ_lat·latency_norm

The penalties are not decoration. Without them evolution reliably discovers that
a longer sandbox timeout and a wider beam score better, and converges on a
configuration that is correct and unaffordable.

``evaluate_variant`` is module-level and takes plain data, because that is the
only kind of thing a process pool can carry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.eval.benchmark import DEFAULT_BENCHMARK, three_way_split
from aegis.eval.generators import generated_benchmark
from aegis.eval.isolated import make_request, run_request
from aegis.eval.skill_library import SkillLibrary
from aegis.layers.evolution.genome import Genome

logger = logging.getLogger("aegis.evolution")

#: How much of the benchmark a variant is scored on. Big enough that a one-task
#: difference does not move the score by a visible amount, small enough that ten
#: variants fit in a generation.
DEFAULT_PER_KIND = 4

#: How many ticks of a real cognitive cycle a variant is judged on.
#:
#: The skill benchmark alone is nearly blind to this genome: only
#: ``solver_timeout`` and ``solver_order`` touch it, and the seeded skills are
#: fast and correct, so a whole population scores within two percent of itself.
#: Selecting on that is the exact failure this stage exists to correct — an
#: evolution searching a space with no gradient in it.
#:
#: So half the fitness comes from actually *running* the system with the genome
#: applied. Planner weights, exploration pressure, world-model smoothing,
#: priority weights and the resource split all change what happens in a tick,
#: and this is where that becomes visible.
ROLLOUT_TICKS = 60

#: How the two halves are weighed. Equal: capability that is never deployed and
#: behaviour with nothing to deploy are both worthless.
SKILL_SHARE = 0.5
ROLLOUT_SHARE = 0.5


@dataclass
class FitnessReport:
    """What came back from scoring one genome."""

    genome_id: str
    score_valid: float = 0.0
    score_test: float | None = None
    fitness: float = 0.0
    subscores: dict = field(default_factory=dict)
    cost_norm: float = 0.0
    latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "score_valid": round(self.score_valid, 6),
            "score_test": (round(self.score_test, 6)
                           if self.score_test is not None else None),
            "fitness": round(self.fitness, 6),
            "subscores": dict(self.subscores),
            "cost_norm": round(self.cost_norm, 6),
            "latency_ms": round(self.latency_ms, 3),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "FitnessReport":
        data = data or {}
        return cls(
            genome_id=str(data.get("genome_id", "")),
            score_valid=float(data.get("score_valid", 0.0)),
            score_test=(float(data["score_test"])
                        if data.get("score_test") is not None else None),
            fitness=float(data.get("fitness", 0.0)),
            subscores=dict(data.get("subscores") or {}),
            cost_norm=float(data.get("cost_norm", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            failures=[str(item) for item in (data.get("failures") or [])],
        )


def benchmark_for(per_kind: int = DEFAULT_PER_KIND, start: int = 0) -> list:
    """The task set a variant is scored on: hand-written plus generated.

    Generated tasks are what make the splits meaningful — the hand-written set
    has two or three tasks per kind, which cannot support a test split at all.
    """
    return list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=per_kind,
                                                         start=start)


def make_variant_request(genome: Genome, *, splits=("valid",),
                         per_kind: int = DEFAULT_PER_KIND,
                         start: int = 0, label: str = "",
                         rollout_ticks: int = ROLLOUT_TICKS) -> dict:
    """Package one genome's evaluation. Plain data, picklable.

    The skill library is exported from a *fresh seeded* one, not from the live
    system: a variant scored against whatever the live library happens to have
    learned this hour would be scored against a moving target, and two
    generations would not be comparable.
    """
    genome = Genome(genome)
    tasks = benchmark_for(per_kind=per_kind, start=start)
    split = three_way_split(tasks)
    selected = [task for name in splits for task in split.get(name, [])]
    request = make_request(
        SkillLibrary(store_path=None), selected,
        timeout=float(genome["solver_timeout"]),
        solver_order=str(genome["solver_order"]),
        label=label or genome.digest())
    request["genome"] = genome.to_dict()
    request["splits"] = list(splits)
    request["rollout_ticks"] = int(rollout_ticks)
    return request


def cost_of(genome: Genome) -> float:
    """What this configuration costs to run, normalised to roughly 0..1.

    Three things a variant can spend more of and score better for: sandbox
    time, planner width and planner depth. Left unpriced, evolution buys all
    three every generation.
    """
    genome = Genome(genome)
    timeout = (float(genome["solver_timeout"]) - 0.5) / 9.5
    beam = (float(genome["plan_beam"]) - 1.0) / 15.0
    depth = (float(genome["plan_depth"]) - 1.0) / 4.0
    return round((timeout + beam + depth) / 3.0, 6)


def rollout_score(genome: Genome, ticks: int = ROLLOUT_TICKS) -> float:
    """Mean reward over a short run of a real substrate under this genome.

    Built fresh, in whatever process this is (a pool worker), against a payoff
    that depends on the *state* — so a configuration that reads the state better
    scores better, which is precisely what the planner and the policy claim to
    do.

    Everything is redirected to a temporary directory and the clock is frozen,
    so a rollout leaves nothing behind and two rollouts of one genome give the
    same number.
    """
    import asyncio
    import importlib
    import tempfile

    from aegis.clock import frozen
    from aegis.util.quasirandom import hash_unit

    rewards: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for module_name, constant, name in (
                ("aegis.layers.memory", "MEMORY_DIR", "memory"),
                ("aegis.telemetry.store", "TELEMETRY_DIR", "telemetry"),
                ("aegis.layers.world_model", "WORLD_MODEL_DIR", "world_model"),
                ("aegis.layers.cognitive_graph", "COGNITIVE_GRAPH_DIR", "graph"),
                ("aegis.layers.evolution_engine", "EVOLUTION_DIR", "evolution"),
                ("aegis.layers.goal_intelligence", "GOAL_INTEL_DIR", "values"),
                ("aegis.layers.feedback_loop", "FEEDBACK_DIR", "feedback"),
                ("aegis.layers.substrate", "CHECKPOINTS_DIR", "checkpoints"),
                ("aegis.layers.substrate", "EVAL_DIR", "eval"),
        ):
            target = root / name
            target.mkdir(parents=True, exist_ok=True)
            module = importlib.import_module(module_name)
            if hasattr(module, constant):
                setattr(module, constant, target)
        for attribute, name in (("MOTIVATION_DIR", "motivation"),
                                ("POLICY_DIR", "policy"),
                                ("CORTEX_DIR", "cortex"),
                                ("EVOLUTION_DIR", "evolution")):
            (root / name).mkdir(parents=True, exist_ok=True)
            setattr(cfg, attribute, root / name)

        from aegis.layers.substrate import Substrate

        with frozen() as clock:
            substrate = Substrate()
            substrate.llm.enabled = False
            substrate.llm.cortex.configure_routes({})

            async def _no_agents():
                return []

            async def _no_learning(*a, **k):
                return {"success": False}

            async def _no_benchmark(tick=None):
                return None

            substrate.agent_system.run_due_agents = _no_agents
            substrate.external_learning.learn_from_source = _no_learning
            substrate.environment.step = lambda: {"reward": 0.0, "solved": False,
                                                  "task": None}
            substrate.health.check = lambda: {"status": "healthy", "warnings": [],
                                              "critical": [], "metrics": {}}
            substrate.sensors.read_all = lambda: {"pinned": True}
            substrate.world.perceive = lambda: {"pinned": True}
            substrate._run_benchmark = _no_benchmark
            substrate._last_benchmark_score = 0.5
            # An evaluation must not evaluate. The action registry now wires
            # `evolve_generation`, so a rollout's own planner can choose to run
            # a generation — each variant of which would start another rollout,
            # without bound. Marking a generation as already in flight makes the
            # action's own precondition refuse it, which is the mechanism that
            # already exists for exactly this.
            substrate.evolution.generation_running = True
            # A rollout writes nothing. Its directory disappears when the
            # context manager closes, and a store that still had a path would
            # try to write into it — which is how a temporary directory that no
            # longer exists ends up in a traceback during an unrelated test.
            substrate.evaluator._store_path = None
            substrate.apply_genome(genome)

            def scenario_reward():
                objective = substrate._ctx.decision
                if not objective:
                    return 0.15
                state = substrate._ctx.state
                energy = state.energy if state is not None else "unknown"
                try:
                    drive = substrate.goal_intelligence._classify_drive(objective)
                except Exception:
                    drive = "knowledge"
                # State-conditional on purpose: a payoff that ignored the state
                # would pay every genome the same, which is where this started.
                return 0.15 + 0.8 * hash_unit("evo_rollout", energy, drive,
                                              objective)

            substrate._compute_reward = scenario_reward

            async def go():
                for index in range(max(1, int(ticks))):
                    # Impose the energy regime, or the state never varies and
                    # the state-conditional half of the payoff is unreachable.
                    substrate.emotions.energy = (
                        0.25 if (index // 5) % 2 == 0 else 0.85)
                    await substrate.tick()
                    clock.advance(cfg.TICK_INTERVAL)
                    rewards.append(substrate._compute_reward())
                await substrate.cancel_background_tasks()

            try:
                asyncio.run(go())
            except Exception:
                logger.warning("Variant rollout failed", exc_info=True)
                return 0.0

    return sum(rewards) / len(rewards) if rewards else 0.0


def evaluate_variant(request: dict) -> dict:
    """Score one variant. Module-level and picklable, so a pool can run it."""
    from aegis.clock import CLOCK

    request = dict(request or {})
    genome = Genome(request.get("genome"))
    started = CLOCK.monotonic()
    failures: list[str] = []
    try:
        report = run_request(request)
    except Exception as exc:                        # pragma: no cover - defensive
        logger.warning("Variant evaluation failed", exc_info=True)
        report = {"score": 0.0, "per_kind": {}}
        failures.append(f"{type(exc).__name__}: {exc}")
    latency_ms = (CLOCK.monotonic() - started) * 1000

    skill_score = float(report.get("score", 0.0))
    rollout = 0.0
    ticks = int(request.get("rollout_ticks", ROLLOUT_TICKS))
    if ticks > 0:
        try:
            rollout = rollout_score(genome, ticks)
        except Exception as exc:                    # pragma: no cover - defensive
            logger.warning("Variant rollout failed", exc_info=True)
            failures.append(f"rollout: {type(exc).__name__}: {exc}")

    cost = cost_of(genome)
    latency_ms = (CLOCK.monotonic() - started) * 1000
    latency_norm = min(1.0, latency_ms / 60_000.0)
    score = SKILL_SHARE * skill_score + ROLLOUT_SHARE * rollout
    fitness = (score
               - float(cfg.EVO_COST_PENALTY) * cost
               - float(cfg.EVO_LATENCY_PENALTY) * latency_norm)

    return FitnessReport(
        genome_id=str(request.get("label") or genome.digest()),
        score_valid=score,
        fitness=fitness,
        subscores={"skills": round(skill_score, 6),
                   "rollout": round(rollout, 6),
                   **{kind: round(counts["passed"] / counts["total"], 6)
                      for kind, counts in (report.get("per_kind") or {}).items()
                      if counts.get("total")}},
        cost_norm=cost,
        latency_ms=latency_ms,
        failures=failures,
    ).as_dict()


class VariantEvaluator:
    """Runs a batch of variants through the pool, in order."""

    def __init__(self, pool=None, per_kind: int = DEFAULT_PER_KIND):
        self.pool = pool
        self.per_kind = int(per_kind)
        self.evaluated = 0

    def evaluate(self, genomes, *, splits=("valid",), start: int = 0
                 ) -> list[FitnessReport]:
        requests = [make_variant_request(genome, splits=splits,
                                         per_kind=self.per_kind, start=start,
                                         label=f"{index:02d}:{Genome(genome).digest()}")
                    for index, genome in enumerate(genomes)]
        if self.pool is None:
            raw = [evaluate_variant(request) for request in requests]
        else:
            results = self.pool.map(evaluate_variant, requests, purpose="evolution")
            raw = []
            for result, request in zip(results, requests):
                if result.ok and isinstance(result.value, dict):
                    raw.append(result.value)
                else:
                    # A variant that could not be scored is a variant that
                    # cannot be selected — reported with its reason rather than
                    # dropped, or a generation would silently shrink.
                    raw.append(FitnessReport(
                        genome_id=str(request.get("label", "")),
                        failures=[result.error or "no result"]).as_dict())
        self.evaluated += len(raw)
        return [FitnessReport.from_dict(row) for row in raw]

    def confirm(self, genome: Genome, start: int = 0) -> FitnessReport:
        """Score a champion on ``test`` — used once, never selected on."""
        request = make_variant_request(genome, splits=("test",),
                                       per_kind=self.per_kind, start=start,
                                       label=f"test:{Genome(genome).digest()}")
        return FitnessReport.from_dict(evaluate_variant(request))

    def status(self) -> dict:
        return {"per_kind": self.per_kind, "evaluated": self.evaluated,
                "pooled": self.pool is not None}
