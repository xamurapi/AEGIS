"""A/B harness for the behaviour policy (spec §M3.8).

Three questions, and the third is the one that keeps the other two honest:

1. **Did behaviour change?** ``behaviour_delta_rate`` must be above zero. A
   policy full of rules that never moves a decision has learned nothing that
   matters, and this is the only number that says so.
2. **Did the change pay?** Mean reward with the policy must be at least what it
   was without it. The bar is non-inferiority rather than a percentage: this
   contour's job is to *remove* what does not work, and on a scenario where
   everything works it should correctly do nothing.
3. **Does it stay quiet on noise?** On a world where outcomes are independent of
   the state, the policy must end with zero active rules. A contour that finds
   rules in noise is worse than no contour, because it looks like knowledge.

Deterministic throughout: frozen clock, no randomness, no network, no model.

Usage:
    python scripts/ab_policy.py                 # 1500 ticks per arm
    python scripts/ab_policy.py --ticks 400     # a quicker read
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)


def _run(coro):
    return _LOOP.run_until_complete(coro)


import aegis.config as cfg  # noqa: E402
from aegis.clock import frozen  # noqa: E402

PAYOFF_LOW = 0.15
PAYOFF_HIGH = 0.95

#: The trap. In this energy regime this drive is worthless, however good it
#: looks elsewhere — the exact shape of thing a suppression rule exists for.
TRAP_ENERGY = "lo"
TRAP_DRIVE = "knowledge"
TRAP_PAYOFF = 0.05


def _spread(*material: str) -> float:
    from aegis.util.quasirandom import hash_unit
    return PAYOFF_LOW + (PAYOFF_HIGH - PAYOFF_LOW) * hash_unit("ab_policy", *material)


class _NullPolicy:
    """The system as it was before stage 5: no preferences, no rules.

    Kept API-compatible rather than replaced with ``None`` so the two arms
    differ in exactly one thing — whether the policy does anything — instead of
    also differing in which code paths run at all.
    """

    def __init__(self):
        self.store = type("S", (), {"weight": 0.0, "max_preferences": 0,
                                    "status": staticmethod(lambda: {})})()
        self.experiences = []

    def delta(self, state, action):
        return 0.0

    def apply_rules(self, state, plans, tick=0):
        return list(plans)

    def observe(self, *a, **k):
        """Nothing is learned: that is what "before stage 5" means."""

    def mine(self, tick=0, safety_critical=()):
        return []

    def review(self, tick=0):
        return {}

    def active_rules(self):
        return []

    def behaviour_delta_rate(self):
        return 0.0

    def set_genome(self, genome):
        """There are no parameters to evolve when there is no policy."""

    def publish_metrics(self, tick):
        """The arm exists to be a baseline, not to be measured."""

    def save(self):
        """No state, so nothing to persist."""

    def status(self):
        return {"disabled": True}


def build(policy_enabled: bool, root: Path, noise: bool = False):
    """One arm, isolated from the other and from the repository."""
    import importlib

    from aegis.eval.skill_library import SkillLibrary
    from aegis.layers.substrate import Substrate

    for module_name, constant, name in (
            ("aegis.layers.memory", "MEMORY_DIR", "memory"),
            ("aegis.telemetry.store", "TELEMETRY_DIR", "telemetry"),
            ("aegis.layers.world_model", "WORLD_MODEL_DIR", "world_model"),
            ("aegis.layers.cognitive_graph", "COGNITIVE_GRAPH_DIR", "cognitive_graph"),
            ("aegis.layers.evolution_engine", "EVOLUTION_DIR", "evolution"),
            ("aegis.layers.goal_intelligence", "GOAL_INTEL_DIR", "goal_intelligence"),
            ("aegis.layers.feedback_loop", "FEEDBACK_DIR", "feedback"),
            ("aegis.layers.dataset_builder", "WEIGHT_DATASETS_DIR", "datasets"),
            ("aegis.layers.substrate", "CHECKPOINTS_DIR", "checkpoints"),
            ("aegis.layers.substrate", "EVAL_DIR", "eval"),
            ("aegis.layers.substrate", "CODE_BACKUPS_DIR", "code_backups"),
    ):
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module(module_name)
        if hasattr(module, constant):
            setattr(module, constant, target)

    for attribute, name in (("WORLD_MODEL_DIR", "world_model"),
                            ("COGNITIVE_GRAPH_DIR", "cognitive_graph"),
                            ("EVOLUTION_DIR", "evolution"),
                            ("GOAL_INTEL_DIR", "goal_intelligence"),
                            ("FEEDBACK_DIR", "feedback"),
                            ("CHECKPOINTS_DIR", "checkpoints"),
                            ("EVAL_DIR", "eval"),
                            ("MOTIVATION_DIR", "motivation"),
                            ("POLICY_DIR", "policy"),
                            ("CORTEX_DIR", "cortex")):
        (root / name).mkdir(parents=True, exist_ok=True)
        setattr(cfg, attribute, root / name)

    from aegis.layers.state_backup import StateBackup
    import aegis.layers.substrate as substrate_mod
    substrate_mod.StateBackup = lambda *a, **k: StateBackup(
        backup_dir=root / "backups")

    substrate = Substrate()
    substrate.skill_library = SkillLibrary(store_path=root / "eval" / "skills.json")
    substrate.solver.library = substrate.skill_library
    substrate.llm.enabled = False
    substrate.llm.cortex.configure_routes({})

    async def _no_agents():
        return []

    async def _no_learning(*args, **kwargs):
        return {"success": False}

    substrate.agent_system.run_due_agents = _no_agents
    substrate.external_learning.learn_from_source = _no_learning
    substrate.environment.step = lambda: {"reward": 0.0, "solved": False,
                                          "task": None}
    substrate.health.check = lambda: {"status": "healthy", "warnings": [],
                                      "critical": [], "metrics": {}}
    substrate.sensors.read_all = lambda: {"pinned": True}
    substrate.world.perceive = lambda: {"pinned": True}

    async def _no_training():
        return None

    substrate._weight_training_cycle = _no_training
    substrate.weight_modifier.load_model = lambda: {
        "success": False, "error": "disabled in the A/B harness"}

    if not policy_enabled:
        substrate.policy = _NullPolicy()
        substrate.planner.policy = substrate.policy

    def scenario_reward():
        objective = substrate._ctx.decision
        if not objective:
            return PAYOFF_LOW
        state = substrate._ctx.state
        energy = state.energy if state is not None else "unknown"
        try:
            drive = substrate.goal_intelligence._classify_drive(objective)
        except Exception:
            drive = "knowledge"
        if noise:
            # Outcomes independent of the state: nothing here is learnable, and
            # a policy that finds a rule in it is manufacturing knowledge.
            return _spread("noise", str(substrate.tick_count))
        if energy == TRAP_ENERGY and drive == TRAP_DRIVE:
            return TRAP_PAYOFF
        return _spread(objective)

    substrate._compute_reward = scenario_reward

    # What "success" means in this world. The miner works on success counts
    # (§M3.4), and the substrate's own definition — did this tick produce a new
    # concept — is orthogonal to the payoff being modelled here. Leaving the two
    # decoupled would give the miner a nearly constant column and nothing to
    # find, which says something about the harness rather than about the policy.
    original_observe = substrate.policy.observe

    def observe(state, action, reward, success, tick=0, experience_id=""):
        return original_observe(state, action, reward, float(reward) >= 0.5,
                                tick, experience_id)

    substrate.policy.observe = observe
    return substrate


async def drive_arm(substrate, ticks: int, clock) -> list[float]:
    rewards = []
    for _ in range(ticks):
        await substrate.tick()
        clock.advance(cfg.TICK_INTERVAL)
        rewards.append(substrate._compute_reward())
        # Mine and review on the spec's cadence. Doing it here rather than
        # through the action registry keeps the harness's timing explicit.
        tick = substrate.tick_count
        if tick % cfg.POLICY_MINE_EVERY_N_TICKS == 0:
            substrate.policy.mine(
                tick, safety_critical=[spec.name for spec
                                       in substrate.actions.safety_critical()])
        if tick % max(1, cfg.POLICY_TRIAL_TICKS // 2) == 0:
            substrate.policy.review(tick)
    return rewards


def run_arm(policy_enabled: bool, ticks: int, root: Path,
            noise: bool = False) -> dict:
    with frozen() as clock:
        substrate = build(policy_enabled, root, noise=noise)
        rewards = _run(drive_arm(substrate, ticks, clock))
        _run(substrate.cancel_background_tasks())

    tail = rewards[len(rewards) // 2:] or rewards
    return {
        "policy": policy_enabled,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_reward_tail": sum(tail) / len(tail),
        "behaviour_delta_rate": substrate.policy.behaviour_delta_rate(),
        "active_rules": len(substrate.policy.active_rules()),
        "experiences": len(substrate.policy.experiences),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=1500,
                        help="ticks per arm (default 1500)")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="how far below the baseline is still acceptable")
    args = parser.parse_args()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        without = run_arm(False, args.ticks, root / "without")
        with_policy = run_arm(True, args.ticks, root / "with")
        on_noise = run_arm(True, args.ticks, root / "noise", noise=True)

    print(f"\nA/B behaviour policy — {args.ticks} ticks per arm\n" + "=" * 56)
    for label, arm in (("without", without), ("with   ", with_policy)):
        print(f"  {label}  mean reward {arm['mean_reward']:.4f}  "
              f"(second half {arm['mean_reward_tail']:.4f})  "
              f"rules {arm['active_rules']}")
    print("-" * 56)
    print(f"  behaviour delta rate  {with_policy['behaviour_delta_rate']}")
    print(f"  active rules          {with_policy['active_rules']}")
    print(f"  rules found on noise  {on_noise['active_rules']}  (must be 0)")
    print(f"  experiences recorded  {with_policy['experiences']}")

    failures = []
    if with_policy["behaviour_delta_rate"] <= 0:
        failures.append("the policy never changed a decision "
                        "(behaviour_delta_rate is zero)")
    if with_policy["mean_reward"] < without["mean_reward"] - args.tolerance:
        failures.append(
            f"reward with the policy ({with_policy['mean_reward']:.4f}) is below "
            f"the baseline ({without['mean_reward']:.4f})")
    if on_noise["active_rules"]:
        failures.append(
            f"{on_noise['active_rules']} rule(s) were activated on pure noise — "
            "false-discovery control is not working")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK — the policy meets §M3.8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
