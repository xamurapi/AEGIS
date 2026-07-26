"""System 3: Evolution Engine — version A → mutation → version B → benchmark → keep the best (EV-001..EV-005).

Natural selection over the system's own tunable parameters:

    champion genome → deterministic mutation → candidate →
    held-out benchmark fitness → keep only if BETTER, else roll back

This closes the "self-modification ≠ self-improvement" gap: a change survives
only when an external, verifiable metric says it is an improvement. Fitness
comes from the Evaluator's held-out benchmark, never from self-report.
"""
import json
import logging
from pathlib import Path

from aegis.config import EVOLUTION_DIR
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.evolution")

# Pure audit trail — never read during inference, so length costs nothing but
# disk. Kept long enough to reconstruct a full selection history.
MAX_LINEAGE = 2000
MUTATION_MAGNITUDE = 0.1  # ±10% per mutated parameter
# A candidate must beat the champion by at least this margin to win —
# guards against benchmark noise promoting a sideways change.
FITNESS_EPSILON = 0.005


class EvolutionEngine:
    def __init__(self, store_path: Path | None = None):
        self.champion: dict | None = None      # {genome, fitness, generation, created}
        self.candidate: dict | None = None     # {genome, mutated_param, old_value, proposed_at_tick}
        self.generation = 0
        self.accepted = 0
        self.rejected = 0
        self.lineage: list[dict] = []
        self._param_idx = 0       # round-robin over genome keys
        self._direction_up = True  # alternate mutation direction
        self._store_path = store_path or (EVOLUTION_DIR / "lineage.json")
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self.champion = data.get("champion")
                self.candidate = data.get("candidate")
                self.generation = data.get("generation", 0)
                self.accepted = data.get("accepted", 0)
                self.rejected = data.get("rejected", 0)
                self.lineage = data.get("lineage", [])
                self._param_idx = data.get("param_idx", 0)
                self._direction_up = data.get("direction_up", True)
            except Exception:
                logger.warning("Failed to load evolution state from %s — starting fresh",
                               self._store_path, exc_info=True)

    def save(self):
        data = {
            "champion": self.champion,
            "candidate": self.candidate,
            "generation": self.generation,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "lineage": self.lineage[-MAX_LINEAGE:],
            "param_idx": self._param_idx,
            "direction_up": self._direction_up,
        }
        try:
            payload = json.dumps(data, ensure_ascii=False)
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to save evolution state to %s", self._store_path, exc_info=True)

    # ── evolution cycle ──────────────────────────────────────────────

    def register_champion(self, genome: dict, fitness: float):
        """Set the current best-known genome (version A). Called on boot with the
        live parameters and the latest benchmark score."""
        self.champion = {
            "genome": {k: float(v) for k, v in genome.items()},
            "fitness": float(fitness),
            "generation": self.generation,
            "created": CLOCK.now(),
        }

    def propose_mutation(self, tick: int) -> dict | None:
        """Create version B: mutate ONE parameter of the champion genome,
        deterministically (round-robin key, alternating ±10%).

        Returns {param, old_value, new_value} for the caller to apply through
        its safety pipeline, or None if a candidate is already pending or
        there is no champion yet.
        """
        if self.candidate is not None or not self.champion:
            return None
        params = sorted(self.champion["genome"])
        if not params:
            return None
        param = params[self._param_idx % len(params)]
        self._param_idx += 1
        old_value = self.champion["genome"][param]
        up = self._direction_up
        factor = (1 + MUTATION_MAGNITUDE) if up else (1 - MUTATION_MAGNITUDE)
        self._direction_up = not up
        new_value = old_value * factor
        if new_value == old_value:
            # A zero (or subnormal) parameter is a fixed point of the
            # multiplicative step, so that genome slot could never be explored —
            # the round-robin kept burning benchmark cycles on a no-op mutation
            # (audit R3-3). Fall back to an ADDITIVE step; the caller clamps it
            # to the parameter's real bounds.
            new_value = old_value + (MUTATION_MAGNITUDE if up else -MUTATION_MAGNITUDE)
        genome = dict(self.champion["genome"])
        genome[param] = new_value

        self.candidate = {
            "genome": genome,
            "mutated_param": param,
            "old_value": old_value,
            "new_value": new_value,
            "proposed_at_tick": tick,
            "created": CLOCK.now(),
        }
        return {"param": param, "old_value": old_value, "new_value": new_value}

    def judge_candidate(self, fitness: float) -> dict:
        """Benchmark arrived — natural selection. The candidate becomes the new
        champion only if it beats the champion's fitness by FITNESS_EPSILON;
        otherwise it is rejected and the caller must restore old_value.

        Returns {decision: "accepted"|"rejected", param, revert_to}.
        """
        if self.candidate is None or not self.champion:
            return {"decision": "no_candidate"}
        cand = self.candidate
        self.candidate = None
        self.generation += 1
        record = {
            "generation": self.generation,
            "param": cand["mutated_param"],
            "old_value": cand["old_value"],
            "new_value": cand["new_value"],
            "champion_fitness": self.champion["fitness"],
            "candidate_fitness": float(fitness),
            "time": CLOCK.now(),
        }
        if fitness > self.champion["fitness"] + FITNESS_EPSILON:
            self.accepted += 1
            record["decision"] = "accepted"
            self.champion = {
                "genome": cand["genome"],
                "fitness": float(fitness),
                "generation": self.generation,
                "created": CLOCK.now(),
            }
            result = {"decision": "accepted", "param": cand["mutated_param"],
                      "revert_to": None}
        else:
            self.rejected += 1
            record["decision"] = "rejected"
            result = {"decision": "rejected", "param": cand["mutated_param"],
                      "revert_to": cand["old_value"]}
        self.lineage.append(record)
        if len(self.lineage) > MAX_LINEAGE:
            self.lineage = self.lineage[-MAX_LINEAGE:]
        self.save()
        return result

    def abandon_candidate(self) -> dict | None:
        """Drop a pending candidate without judging it (e.g. it could not be
        applied through the safety pipeline). Returns revert info."""
        if self.candidate is None:
            return None
        cand = self.candidate
        self.candidate = None
        return {"param": cand["mutated_param"], "revert_to": cand["old_value"]}

    # ── status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "generation": self.generation,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "champion_fitness": self.champion["fitness"] if self.champion else None,
            "candidate_pending": self.candidate is not None,
            "candidate_param": self.candidate["mutated_param"] if self.candidate else None,
            "recent_lineage": self.lineage[-5:],
        }
