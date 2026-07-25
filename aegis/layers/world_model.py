"""System 1: World Model — causal model of actions and consequences (WM-001..WM-005).

Learns cause→effect links from REAL observed outcomes (environment steps,
decisions, training runs) and predicts likely consequences of actions.
Builds structured causal chains for objectives:

    objective → constraints → risks → plan → expected result

The core is deterministic (Laplace-smoothed frequency estimates); LLM-assisted
chain refinement is optional and driven by the Substrate.
"""
import json
import time
import logging
from pathlib import Path

from aegis.config import WORLD_MODEL_DIR

logger = logging.getLogger("aegis.world_model")

MAX_LINKS = 2000
MAX_CHAINS = 50
MIN_OBSERVATIONS_FOR_PREDICTION = 2


class WorldModel:
    def __init__(self, store_path: Path | None = None):
        # links[cause][effect] = {observations, successes, updated}
        self.links: dict[str, dict[str, dict]] = {}
        self.chains: list[dict] = []
        self.total_observations = 0
        self._store_path = store_path or (WORLD_MODEL_DIR / "model.json")
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self.links = data.get("links", {})
                self.chains = data.get("chains", [])
                self.total_observations = data.get("total_observations", 0)
            except Exception:
                logger.warning("Failed to load world model from %s — starting empty",
                               self._store_path, exc_info=True)

    def save(self):
        data = {
            "links": self.links,
            "chains": self.chains[-MAX_CHAINS:],
            "total_observations": self.total_observations,
        }
        try:
            payload = json.dumps(data, ensure_ascii=False)
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to save world model to %s", self._store_path, exc_info=True)

    # ── learning ─────────────────────────────────────────────────────

    def observe(self, cause: str, effect: str, success: bool = True):
        """Record one observed cause→effect transition with its outcome."""
        cause = cause[:80]
        effect = effect[:80]
        link = self.links.setdefault(cause, {}).setdefault(effect, {
            "observations": 0, "successes": 0, "updated": time.time(),
        })
        link["observations"] += 1
        if success:
            link["successes"] += 1
        link["updated"] = time.time()
        self.total_observations += 1
        self._prune()

    @staticmethod
    def _strength(link: dict) -> float:
        """Laplace-smoothed success probability of the link."""
        return (link["successes"] + 1) / (link["observations"] + 2)

    def _prune(self):
        total_links = sum(len(v) for v in self.links.values())
        if total_links <= MAX_LINKS:
            return
        # Drop the least-observed, oldest links first.
        flat = [(c, e, l) for c, effects in self.links.items() for e, l in effects.items()]
        flat.sort(key=lambda x: (x[2]["observations"], x[2]["updated"]))
        for cause, effect, _ in flat[:total_links - MAX_LINKS]:
            del self.links[cause][effect]
            if not self.links[cause]:
                del self.links[cause]

    # ── inference ────────────────────────────────────────────────────

    def predict(self, cause: str, k: int = 5) -> list[dict]:
        """Predict the most likely effects of a cause, by observed strength."""
        effects = self.links.get(cause[:80], {})
        ranked = sorted(
            ({"effect": e, "strength": round(self._strength(l), 3),
              "observations": l["observations"]}
             for e, l in effects.items()
             if l["observations"] >= MIN_OBSERVATIONS_FOR_PREDICTION),
            key=lambda x: x["strength"], reverse=True,
        )
        return ranked[:k]

    def explain(self, effect: str, k: int = 5) -> list[dict]:
        """Find the most likely causes of an observed effect."""
        effect = effect[:80]
        causes = []
        for cause, effects in self.links.items():
            link = effects.get(effect)
            if link and link["observations"] >= MIN_OBSERVATIONS_FOR_PREDICTION:
                causes.append({"cause": cause, "strength": round(self._strength(link), 3),
                               "observations": link["observations"]})
        causes.sort(key=lambda x: x["strength"], reverse=True)
        return causes[:k]

    def risks_for(self, tokens: list[str], k: int = 5) -> list[dict]:
        """Weak links (low success rate) whose cause matches any token — the
        model's memory of what tends to FAIL around this topic."""
        toks = [t.lower() for t in tokens if t]
        risks = []
        for cause, effects in self.links.items():
            if toks and not any(t in cause.lower() for t in toks):
                continue
            for effect, link in effects.items():
                s = self._strength(link)
                if s < 0.5 and link["observations"] >= MIN_OBSERVATIONS_FOR_PREDICTION:
                    risks.append({"cause": cause, "effect": effect,
                                  "failure_rate": round(1 - s, 3),
                                  "observations": link["observations"]})
        risks.sort(key=lambda x: x["failure_rate"], reverse=True)
        return risks[:k]

    # ── causal chains ────────────────────────────────────────────────

    def build_chain(self, objective: str, constraints: list[str] | None = None) -> dict:
        """Build a deterministic causal chain for an objective:
        objective → constraints → risks → plan → expected result.

        Plan steps come from the strongest known links matching the objective;
        risks from the weakest. An LLM refinement can later replace the plan
        via refine_chain() — the deterministic version keeps the system
        functional without any LLM.
        """
        tokens = [t for t in objective.lower().split() if len(t) > 2]
        risks = self.risks_for(tokens)

        # Plan: strongest links whose cause matches the objective.
        steps = []
        for cause, effects in self.links.items():
            if tokens and not any(t in cause.lower() for t in tokens):
                continue
            for effect, link in effects.items():
                s = self._strength(link)
                if s >= 0.5 and link["observations"] >= MIN_OBSERVATIONS_FOR_PREDICTION:
                    steps.append({"action": cause, "expected": effect,
                                  "confidence": round(s, 3)})
        steps.sort(key=lambda x: x["confidence"], reverse=True)
        steps = steps[:5]

        chain = {
            "objective": objective[:200],
            "constraints": list(constraints or [])[:5],
            "risks": risks,
            "plan": steps,
            "expected_result": steps[0]["expected"] if steps else "unknown — no causal data yet",
            "confidence": round(sum(s["confidence"] for s in steps) / len(steps), 3) if steps else 0.0,
            "source": "world_model",
            "created": time.time(),
        }
        self.chains.append(chain)
        if len(self.chains) > MAX_CHAINS:
            self.chains = self.chains[-MAX_CHAINS:]
        return chain

    def refine_chain(self, parsed: dict) -> dict | None:
        """Accept an LLM-proposed chain (already JSON-parsed), validate its
        shape, and store it. Returns the stored chain or None if malformed."""
        if not isinstance(parsed, dict) or not parsed.get("objective"):
            return None
        chain = {
            "objective": str(parsed.get("objective"))[:200],
            "constraints": [str(c)[:120] for c in parsed.get("constraints", [])[:5]],
            "risks": [{"cause": str(r)[:120], "effect": "", "failure_rate": 0.5, "observations": 0}
                      if isinstance(r, str) else r
                      for r in parsed.get("risks", [])[:5]],
            "plan": [{"action": str(s)[:160], "expected": "", "confidence": 0.5}
                     if isinstance(s, str) else s
                     for s in parsed.get("plan", [])[:7]],
            "expected_result": str(parsed.get("expected_result", ""))[:200],
            "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5) or 0.5))),
            "source": "llm",
            "created": time.time(),
        }
        self.chains.append(chain)
        if len(self.chains) > MAX_CHAINS:
            self.chains = self.chains[-MAX_CHAINS:]
        return chain

    # ── status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        flat = [(c, e, self._strength(l), l["observations"])
                for c, effects in self.links.items() for e, l in effects.items()]
        flat.sort(key=lambda x: (x[3], x[2]), reverse=True)
        return {
            "causes": len(self.links),
            "links": sum(len(v) for v in self.links.values()),
            "total_observations": self.total_observations,
            "chains_built": len(self.chains),
            "strongest_links": [
                {"cause": c, "effect": e, "strength": round(s, 3), "observations": n}
                for c, e, s, n in flat[:5]
            ],
            "latest_chain": self.chains[-1] if self.chains else None,
        }
