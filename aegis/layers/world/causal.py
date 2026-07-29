"""The causal layer: which causes tend to produce which effects.

This is the original world model, unchanged in behaviour and renamed. It answers
a different question from the predictive layer above it and remains useful for
it: the predictive model is keyed on *encoded system state*, which is precise
but says nothing a human recognises, while these links are keyed on free text —
"attempt:gcd" leads to "solved" — and are what the autobiography, the risk
lookup and the objective chains are built from.

Kept as its own class rather than folded in, so that the part with years of
recorded observations behind it is not disturbed by the part being introduced.
"""
import json
import re
import logging
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.world.causal")

MAX_LINKS = 2000          # baseline floor; the live cap is self.max_links
MAX_CHAINS = 50
MIN_OBSERVATIONS_FOR_PREDICTION = 2

# Retention. Pruning used to sort on observation count alone, so a rare but
# decisive failure was dropped before a frequent coin-flip link — backwards for
# a memory of what goes wrong. A link is now kept for how much it TELLS us:
# how far its success rate sits from 50/50 (decisiveness), how much evidence
# backs that (saturating), and a deliberate bias toward remembering failure.
EVIDENCE_SATURATION = 10.0   # observations beyond this add no further weight
FAILURE_RETENTION_BIAS = 1.5  # a known failure outranks an equally decisive win

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


class CausalLinks:
    def __init__(self, store_path: Path | None = None):
        # links[cause][effect] = {observations, successes, updated}
        self.links: dict[str, dict[str, dict]] = {}
        self.chains: list[dict] = []
        self.total_observations = 0
        # Live capacity. A module constant could only ever be a guess; this is
        # the floor, and Substrate.regulate_capacity() raises it while ticks are
        # cheap and gives it back when they are not.
        self.max_links = MAX_LINKS
        # Derived index: token -> causes containing it. Not persisted (the
        # on-disk format is unchanged); rebuilt on load, like the cognitive
        # graph's in-degree index. Without it every risk lookup walked the whole
        # cause table, and risks_for now runs on every tick.
        self._cause_index: dict[str, set[str]] = {}
        # How much a known failure outranks an equally decisive win when
        # pruning. An instance attribute rather than the module constant because
        # `mem_retention_bias` is a gene: it declared `WorldModel._retention_score`
        # as its reader, but the value was being written onto MemorySystem, where
        # nothing read it. The constant is the default, so an un-evolved system
        # prunes exactly as before.
        self.failure_retention_bias = FAILURE_RETENTION_BIAS
        self._store_path = store_path or (cfg.WORLD_MODEL_DIR / "model.json")
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
        self._rebuild_index()

    # ── cause index ──────────────────────────────────────────────────

    @staticmethod
    def _tokens(text: str) -> set[str]:
        """Tokens of a cause/query string, lowercased and punctuation-split.

        Matching is by token, not by raw substring: a query for ``rest`` no
        longer matches ``forest_fire``. Identifiers like ``expand_knowledge``
        still match ``expand`` because ``_`` is a separator.
        """
        return {t for t in _TOKEN_SPLIT.split(text.lower()) if t}

    def _rebuild_index(self):
        self._cause_index = {}
        for cause in self.links:
            self._index_cause(cause)

    def _index_cause(self, cause: str):
        for token in self._tokens(cause):
            self._cause_index.setdefault(token, set()).add(cause)

    def _deindex_cause(self, cause: str):
        for token in self._tokens(cause):
            bucket = self._cause_index.get(token)
            if bucket is not None:
                bucket.discard(cause)
                if not bucket:
                    del self._cause_index[token]

    def _candidate_causes(self, tokens: list[str]) -> list[tuple[str, dict]]:
        """Causes worth examining for these query tokens.

        Sorted so the result order never depends on set iteration order — the
        zero-randomness guarantee covers this path too.
        """
        query = {t for raw in tokens if raw for t in self._tokens(raw)}
        if not query:
            return list(self.links.items())
        causes: set[str] = set()
        for token in query:
            causes |= self._cause_index.get(token, set())
        return [(c, self.links[c]) for c in sorted(causes) if c in self.links]

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
            "observations": 0, "successes": 0, "updated": CLOCK.now(),
        })
        link["observations"] += 1
        if success:
            link["successes"] += 1
        link["updated"] = CLOCK.now()
        self.total_observations += 1
        self._index_cause(cause)
        self._prune(protect=(cause, effect))

    @staticmethod
    def _strength(link: dict) -> float:
        """Laplace-smoothed success probability of the link."""
        return (link["successes"] + 1) / (link["observations"] + 2)

    def _retention_score(self, link: dict) -> float:
        """How much this link tells us. Higher survives pruning longer."""
        strength = self._strength(link)
        decisiveness = abs(strength - 0.5) * 2
        evidence = min(1.0, link["observations"] / EVIDENCE_SATURATION)
        bias = self.failure_retention_bias if strength < 0.5 else 1.0
        return decisiveness * evidence * bias

    def _prune(self, protect: tuple[str, str] | None = None):
        total_links = sum(len(v) for v in self.links.values())
        if total_links <= self.max_links:
            return
        # Drop the least informative links first — NOT simply the least
        # observed. `protect` shields the link just recorded: otherwise a fresh
        # observation, which necessarily has the least evidence, could be
        # evicted by the very prune it triggered and nothing new could ever be
        # learned once the model was full.
        flat = [(c, e, l) for c, effects in self.links.items() for e, l in effects.items()
                if (c, e) != protect]
        flat.sort(key=lambda x: (self._retention_score(x[2]), x[2]["updated"]))
        for cause, effect, _ in flat[:total_links - self.max_links]:
            del self.links[cause][effect]
            if not self.links[cause]:
                del self.links[cause]
                self._deindex_cause(cause)

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
        risks = []
        for cause, effects in self._candidate_causes(tokens):
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
        for cause, effects in self._candidate_causes(tokens):
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
            "created": CLOCK.now(),
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

        def _as_list(value) -> list:
            """An LLM may answer with a dict, a bare string or a number where a
            list was asked for. Slicing those either explodes (dict) or silently
            iterates characters (str), so coerce the shape first (audit R3-8)."""
            return value if isinstance(value, list) else []

        def _as_confidence(value) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.5

        chain = {
            "objective": str(parsed.get("objective"))[:200],
            "constraints": [str(c)[:120] for c in _as_list(parsed.get("constraints"))[:5]],
            "risks": [r if isinstance(r, dict) else
                      {"cause": str(r)[:120], "effect": "", "failure_rate": 0.5, "observations": 0}
                      for r in _as_list(parsed.get("risks"))[:5]],
            "plan": [s if isinstance(s, dict) else
                     {"action": str(s)[:160], "expected": "", "confidence": 0.5}
                     for s in _as_list(parsed.get("plan"))[:7]],
            "expected_result": str(parsed.get("expected_result", ""))[:200],
            "confidence": _as_confidence(parsed.get("confidence", 0.5)),
            "source": "llm",
            "created": CLOCK.now(),
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
