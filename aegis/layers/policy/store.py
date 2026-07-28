"""Learned preferences over (state, action) pairs (spec M3.3).

This is the fast, always-on half of "experience changes behaviour". Every closed
experience nudges one weight, and that weight enters the planner's score
directly. It learns quickly, forgets nothing abruptly, and is never confident —
which is the right division of labour with the rule miner (M3.4), whose job is
to be slow, explicit and evidential.

Two decisions carry the design.

**Advantage, not reward.** The update is driven by ``r − baseline(state)``,
where the baseline is the running mean reward *in that state*. Learning on
absolute reward would teach the system that everything works in good states and
nothing works in bad ones — which is true and useless, because the choice it
faces is always between actions available in the state it is actually in.

**Saturating steps.** The step is scaled by ``1 − |w|``, so a weight approaches
±1 asymptotically and never gets there. Nothing learned from experience alone
is allowed to become certain: the planner's other terms must still be able to
outvote a preference, or a run of luck early on would lock a choice in
permanently.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.store.migrations import read_store, write_store
from aegis.util.stats import clamp, exponential_smooth

logger = logging.getLogger("aegis.policy")

#: How fast a state's reward baseline follows what actually happens there. Slow
#: on purpose: the baseline is the thing preferences are measured against, and a
#: baseline that chased every sample would leave nothing to measure.
BASELINE_ALPHA = 0.1

#: Below this many observations a preference is treated as evidence-free and
#: contributes nothing, no matter what its weight says. One sample is a story.
MIN_OBSERVATIONS = 2


class PolicyStore:
    """``(state, action) -> weight``, learned from closed experience."""

    def __init__(self, store_path: Path | None = None,
                 learning_rate: float | None = None,
                 weight: float | None = None,
                 max_preferences: int | None = None):
        self._store_path = store_path or (cfg.POLICY_DIR / "preferences.json")
        self.learning_rate = float(
            cfg.POLICY_LR if learning_rate is None else learning_rate)
        #: How much a preference is allowed to move the planner's score. A gene
        #: (Appendix C, ``policy_weight``), not a constant.
        self.weight = float(cfg.POLICY_WEIGHT if weight is None else weight)
        self.max_preferences = int(
            cfg.POLICY_MAX_PREFS if max_preferences is None else max_preferences)

        #: "state|action" -> {"w", "n", "updated"}
        self.preferences: dict[str, dict] = {}
        #: state key -> running mean reward in that state
        self.baselines: dict[str, float] = {}
        self.updates = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    @staticmethod
    def pair_key(state, action: str) -> str:
        state_key = state.key() if hasattr(state, "key") else str(state)
        return f"{state_key}#{action}"

    def _load(self) -> None:
        data = read_store(self._store_path, store="policy_preferences")
        for key, row in (data.get("preferences") or {}).items():
            if not isinstance(row, dict):
                continue
            try:
                self.preferences[str(key)] = {
                    "w": clamp(float(row.get("w", 0.0)), -1.0, 1.0),
                    "n": max(0, int(row.get("n", 0))),
                    "updated": float(row.get("updated", 0.0)),
                }
            except (TypeError, ValueError):
                logger.debug("Ignoring malformed preference %r", key)
        for key, value in (data.get("baselines") or {}).items():
            try:
                self.baselines[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        try:
            self.updates = max(0, int(data.get("updates", 0)))
        except (TypeError, ValueError):
            self.updates = 0

    def save(self) -> None:
        write_store(self._store_path, {
            "preferences": {key: dict(row)
                            for key, row in sorted(self.preferences.items())},
            "baselines": dict(sorted(self.baselines.items())),
            "updates": self.updates,
        })

    # ── learning ─────────────────────────────────────────────────────

    def baseline(self, state) -> float:
        """What this state normally pays, regardless of what is done in it."""
        state_key = state.key() if hasattr(state, "key") else str(state)
        return self.baselines.get(state_key, 0.0)

    def update(self, state, action: str, reward: float) -> float:
        """Fold one closed experience in, and return the new weight.

        The baseline moves *after* the advantage is computed. Updating it first
        would mean measuring each sample against a baseline that had already
        absorbed it, which shrinks every advantage toward zero and makes the
        store learn more slowly the more it sees.
        """
        if not action:
            return 0.0
        state_key = state.key() if hasattr(state, "key") else str(state)
        try:
            reward = float(reward)
        except (TypeError, ValueError):
            return 0.0

        advantage = reward - self.baselines.get(state_key, reward)
        self.baselines[state_key] = exponential_smooth(
            self.baselines.get(state_key), reward, BASELINE_ALPHA)

        key = f"{state_key}#{action}"
        row = self.preferences.setdefault(key, {"w": 0.0, "n": 0, "updated": 0.0})
        step = self.learning_rate * advantage * (1.0 - abs(row["w"]))
        row["w"] = clamp(row["w"] + step, -1.0, 1.0)
        row["n"] += 1
        row["updated"] = CLOCK.now()
        self.updates += 1
        self._evict_if_needed()
        return row["w"]

    # ── reading ──────────────────────────────────────────────────────

    def weight_for(self, state, action: str) -> float:
        """The raw learned weight, before it is scaled into a score."""
        row = self.preferences.get(self.pair_key(state, action))
        if row is None or row["n"] < MIN_OBSERVATIONS:
            return 0.0
        return float(row["w"])

    def delta(self, state, action: str) -> float:
        """What this preference adds to the planner's score (Appendix J, step 4)."""
        return self.weight_for(state, action) * self.weight

    def preferred(self, state, actions) -> list[tuple[str, float]]:
        """Actions ranked by learned preference, best first.

        Ties break on the action name, because a ranking that depended on the
        order the caller happened to pass would make two identical runs diverge.
        """
        scored = [(action, self.weight_for(state, action)) for action in actions]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

    # ── capacity ─────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Drop the least informative preferences when the store is full.

        Informativeness, not age: a weight far from zero backed by many
        observations is the store's actual knowledge, and evicting by recency
        would throw exactly that away first while keeping a shelf of noise.
        """
        if len(self.preferences) <= self.max_preferences:
            return
        ranked = sorted(self.preferences.items(),
                        key=lambda item: (self._retention_score(item[1]), item[0]))
        for key, _ in ranked[:len(self.preferences) - self.max_preferences]:
            del self.preferences[key]

    @staticmethod
    def _retention_score(row: dict) -> float:
        return abs(float(row.get("w", 0.0))) * min(1.0, float(row.get("n", 0)) / 10.0)

    def regulate_capacity(self, max_preferences: int) -> None:
        self.max_preferences = max(100, int(max_preferences))
        self._evict_if_needed()

    # ── reporting ────────────────────────────────────────────────────

    def strongest(self, limit: int = 10) -> list[dict]:
        ranked = sorted(self.preferences.items(),
                        key=lambda item: (-abs(item[1]["w"]), item[0]))
        return [{"pair": key, "w": round(row["w"], 4), "n": row["n"]}
                for key, row in ranked[:limit]]

    def status(self) -> dict:
        return {
            "preferences": len(self.preferences),
            "states": len(self.baselines),
            "updates": self.updates,
            "learning_rate": self.learning_rate,
            "weight": self.weight,
            "max_preferences": self.max_preferences,
            "strongest": self.strongest(5),
        }
