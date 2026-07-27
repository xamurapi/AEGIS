"""What an action is worth: P(success), E[r], cost (spec M1.3).

Where the transition model answers "where does this lead", this one answers
"was it worth it". Three quantities per state/action pair, and one asymmetry
that matters more than any of them:

**Choosing uses the pessimistic estimate; reporting uses the point estimate.**
When the planner compares options it reads the lower bound of the confidence
interval, so one lucky success out of one cannot outrank a method with a solid
70% record. When a human reads the dashboard they get the actual rate, because
that is the honest answer to "how well does this work". Using the pessimistic
number for both would make the system look worse than it is; using the point
estimate for both would make it chase flukes.

Reward variance is tracked alongside the mean because it is the risk term: an
action averaging 0.5 with high variance is a different proposition from one
that reliably returns 0.5, and the planner subtracts the difference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.world.state import StateKey
from aegis.util.stats import Welford, laplace_rate, wilson_lower

logger = logging.getLogger("aegis.world.outcome")


@dataclass
class OutcomeEntry:
    """Success, reward and cost statistics for one state/action pair."""

    n: float = 0.0
    successes: float = 0.0
    reward: Welford = field(default_factory=Welford)
    cost: Welford = field(default_factory=Welford)
    updated: float = 0.0
    decayed_at: float = 0.0

    def to_dict(self) -> dict:
        return {"n": round(self.n, 6), "successes": round(self.successes, 6),
                "reward": self.reward.to_dict(), "cost": self.cost.to_dict(),
                "updated": self.updated, "decayed_at": round(self.decayed_at, 6)}

    @classmethod
    def from_dict(cls, data: dict) -> OutcomeEntry | None:
        try:
            return cls(
                n=float(data.get("n", 0.0)),
                successes=float(data.get("successes", 0.0)),
                reward=Welford.from_dict(data.get("reward")),
                cost=Welford.from_dict(data.get("cost")),
                updated=float(data.get("updated", 0.0)),
                decayed_at=float(data.get("decayed_at", 0.0)),
            )
        except (AttributeError, TypeError, ValueError):
            return None


@dataclass
class OutcomePrediction:
    """What the model expects from one action in one state."""

    state: str
    action: str
    p_success: float
    p_success_pessimistic: float
    expected_reward: float
    reward_sd: float
    expected_cost: float
    support: float
    known: float
    backed_off: bool

    def as_dict(self) -> dict:
        return {
            "state": self.state, "action": self.action,
            "p_success": round(self.p_success, 4),
            "p_success_pessimistic": round(self.p_success_pessimistic, 4),
            "expected_reward": round(self.expected_reward, 4),
            "reward_sd": round(self.reward_sd, 4),
            "expected_cost": round(self.expected_cost, 4),
            "support": round(self.support, 2),
            "known": round(self.known, 3),
            "backed_off": self.backed_off,
        }


class OutcomeModel:
    """Success rate, expected reward and cost per state/action pair."""

    #: What an action is assumed to return before anything has been observed.
    #: 0.5 rather than 0 — "no evidence" is not "evidence of failure", and a
    #: zero prior would make the planner refuse to try anything new.
    NEUTRAL_REWARD = 0.5

    def __init__(self, min_n: int | None = None, half_life: int | None = None,
                 smoothing: float | None = None):
        self.min_n = int(cfg.WM_MIN_N if min_n is None else min_n)
        self.half_life = int(cfg.WM_HALF_LIFE if half_life is None else half_life)
        self.smoothing = float(cfg.WM_SMOOTHING if smoothing is None else smoothing)
        self.pairs: dict[str, OutcomeEntry] = {}
        #: action -> entry, the back-off level for unseen states
        self.by_action: dict[str, OutcomeEntry] = {}
        self.observations = 0

    @staticmethod
    def pair_key(state, action: str) -> str:
        state_key = state.key() if isinstance(state, StateKey) else str(state)
        return f"{state_key}#{action}"

    # ── learning ─────────────────────────────────────────────────────

    def observe(self, state, action: str, success: bool,
                reward: float = 0.0, cost: float = 0.0) -> None:
        for entry in (self._entry(self.pairs, self.pair_key(state, action)),
                      self._entry(self.by_action, str(action))):
            self._decay(entry)
            entry.n += 1.0
            if success:
                entry.successes += 1.0
            entry.reward.update(float(reward))
            entry.cost.update(float(cost))
            entry.updated = CLOCK.now()
            entry.decayed_at = float(self.observations)
        self.observations += 1

    @staticmethod
    def _entry(table: dict[str, OutcomeEntry], key: str) -> OutcomeEntry:
        entry = table.get(key)
        if entry is None:
            entry = OutcomeEntry(updated=CLOCK.now())
            table[key] = entry
        return entry

    def _decay(self, entry: OutcomeEntry) -> None:
        """Age this entry by how much the model has observed elsewhere.

        Measured against the model's global count, not the entry's own: an
        entry's count only moves when the entry is updated, so using it would
        make the elapsed interval zero every time and the decay a no-op.
        """
        if self.half_life <= 0:
            return
        elapsed = float(self.observations) - entry.decayed_at
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self.half_life)
        entry.n *= factor
        entry.successes *= factor
        entry.reward.scale(factor)
        entry.cost.scale(factor)
        entry.decayed_at = float(self.observations)

    # ── estimation ───────────────────────────────────────────────────

    def _resolve(self, state, action: str) -> tuple[OutcomeEntry | None, bool]:
        """The entry to answer from, and whether it is a back-off."""
        entry = self.pairs.get(self.pair_key(state, action))
        if entry is not None and entry.n >= self.min_n:
            return entry, False
        fallback = self.by_action.get(str(action))
        if entry is not None and entry.n > 0 and fallback is None:
            return entry, False
        return fallback, True

    def p_success(self, state, action: str, pessimistic: bool = False) -> float:
        """Probability the action succeeds here.

        ``pessimistic`` picks the lower confidence bound — what a *choice*
        should use, so that a single lucky observation cannot outrank a proven
        alternative.
        """
        entry, _ = self._resolve(state, action)
        if entry is None or entry.n <= 0:
            return 0.5
        successes, trials = int(round(entry.successes)), int(round(entry.n))
        if pessimistic:
            return wilson_lower(successes, trials)
        return laplace_rate(successes, trials, self.smoothing)

    def expected_reward(self, state, action: str) -> float:
        entry, _ = self._resolve(state, action)
        if entry is None or entry.reward.n <= 0:
            return self.NEUTRAL_REWARD
        return entry.reward.mean

    def reward_sd(self, state, action: str) -> float:
        entry, _ = self._resolve(state, action)
        return entry.reward.sd() if entry else 0.0

    def expected_cost(self, state, action: str) -> float:
        entry, _ = self._resolve(state, action)
        if entry is None or entry.cost.n <= 0:
            return 0.0
        return entry.cost.mean

    def support(self, state, action: str) -> float:
        entry = self.pairs.get(self.pair_key(state, action))
        return entry.n if entry else 0.0

    def knows(self, state, action: str) -> float:
        if self.min_n <= 0:
            return 1.0
        return min(1.0, self.support(state, action) / self.min_n)

    def predict(self, state, action: str) -> OutcomePrediction:
        """Everything the model has to say about one option.

        Never raises and never refuses: an unseen pair gets the back-off
        estimate, because a planner that could not price a new action would
        never try one.
        """
        entry, backed_off = self._resolve(state, action)
        state_key = state.key() if isinstance(state, StateKey) else str(state)
        return OutcomePrediction(
            state=state_key,
            action=str(action),
            p_success=self.p_success(state, action),
            p_success_pessimistic=self.p_success(state, action, pessimistic=True),
            expected_reward=self.expected_reward(state, action),
            reward_sd=self.reward_sd(state, action),
            expected_cost=self.expected_cost(state, action),
            support=self.support(state, action),
            known=self.knows(state, action),
            backed_off=backed_off or entry is None,
        )

    def actions_seen(self) -> list[str]:
        return sorted(self.by_action)

    # ── persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "pairs": {key: entry.to_dict() for key, entry in sorted(self.pairs.items())},
            "by_action": {key: entry.to_dict()
                          for key, entry in sorted(self.by_action.items())},
            "observations": self.observations,
        }

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for table_name, table in (("pairs", self.pairs), ("by_action", self.by_action)):
            rows = data.get(table_name)
            for key, row in (rows if isinstance(rows, dict) else {}).items():
                entry = OutcomeEntry.from_dict(row) if isinstance(row, dict) else None
                if entry is not None:
                    table[str(key)] = entry
        try:
            self.observations = int(data.get("observations", 0))
        except (TypeError, ValueError):
            self.observations = 0

    def status(self) -> dict:
        return {
            "pairs": len(self.pairs),
            "actions": len(self.by_action),
            "observations": self.observations,
            "min_n": self.min_n,
        }
