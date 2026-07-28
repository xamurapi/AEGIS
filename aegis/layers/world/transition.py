"""Where an action leads: P(s' | s, a) (spec M1.3, M1.5).

The counting is trivial; the two things that make it useful are not.

**Back-off.** A state/action pair seen twice has no business reporting a
confident distribution. Below ``WM_MIN_N`` observations the estimate falls back
to what the action does *in general*, and below that to what happens in
general. So a planner asking about an unseen combination gets the honest prior
instead of an accident of the first two samples.

**Forgetting.** After evolution promotes a genome or a new skill is accepted,
the world the model learned is literally not the world any more. Counts decay
with a half-life measured in observations, so old evidence loses weight without
being deleted — the model follows the drift instead of averaging over a system
that no longer exists.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.world.state import StateKey

logger = logging.getLogger("aegis.world.transition")


@dataclass
class TransitionEntry:
    """Everything observed after taking one action in one state."""

    n: float = 0.0
    next: dict[str, float] = field(default_factory=dict)
    updated: float = 0.0
    #: The MODEL's total observation count when this entry was last touched.
    #: Ageing is driven by how much the world has moved on, not by wall-clock
    #: time — a system that sat idle overnight has learned nothing new to
    #: forget — and not by this entry's own count, which would make the elapsed
    #: interval zero at every update and the decay a no-op.
    decayed_at: float = 0.0

    def to_dict(self) -> dict:
        return {"n": round(self.n, 6),
                "next": {k: round(v, 6) for k, v in sorted(self.next.items())},
                "updated": self.updated, "decayed_at": round(self.decayed_at, 6)}

    @classmethod
    def from_dict(cls, data: dict) -> TransitionEntry | None:
        try:
            successors = data.get("next") or {}
            return cls(
                n=float(data.get("n", 0.0)),
                next={str(k): float(v) for k, v in successors.items()},
                updated=float(data.get("updated", 0.0)),
                decayed_at=float(data.get("decayed_at", 0.0)),
            )
        except (AttributeError, TypeError, ValueError):
            return None


class TransitionModel:
    """Smoothed, decaying estimate of where each action leads."""

    def __init__(self, smoothing: float | None = None, min_n: int | None = None,
                 half_life: int | None = None, max_states: int | None = None):
        self.smoothing = float(cfg.WM_SMOOTHING if smoothing is None else smoothing)
        self.min_n = int(cfg.WM_MIN_N if min_n is None else min_n)
        self.half_life = int(cfg.WM_HALF_LIFE if half_life is None else half_life)
        self.max_states = int(cfg.WM_MAX_STATES if max_states is None else max_states)

        #: "state|action" -> entry
        self.pairs: dict[str, TransitionEntry] = {}
        #: action -> successor counts, the first back-off level
        self.by_action: dict[str, dict[str, float]] = {}
        #: successor counts overall, the last back-off level
        self.prior: dict[str, float] = {}
        self.observations = 0
        self.collapsed = 0

    # ── keys ─────────────────────────────────────────────────────────

    @staticmethod
    def pair_key(state, action: str) -> str:
        state_key = state.key() if isinstance(state, StateKey) else str(state)
        return f"{state_key}#{action}"

    def states(self) -> set[str]:
        return {key.split("#", 1)[0] for key in self.pairs}

    # ── learning ─────────────────────────────────────────────────────

    def observe(self, state, action: str, next_state) -> None:
        """Record one observed transition."""
        key = self.pair_key(state, action)
        successor = next_state.key() if isinstance(next_state, StateKey) else str(next_state)

        entry = self.pairs.get(key)
        if entry is None:
            entry = TransitionEntry(updated=CLOCK.now())
            self.pairs[key] = entry
        else:
            self._decay(entry)

        entry.n += 1.0
        entry.next[successor] = entry.next.get(successor, 0.0) + 1.0
        entry.updated = CLOCK.now()
        entry.decayed_at = float(self.observations)

        action_counts = self.by_action.setdefault(str(action), {})
        action_counts[successor] = action_counts.get(successor, 0.0) + 1.0
        self.prior[successor] = self.prior.get(successor, 0.0) + 1.0
        self.observations += 1
        self._collapse_if_needed()

    def _decay(self, entry: TransitionEntry) -> None:
        """Halve this entry's weight for every ``half_life`` observations the
        model made elsewhere since it was last seen."""
        if self.half_life <= 0:
            return
        elapsed = float(self.observations) - entry.decayed_at
        if elapsed <= 0:
            return
        factor = 0.5 ** (elapsed / self.half_life)
        entry.n *= factor
        for successor in list(entry.next):
            entry.next[successor] *= factor
            # A successor whose weight has faded below a millionth of an
            # observation is noise; keeping it would grow the table forever.
            if entry.next[successor] < 1e-6:
                del entry.next[successor]
        entry.decayed_at = float(self.observations)

    # ── estimation ───────────────────────────────────────────────────

    def _backoff(self, action: str, successor: str) -> float:
        """P(s' | a), falling through to P(s') when the action is new."""
        action_counts = self.by_action.get(str(action))
        if action_counts:
            total = sum(action_counts.values())
            if total > 0:
                return action_counts.get(successor, 0.0) / total
        total = sum(self.prior.values())
        if total > 0:
            return self.prior.get(successor, 0.0) / total
        return 0.0

    def probability(self, state, action: str, next_state) -> float:
        """P(s' | s, a) with additive smoothing toward the back-off estimate."""
        successor = next_state.key() if isinstance(next_state, StateKey) else str(next_state)
        entry = self.pairs.get(self.pair_key(state, action))
        alpha = self.smoothing
        backoff = self._backoff(action, successor)
        if entry is None or entry.n <= 0:
            return backoff
        count = entry.next.get(successor, 0.0)
        return (count + alpha * backoff) / (entry.n + alpha)

    def top_next(self, state, action: str, k: int = 3) -> list[tuple[str, float]]:
        """The ``k`` most likely successors, most likely first.

        Ties break on the state key so a rollout never depends on dict order
        (§3.1) — two identical runs must expand the same branches.
        """
        entry = self.pairs.get(self.pair_key(state, action))
        candidates = dict(entry.next) if entry and entry.n > 0 else {}
        if not candidates:
            candidates = dict(self.by_action.get(str(action), {})) or dict(self.prior)
        if not candidates:
            return []
        scored = [(successor, self.probability(state, action, successor))
                  for successor in candidates]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored[:max(0, int(k))]

    def support(self, state, action: str) -> float:
        entry = self.pairs.get(self.pair_key(state, action))
        return entry.n if entry else 0.0

    def knows(self, state, action: str) -> float:
        """How well this pair is understood, on 0..1.

        Saturating at ``min_n`` rather than growing without bound: past the
        point where the estimate is usable, more evidence should not keep
        suppressing the exploration bonus.
        """
        if self.min_n <= 0:
            return 1.0
        return min(1.0, self.support(state, action) / self.min_n)

    def surprise(self, state, action: str, actual_next) -> float:
        """−log P(actual), the information content of what happened."""
        from aegis.util.stats import safe_log
        return -safe_log(self.probability(state, action, actual_next))

    # ── capacity ─────────────────────────────────────────────────────

    def _collapse_if_needed(self) -> None:
        """Drop the least informative pairs once the table outgrows its budget.

        Least *informative*, not least recent: a pair with one observation
        tells almost nothing, while a rare pair observed forty times is
        precisely the kind of thing worth keeping. What is dropped still
        survives in the back-off marginals, so the model loses resolution
        rather than knowledge.
        """
        if self.max_states <= 0 or len(self.pairs) <= self.max_states:
            return
        excess = len(self.pairs) - self.max_states
        ranked = sorted(self.pairs.items(), key=lambda kv: (kv[1].n, kv[0]))
        for key, _ in ranked[:excess]:
            del self.pairs[key]
        self.collapsed += excess

    # ── persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "pairs": {key: entry.to_dict() for key, entry in sorted(self.pairs.items())},
            "by_action": {action: {k: round(v, 6) for k, v in sorted(counts.items())}
                          for action, counts in sorted(self.by_action.items())},
            "prior": {k: round(v, 6) for k, v in sorted(self.prior.items())},
            "observations": self.observations,
            "collapsed": self.collapsed,
        }

    def load(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        pairs = data.get("pairs")
        for key, row in (pairs if isinstance(pairs, dict) else {}).items():
            entry = TransitionEntry.from_dict(row) if isinstance(row, dict) else None
            if entry is not None:
                self.pairs[str(key)] = entry
        by_action = data.get("by_action")
        for action, counts in (by_action if isinstance(by_action, dict) else {}).items():
            if isinstance(counts, dict):
                self.by_action[str(action)] = {
                    str(k): float(v) for k, v in counts.items()
                    if isinstance(v, (int, float))}
        prior = data.get("prior")
        if isinstance(prior, dict):
            self.prior = {str(k): float(v) for k, v in prior.items()
                          if isinstance(v, (int, float))}
        try:
            self.observations = int(data.get("observations", 0))
            self.collapsed = int(data.get("collapsed", 0))
        except (TypeError, ValueError):
            self.observations, self.collapsed = 0, 0

    def status(self) -> dict:
        return {
            "pairs": len(self.pairs),
            "states": len(self.states()),
            "actions": len(self.by_action),
            "observations": self.observations,
            "collapsed": self.collapsed,
            "smoothing": self.smoothing,
            "min_n": self.min_n,
            "half_life": self.half_life,
        }
