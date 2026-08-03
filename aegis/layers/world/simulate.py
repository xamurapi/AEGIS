"""Looking ahead: deterministic rollouts over the learned model (spec M1.5).

This is the step that turns "model of the world" into "decision": the planner
asks what a sequence of actions is likely to be worth *before* committing to
the first one.

Three choices that define the search:

**Expectation, not sampling.** Monte Carlo would need a random number generator
and would give a different answer on every run, which is exactly what the
zero-randomness guarantee (§3.1) forbids and what would make an A/B comparison
meaningless. Instead each node expands its top-``branch`` successors weighted
by probability. The result is the same on every run, by construction.

**Beam, not exhaustive.** Full expansion is |A|^depth; at the configured depth
of 3 over thirty-odd actions that is tens of thousands of nodes inside a 30 ms
budget. The beam keeps the best few partial sequences and drops the rest.

**Memoised by (state, depth).** The same successor state reappears constantly
across branches, and its value does not depend on how the search arrived there.

The node value is

    V(s, d) = E[r|s,a] − λ·cost − ρ·risk + γ·Σ P(s'|s,a)·V(s', d−1)

with risk taken from reward variance: an action that averages well but varies
wildly is not the same proposition as one that reliably returns the average.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.layers.world.state import StateKey

logger = logging.getLogger("aegis.world.simulate")

#: Weight of expected cost in a node's value.
COST_WEIGHT = 0.2
#: Weight of reward spread (risk) in a node's value.
RISK_WEIGHT = 0.3


@dataclass
class RolloutResult:
    """What a look-ahead concluded."""

    sequence: list[str]
    value: float
    steps: list[dict] = field(default_factory=list)
    nodes_expanded: int = 0
    memo_hits: int = 0
    elapsed_ms: float = 0.0
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "sequence": list(self.sequence),
            "value": round(self.value, 5),
            "steps": self.steps,
            "nodes_expanded": self.nodes_expanded,
            "memo_hits": self.memo_hits,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "truncated": self.truncated,
        }


class Simulator:
    """Deterministic look-ahead over the transition and outcome models."""

    def __init__(self, transitions, outcomes, *, branch: int | None = None,
                 discount: float | None = None, explore_bonus: float | None = None,
                 max_nodes: int = 20000):
        self.transitions = transitions
        self.outcomes = outcomes
        self.branch = int(cfg.WM_BRANCH if branch is None else branch)
        self.discount = float(cfg.WM_DISCOUNT if discount is None else discount)
        self.explore_bonus = float(cfg.WM_EXPLORE_BONUS
                                   if explore_bonus is None else explore_bonus)
        #: Hard ceiling on expansion, so a pathological branching factor cannot
        #: turn a 30 ms budget into a stall.
        self.max_nodes = int(max_nodes)
        self.rollouts = 0
        self.last_elapsed_ms = 0.0

    # ── one step's worth ─────────────────────────────────────────────

    def immediate_value(self, state_key: str, action: str) -> float:
        """What one action is worth here, before anything that follows it.

        Uses the pessimistic success estimate: choosing is exactly the moment to
        distrust a thin sample, and an action that succeeded once out of once
        must not outrank one with a long solid record.
        """
        outcome = self.outcomes.predict(state_key, action)
        risk = outcome.reward_sd * (1.0 - outcome.p_success_pessimistic)
        curiosity = self.explore_bonus * (1.0 - outcome.known)
        return (outcome.expected_reward * outcome.p_success_pessimistic
                - COST_WEIGHT * outcome.expected_cost
                - RISK_WEIGHT * risk
                + curiosity)

    # ── the search ───────────────────────────────────────────────────

    def rollout(self, state, actions: list[str], depth: int | None = None,
                beam: int | None = None, discount: float | None = None
                ) -> RolloutResult:
        """Best action sequence from ``state``, and what it is worth."""
        started = CLOCK.monotonic()
        depth = int(cfg.WM_DEPTH if depth is None else depth)
        beam = max(1, int(cfg.WM_BEAM if beam is None else beam))
        gamma = float(self.discount if discount is None else discount)
        options = sorted({str(a) for a in (actions or [])})

        if not options or depth <= 0:
            # Nothing to search. Timed through the same path as a real search
            # so there is one place that converts an interval to milliseconds
            # rather than two that could drift apart.
            return self._finish(RolloutResult(sequence=[], value=0.0), started)

        state_key = state.key() if isinstance(state, StateKey) else str(state)
        memo: dict[tuple[str, int], float] = {}
        stats = {"nodes": 0, "hits": 0, "truncated": False}

        def value_of(node: str, remaining: int) -> float:
            if remaining <= 0:
                return 0.0
            cached = memo.get((node, remaining))
            if cached is not None:
                stats["hits"] += 1
                return cached
            if stats["nodes"] >= self.max_nodes:
                stats["truncated"] = True
                return 0.0
            stats["nodes"] += 1

            # The beam, applied where it means something: below the first
            # level, only the `beam` most promising actions at this node are
            # expanded. This is what makes the docstring's "beam, not
            # exhaustive" true — and what gives `plan_beam` the behavioural
            # effect the genome contract requires. The harness charges fitness
            # for a wide beam (harness.cost_of), and a parameter that costs
            # fitness while changing nothing would simply be selected to its
            # minimum, a decorative gene bought and never used.
            best = max(self._action_value(node, action, remaining, gamma, value_of)
                       for action in self._beam_actions(node, options, beam))
            memo[(node, remaining)] = best
            return best

        # The first level is expanded explicitly — and in full, not through the
        # beam — so the chosen action, and the per-step breakdown a human
        # reads, come out of the same computation that produced the number, and
        # so every offered action gets priced before one is picked. The beam
        # prunes the lookahead beneath a choice, never the choice itself.
        scored = [(self._action_value(state_key, action, depth, gamma, value_of), action)
                  for action in options]
        scored.sort(key=lambda row: (-row[0], row[1]))
        best_value, best_action = scored[0]

        sequence, steps = self._trace(state_key, best_action, depth, gamma,
                                      options, value_of)
        self.rollouts += 1
        return self._finish(RolloutResult(
            sequence=sequence, value=best_value, steps=steps,
            nodes_expanded=stats["nodes"], memo_hits=stats["hits"],
            truncated=stats["truncated"],
        ), started)

    def _finish(self, result: RolloutResult, started: float) -> RolloutResult:
        """Stamp a result with how long the search took."""
        result.elapsed_ms = (CLOCK.monotonic() - started) * 1000
        self.last_elapsed_ms = result.elapsed_ms
        return result

    def _beam_actions(self, node: str, options: list[str], beam: int) -> list[str]:
        """The ``beam`` most promising actions at this node.

        Ranked by immediate value: the search cannot know an action's future
        worth without expanding it, and expanding everything to decide what
        not to expand would be the exhaustive search the beam exists to avoid.
        The price of that heuristic is real — an action that pays nothing now
        and everything later can fall off a narrow beam — which is exactly why
        the width is a gene worth evolving rather than a constant. Ties break
        on the action name, so the pruning is deterministic (§3.1).
        """
        if beam >= len(options):
            return options
        ranked = sorted(options,
                        key=lambda action: (-self.immediate_value(node, action),
                                            action))
        return ranked[:beam]

    def _action_value(self, node: str, action: str, remaining: int,
                      gamma: float, value_of) -> float:
        """V(s, a, d) — this action's immediate worth plus its discounted future."""
        total = self.immediate_value(node, action)
        if remaining <= 1:
            return total
        successors = self.transitions.top_next(node, action, self.branch)
        if not successors:
            return total
        # Renormalise over the expanded branch: the top-k probabilities do not
        # sum to one, and treating the missing mass as worth zero would make
        # every deep plan look worse than a shallow one for no reason.
        mass = sum(p for _, p in successors)
        if mass <= 0:
            return total
        future = sum((p / mass) * value_of(successor, remaining - 1)
                     for successor, p in successors)
        return total + gamma * future

    def _trace(self, state_key: str, first_action: str, depth: int, gamma: float,
               options: list[str], value_of) -> tuple[list[str], list[dict]]:
        """Follow the greedy path forward, for the plan and its explanation."""
        sequence, steps = [], []
        node, action, remaining = state_key, first_action, depth
        # Depth is the only thing that ends this walk: every iteration picks a
        # concrete next action, so a second "did we get an action" condition
        # would be a clause that can never be false.
        while remaining > 0:
            outcome = self.outcomes.predict(node, action)
            successors = self.transitions.top_next(node, action, self.branch)
            sequence.append(action)
            steps.append({
                "state": node, "action": action,
                "p_success": round(outcome.p_success, 4),
                "expected_reward": round(outcome.expected_reward, 4),
                "reward_sd": round(outcome.reward_sd, 4),
                "known": round(outcome.known, 3),
                "next": [[k, round(p, 4)] for k, p in successors[:self.branch]],
            })
            remaining -= 1
            if remaining <= 0 or not successors:
                break
            node = successors[0][0]
            scored = [(self._action_value(node, candidate, remaining, gamma, value_of),
                       candidate) for candidate in options]
            scored.sort(key=lambda row: (-row[0], row[1]))
            action = scored[0][1]
        return sequence, steps

    def best_sequence(self, state, actions: list[str], depth: int | None = None,
                      beam: int | None = None,
                      discount: float | None = None) -> list[str]:
        """Just the sequence — the common case."""
        return self.rollout(state, actions, depth, beam, discount).sequence

    def evaluate(self, state, sequence: list[str], discount: float | None = None
                 ) -> float:
        """What one specific sequence is worth, without searching.

        Used to compare a proposed plan against the planner's own — including
        one suggested by the cortex, which must be priced by the same yardstick
        rather than trusted.
        """
        gamma = float(self.discount if discount is None else discount)
        node = state.key() if isinstance(state, StateKey) else str(state)
        total, weight = 0.0, 1.0
        for action in (sequence or []):
            total += weight * self.immediate_value(node, action)
            successors = self.transitions.top_next(node, action, 1)
            if not successors:
                break
            node = successors[0][0]
            weight *= gamma
        return total

    def status(self) -> dict:
        return {
            "rollouts": self.rollouts,
            "branch": self.branch,
            "discount": self.discount,
            "explore_bonus": self.explore_bonus,
            "last_rollout_ms": round(self.last_elapsed_ms, 3),
            "max_nodes": self.max_nodes,
        }
