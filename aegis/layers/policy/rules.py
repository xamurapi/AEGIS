"""Behaviour rules: the explicit, evidential half of the policy (spec M3.4-M3.5).

A preference weight (M3.3) says "this tends to pay here". A rule says something
much stronger — "do not do this here" — and strength like that has to be earned.
So every rule carries its evidence, its confidence interval, the experiences
that produced it, and a measured effect from a controlled trial; and every rule
can be taken away again when the world stops agreeing with it.

The two failure modes this design is built against:

**Rules mined from noise.** Enumerating every feature subset against every
action is hundreds of comparisons per generation, and at α = 0.05 one in twenty
pure-noise combinations clears an uncorrected threshold. A system that mines
often enough would therefore *always* find rules, and its policy would fill with
descriptions of nothing. Benjamini–Hochberg over the whole generation is what
makes "significant" mean something here.

**Rules that paralyse the system.** A rule may lower a plan's score or suppress
it outright, and enough suppression is indistinguishable from a system that has
stopped working. Three limits: safety-critical actions can never be suppressed,
a rule cannot activate without a controlled trial showing it actually helped,
and an active rule that loses its significance is retired automatically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import aegis.config as cfg
from aegis.layers.world.state import FIELDS, StateKey
from aegis.util.stats import (
    benjamini_hochberg, compare_samples, two_proportion_z, wilson_interval,
)

logger = logging.getLogger("aegis.policy")

#: What a rule does to a plan that matches it.
SUPPRESS = "suppress"          # remove it from consideration entirely
PREFER = "prefer"              # push it up the ranking
EFFECTS = (SUPPRESS, PREFER)

#: Lifecycle states (§M3.5).
CANDIDATE = "candidate"        # mined, not yet trusted
TRIAL = "trial"                # under controlled comparison
ACTIVE = "active"              # applied to real decisions
RETIRED = "retired"            # lost its significance
REFUTED = "refuted"            # the trial showed the opposite; never re-mined

#: How much a `prefer` rule adds to a plan's score. Deliberately smaller than
#: the suppression it mirrors: promoting something on evidence is a weaker claim
#: than forbidding it, and the planner's own terms should still dominate.
PREFER_BONUS = 0.25


@dataclass
class Rule:
    """One learned constraint on behaviour, with its evidence attached."""

    id: str
    condition: dict                 # {"state": {"error": "high"}, "action": "x"}
    effect: str
    support: int = 0
    success_rate: float = 0.0
    wilson_low: float = 0.0
    wilson_high: float = 1.0
    base_rate: float = 0.0
    p_value: float = 1.0
    status: str = CANDIDATE
    created_tick: int = 0
    activated_tick: int | None = None
    reviewed_tick: int = 0
    measured_effect: float | None = None
    provenance: list[str] = field(default_factory=list)
    trial_started: int | None = None
    #: How many times a retired rule has re-entered trial. Part of the record
    #: an operator reads: a rule on its third re-trial is a claim the world
    #: keeps half-agreeing with, which is worth seeing as such.
    retrials: int = 0

    # ── matching ─────────────────────────────────────────────────────

    @property
    def action(self) -> str:
        return str(self.condition.get("action", ""))

    @property
    def state_condition(self) -> dict:
        got = self.condition.get("state")
        return dict(got) if isinstance(got, dict) else {}

    def matches_state(self, state) -> bool:
        """Whether this rule's state condition holds.

        A condition names a subset of state fields; every named field must
        agree. An empty condition matches everything, which is what makes an
        action-only rule ("never do this at all") expressible.
        """
        if state is None:
            return False
        values = state.as_dict() if hasattr(state, "as_dict") else \
            StateKey.parse(str(state)).as_dict()
        return all(values.get(field_name) == expected
                   for field_name, expected in self.state_condition.items())

    def matches(self, state, action: str) -> bool:
        return action == self.action and self.matches_state(state)

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "condition": {"state": dict(sorted(self.state_condition.items())),
                          "action": self.action},
            "effect": self.effect,
            "support": self.support,
            "success_rate": round(self.success_rate, 4),
            "wilson_low": round(self.wilson_low, 4),
            "wilson_high": round(self.wilson_high, 4),
            "base_rate": round(self.base_rate, 4),
            "p_value": round(self.p_value, 6),
            "status": self.status,
            "created_tick": self.created_tick,
            "activated_tick": self.activated_tick,
            "reviewed_tick": self.reviewed_tick,
            "measured_effect": (round(self.measured_effect, 4)
                                if self.measured_effect is not None else None),
            "provenance": list(self.provenance[:10]),
            "trial_started": self.trial_started,
            "retrials": self.retrials,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Rule | None:
        """Rebuild from disk, refusing anything malformed.

        Returning None rather than raising: one corrupt row must cost its own
        rule, not the whole rule base — and a rule that cannot be read cannot be
        allowed to suppress anything either.
        """
        if not isinstance(data, dict):
            return None
        condition = data.get("condition")
        if not isinstance(condition, dict) or not condition.get("action"):
            return None
        effect = str(data.get("effect", ""))
        if effect not in EFFECTS:
            return None
        try:
            return cls(
                id=str(data["id"]),
                condition={"state": dict(condition.get("state") or {}),
                           "action": str(condition["action"])},
                effect=effect,
                support=max(0, int(data.get("support", 0))),
                success_rate=float(data.get("success_rate", 0.0)),
                wilson_low=float(data.get("wilson_low", 0.0)),
                wilson_high=float(data.get("wilson_high", 1.0)),
                base_rate=float(data.get("base_rate", 0.0)),
                p_value=float(data.get("p_value", 1.0)),
                status=str(data.get("status", CANDIDATE)),
                created_tick=int(data.get("created_tick", 0)),
                activated_tick=(int(data["activated_tick"])
                                if data.get("activated_tick") is not None else None),
                reviewed_tick=int(data.get("reviewed_tick", 0)),
                measured_effect=(float(data["measured_effect"])
                                 if data.get("measured_effect") is not None else None),
                provenance=[str(p) for p in (data.get("provenance") or [])][:10],
                trial_started=(int(data["trial_started"])
                               if data.get("trial_started") is not None else None),
                retrials=max(0, int(data.get("retrials", 0))),
            )
        except (KeyError, TypeError, ValueError):
            logger.debug("Discarding unreadable rule row", exc_info=True)
            return None


def rule_id(condition: dict, effect: str) -> str:
    """A stable identity for a rule, so the same finding is the same rule.

    Derived from the condition rather than from a counter: a rule re-mined next
    generation must be recognised as the one already on trial, or the miner
    would keep proposing what it has already proposed and the refuted archive
    would never match anything.
    """
    state = condition.get("state") or {}
    parts = [f"{name}={state[name]}" for name in sorted(state)]
    parts.append(f"do={condition.get('action', '')}")
    return f"{effect}:" + "&".join(parts)


class RuleMiner:
    """Finds candidate rules in the experience log. Deterministic."""

    def __init__(self, min_support: int | None = None,
                 max_condition_size: int | None = None,
                 alpha: float | None = None):
        self.min_support = int(
            cfg.POLICY_MIN_SUPPORT if min_support is None else min_support)
        self.max_condition_size = int(
            cfg.POLICY_MAX_COND if max_condition_size is None else max_condition_size)
        self.alpha = float(cfg.POLICY_ALPHA if alpha is None else alpha)
        self.generations = 0
        self.tested = 0
        self.proposed = 0

    def mine(self, experiences, tick: int = 0) -> list[Rule]:
        """One generation of candidates, in a fixed order.

        Every combination is tested, the whole family is corrected together, and
        only then are candidates emitted. Correcting per-combination — or
        stopping early at the first find — would break the false-discovery
        control that makes the output trustworthy.
        """
        rows = [row for row in (experiences or []) if self._usable(row)]
        self.generations += 1
        if len(rows) < self.min_support:
            return []

        by_action = self._group_by_action(rows)
        candidates = []
        for action in sorted(by_action):
            action_rows = by_action[action]
            base_successes = sum(1 for row in action_rows if row["success"])
            base_rate = base_successes / len(action_rows)
            base_outcomes = [1.0 if row["success"] else 0.0 for row in action_rows]

            for condition_fields in self._condition_shapes():
                for values, matched in sorted(
                        self._partition(action_rows, condition_fields).items()):
                    if len(matched) < self.min_support:
                        continue
                    # The comparison is against the *rest* of this action's
                    # history, not against its overall rate: a subset compared
                    # with a pool that contains it is compared with itself.
                    rest = [row for row in action_rows if row not in matched]
                    if len(rest) < 2:
                        continue
                    candidates.append(self._assess(
                        action, condition_fields, values, matched, rest,
                        base_rate, base_outcomes, tick))

        self.tested += len(candidates)
        return self._select(candidates)

    # ── the pieces ───────────────────────────────────────────────────

    @staticmethod
    def _usable(row) -> bool:
        return (isinstance(row, dict) and row.get("action")
                and row.get("state") is not None and "success" in row)

    @staticmethod
    def _group_by_action(rows) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row["action"]), []).append(row)
        return grouped

    def _condition_shapes(self):
        """Which field combinations to test, smallest first.

        Smallest first because a one-field rule that explains the data is
        preferable to a two-field one that explains it equally well — and
        because the count grows fast: seven fields give 7 singles and 21 pairs
        per action.
        """
        size = max(1, min(self.max_condition_size, len(FIELDS)))
        for width in range(1, size + 1):
            yield from combinations(FIELDS, width)

    @staticmethod
    def _partition(rows, condition_fields) -> dict[tuple, list[dict]]:
        groups: dict[tuple, list[dict]] = {}
        for row in rows:
            values = StateKey.parse(str(row["state"])).as_dict()
            key = tuple(values.get(name, "") for name in condition_fields)
            groups.setdefault(key, []).append(row)
        return groups

    def _assess(self, action, condition_fields, values, matched, rest,
                base_rate, base_outcomes, tick) -> Rule:
        successes = sum(1 for row in matched if row["success"])
        rate = successes / len(matched)
        low, high = wilson_interval(successes, len(matched))
        # Proportions, not measurements: a t-test on the 0/1 indicators divides
        # by a zero standard error when a cell is all-fail against an
        # all-succeed rest, and would report the cleanest possible signal as no
        # evidence at all.
        rest_successes = sum(1 for row in rest if row["success"])
        comparison = two_proportion_z(successes, len(matched),
                                      rest_successes, len(rest))

        condition = {"state": {name: values[index]
                               for index, name in enumerate(condition_fields)},
                     "action": action}
        effect = SUPPRESS if rate < base_rate else PREFER
        return Rule(
            id=rule_id(condition, effect),
            condition=condition,
            effect=effect,
            support=len(matched),
            success_rate=rate,
            wilson_low=low,
            wilson_high=high,
            base_rate=base_rate,
            p_value=comparison.p_value,
            created_tick=tick,
            provenance=[str(row.get("id", "")) for row in matched[:10] if row.get("id")],
        )

    def _select(self, candidates: list[Rule]) -> list[Rule]:
        """Keep only what survives false-discovery control and is worth acting on."""
        if not candidates:
            return []
        survives = benjamini_hochberg([rule.p_value for rule in candidates],
                                      self.alpha)
        kept = []
        for rule, passed in zip(candidates, survives):
            if not passed:
                continue
            # A significant difference is not yet a useful one. The interval has
            # to sit clear of the base rate, or the rule is describing a
            # detectable but negligible wobble.
            if rule.effect == SUPPRESS and rule.wilson_high >= rule.base_rate:
                continue
            if rule.effect == PREFER and rule.wilson_low <= rule.base_rate:
                continue
            kept.append(rule)
        kept = self._prune_specialisations(kept)
        kept.sort(key=lambda rule: (rule.p_value, rule.id))
        self.proposed += len(kept)
        return kept

    @staticmethod
    def _prune_specialisations(rules: list[Rule]) -> list[Rule]:
        """Drop a rule when a simpler one already says the same thing.

        Enumerating subsets means one fact is found many times over: "fails when
        energy is low" also surfaces as "fails when energy is low and the error
        rate is unknown", and so on for every other field. All of them are true
        and all but the first are noise — they would each get their own trial,
        each consume its own share of the budget, and clutter the rule base with
        a dozen restatements of one finding.

        A rule survives only if no already-kept rule for the same action and
        effect has a condition that is a *subset* of its own. Two conditions
        that merely coincide on this data (low energy and a tired mood, when
        those always occur together) both survive: the evidence genuinely
        cannot tell them apart, and picking one arbitrarily would be inventing a
        distinction the data does not support.
        """
        # Simplest first, then most significant — so the general form is the one
        # that gets to survive.
        ordered = sorted(rules, key=lambda rule: (len(rule.state_condition),
                                                  rule.p_value, rule.id))
        kept: list[Rule] = []
        for rule in ordered:
            condition = set(rule.state_condition.items())
            if any(other.action == rule.action and other.effect == rule.effect
                   and set(other.state_condition.items()) <= condition
                   for other in kept):
                continue
            kept.append(rule)
        return kept

    def status(self) -> dict:
        return {
            "generations": self.generations,
            "tested": self.tested,
            "proposed": self.proposed,
            "min_support": self.min_support,
            "max_condition_size": self.max_condition_size,
            "alpha": self.alpha,
        }


class RuleLifecycle:
    """Owns the rule base: what is on trial, what is active, what is refuted."""

    def __init__(self, store_path: Path | None = None,
                 max_rules: int | None = None):
        self._store_path = store_path or (cfg.POLICY_DIR / "rules.json")
        self.max_rules = int(cfg.POLICY_MAX_RULES if max_rules is None else max_rules)
        self.rules: dict[str, Rule] = {}
        #: Refuted rules are remembered forever — an opposite result is
        #: knowledge, and without this the miner would re-propose the same
        #: disproved rule every generation.
        self.refuted: dict[str, dict] = {}
        self.activations = 0
        self.retirements = 0
        self.refutations = 0
        self.retrials = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        from aegis.store.migrations import read_store

        data = read_store(self._store_path, store="policy_rules")
        for row in (data.get("rules") or []):
            rule = Rule.from_dict(row)
            if rule is not None:
                self.rules[rule.id] = rule
        for row in (data.get("refuted") or []):
            if isinstance(row, dict) and row.get("id"):
                self.refuted[str(row["id"])] = dict(row)
        for name in ("activations", "retirements", "refutations", "retrials"):
            try:
                setattr(self, name, max(0, int(data.get(name, 0))))
            except (TypeError, ValueError):
                setattr(self, name, 0)

    def save(self) -> None:
        from aegis.store.migrations import write_store

        write_store(self._store_path, {
            "rules": [rule.to_dict() for rule in self.ordered()],
            "refuted": [self.refuted[key] for key in sorted(self.refuted)],
            "activations": self.activations,
            "retirements": self.retirements,
            "refutations": self.refutations,
            "retrials": self.retrials,
        })

    # ── admission ────────────────────────────────────────────────────

    def admit(self, candidates, tick: int, safety_critical=()) -> list[Rule]:
        """Take in a generation of candidates. Returns the ones newly accepted."""
        protected = set(safety_critical or ())
        accepted = []
        for candidate in candidates:
            if candidate.effect == SUPPRESS and candidate.action in protected:
                # §M3.5: health checks, checkpoints and perception are not
                # negotiable, however convincing the evidence looks.
                logger.debug("Refusing a suppression of safety-critical %s",
                             candidate.action)
                continue
            if candidate.id in self.refuted:
                continue                       # already disproved; do not re-open
            existing = self.rules.get(candidate.id)
            if existing is not None:
                # Refresh the evidence but not a live status: a rule already on
                # trial keeps its trial, or a re-mine would restart the clock
                # forever and nothing would ever activate.
                existing.support = candidate.support
                existing.success_rate = candidate.success_rate
                existing.wilson_low = candidate.wilson_low
                existing.wilson_high = candidate.wilson_high
                existing.base_rate = candidate.base_rate
                existing.p_value = candidate.p_value
                # Retirement, though, must be reversible (the module docstring
                # promises a two-way lifecycle). The miner re-finds a retired
                # rule every generation; with no path back to trial, one
                # inconclusive review made retirement permanent however strong
                # the evidence became afterwards. The bar for re-entry is
                # deliberately *stricter* than a first admission — half the
                # mining alpha — so a rule cannot thrash between retired and
                # trial on the same borderline evidence that retired it.
                if existing.status == RETIRED \
                        and candidate.p_value <= self.retrial_alpha():
                    existing.status = TRIAL
                    existing.trial_started = tick
                    existing.activated_tick = None
                    existing.measured_effect = None
                    existing.retrials += 1
                    self.retrials += 1
                    logger.info("Retired rule %s re-entered trial on fresh "
                                "evidence (p=%.6f)", existing.id,
                                candidate.p_value)
                    accepted.append(existing)
                continue
            candidate.status = TRIAL
            candidate.trial_started = tick
            self.rules[candidate.id] = candidate
            accepted.append(candidate)
        self._evict_if_needed()
        return accepted

    @staticmethod
    def retrial_alpha() -> float:
        """The significance a retired rule must clear to re-enter trial.

        Half the mining alpha: the miner's own bar admitted the rule once and
        the world then failed to confirm it, so the second admission has to be
        on evidence that would have survived a harsher correction — not merely
        the same borderline signal found again.
        """
        return float(cfg.POLICY_ALPHA) / 2.0

    # ── the trial verdict ────────────────────────────────────────────

    def conclude_trial(self, rule: Rule, applied, withheld, tick: int,
                       alpha: float | None = None,
                       min_effect: float | None = None) -> str:
        """Decide a rule's fate from its controlled comparison.

        ``applied`` and ``withheld`` are the realised rewards on the ticks where
        the rule was eligible, split by whether it was allowed to act. A rule
        earns activation only by having *helped*; a rule that measurably hurt is
        refuted and archived rather than merely dropped, so the miner cannot
        rediscover it next generation.
        """
        alpha = float(cfg.POLICY_ALPHA if alpha is None else alpha)
        min_effect = float(
            cfg.POLICY_MIN_EFFECT if min_effect is None else min_effect)

        comparison = compare_samples(applied, withheld)
        rule.measured_effect = comparison.effect
        rule.reviewed_tick = tick

        if comparison.significant(alpha) and comparison.effect >= min_effect:
            rule.status = ACTIVE
            rule.activated_tick = tick
            self.activations += 1
            return ACTIVE
        if comparison.significant(alpha) and comparison.effect <= -min_effect:
            self._refute(rule, tick, "the trial measured the opposite effect")
            return REFUTED
        rule.status = RETIRED
        self.retirements += 1
        return RETIRED

    def review(self, rule: Rule, applied, withheld, tick: int,
               alpha: float | None = None,
               min_effect: float | None = None) -> str:
        """Re-judge an active rule (§M3.5).

        Losing significance retires a rule rather than refuting it: the world
        may simply have changed, and a rule that stopped applying is not the
        same as one that was wrong. Only a measured *reverse* effect refutes.
        """
        alpha = float(cfg.POLICY_ALPHA if alpha is None else alpha)
        min_effect = float(
            cfg.POLICY_MIN_EFFECT if min_effect is None else min_effect)

        comparison = compare_samples(applied, withheld)
        rule.reviewed_tick = tick
        if comparison.significant(alpha) and comparison.effect <= -min_effect:
            self._refute(rule, tick, "an active rule reversed under review")
            return REFUTED
        if not comparison.significant(alpha) or comparison.effect < min_effect:
            rule.status = RETIRED
            rule.measured_effect = comparison.effect
            self.retirements += 1
            return RETIRED
        rule.measured_effect = comparison.effect
        return ACTIVE

    def _refute(self, rule: Rule, tick: int, reason: str) -> None:
        rule.status = REFUTED
        rule.reviewed_tick = tick
        self.refutations += 1
        self.refuted[rule.id] = {
            "id": rule.id, "condition": rule.condition, "effect": rule.effect,
            "refuted_tick": tick, "reason": reason,
            "measured_effect": rule.measured_effect,
        }
        self.rules.pop(rule.id, None)

    # ── reading ──────────────────────────────────────────────────────

    def ordered(self) -> list[Rule]:
        return [self.rules[key] for key in sorted(self.rules)]

    def by_status(self, status: str) -> list[Rule]:
        return [rule for rule in self.ordered() if rule.status == status]

    def active(self) -> list[Rule]:
        return self.by_status(ACTIVE)

    def on_trial(self) -> list[Rule]:
        return self.by_status(TRIAL)

    def _evict_if_needed(self) -> None:
        """Bound the rule base, dropping the least-evidenced non-active rules."""
        if len(self.rules) <= self.max_rules:
            return
        droppable = [rule for rule in self.ordered() if rule.status != ACTIVE]
        droppable.sort(key=lambda rule: (rule.support, -rule.p_value, rule.id))
        for rule in droppable[:len(self.rules) - self.max_rules]:
            self.rules.pop(rule.id, None)

    def status(self) -> dict:
        return {
            "total": len(self.rules),
            "active": len(self.by_status(ACTIVE)),
            "trial": len(self.by_status(TRIAL)),
            "retired": len(self.by_status(RETIRED)),
            "refuted_archive": len(self.refuted),
            "activations": self.activations,
            "retirements": self.retirements,
            "refutations": self.refutations,
            "retrials": self.retrials,
            "rules": [rule.to_dict() for rule in self.ordered()[:20]],
        }
