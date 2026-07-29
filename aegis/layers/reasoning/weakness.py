"""Where the system is bad, stated precisely enough to act on (spec M6.6).

"Reasoning is weak" is not something anything can act on. A weakness here is a
*combination of task features* — ``family=arithmetic_chain`` and ``incomplete``,
say — together with how often it fails, how much evidence that rests on, and how
much worse it is than the system's own average. The synthesiser takes the
combination; the arena takes the held-out half of it; an operator reads the
whole row.

Three things keep it from inventing weaknesses.

**A base rate.** A group is weak only relative to what the system does
everywhere else. A system that fails a third of everything has no weakness at
33% — it has a level.

**A significance test with false-discovery control.** Every feature and every
pair of features is a hypothesis, and there are hundreds of them; at α = 0.05
roughly one in twenty of the useless ones would look real. Benjamini–Hochberg
over the whole family of tests is what makes "significant" mean something when
the tests are counted.

**Specialisation pruning.** If ``family=arithmetic_chain`` fails and
``family=arithmetic_chain AND numeric`` fails identically, the second is the
first wearing a longer name. Reporting both would send the synthesiser after the
same problem twice and split the evidence for it in half.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import aegis.config as cfg
from aegis.util.stats import benjamini_hochberg, two_proportion_z, wilson_interval

logger = logging.getLogger("aegis.reasoning")

#: Below this many attempts a group is not weak, it is unmeasured. Two failures
#: out of two is a story, not a finding.
MIN_SUPPORT = 12

#: How many features may be combined. Two is where the useful specificity is:
#: one feature is usually too coarse to synthesise against, and three multiplies
#: the number of hypotheses by an order of magnitude for combinations that no
#: longer have the support to reach significance anyway.
MAX_COMBO = 2

#: How many failing examples a weakness carries, for the synthesiser's prompt
#: and for the operator.
EXAMPLES = 5


@dataclass(frozen=True)
class Weakness:
    """One combination of features that fails more than the system's average."""

    combo: tuple[str, ...]
    fail_rate: float
    base_rate: float
    support: int
    fails: int
    lower: float
    excess: float
    p_value: float
    rank: float
    family: str = ""
    examples: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return " AND ".join(self.combo)

    def as_dict(self) -> dict:
        return {"combo": list(self.combo), "label": self.label,
                "family": self.family, "fail_rate": round(self.fail_rate, 4),
                "base_rate": round(self.base_rate, 4), "support": self.support,
                "fails": self.fails, "lower": round(self.lower, 4),
                "excess": round(self.excess, 4),
                "p_value": round(self.p_value, 6), "rank": round(self.rank, 4),
                "examples": list(self.examples)}


def labels_of(row: dict) -> tuple[str, ...]:
    """The axes one attempt sits on.

    Rendered as strings rather than kept as a mapping because a weakness is a
    *set* of labels: that makes combining them, deduplicating them and printing
    them the same operation, and it is what the task generator already produces
    for the feature counter.

    Returned sorted, and that is load-bearing rather than tidy. ``combinations``
    preserves the order it is given, so the order here *is* the order of every
    group key. ``_prune`` looks a parent up by its sorted tuple; if the two
    orders can differ, the lookup misses, no parent is ever found, and the
    pruning that stops one problem being reported twice silently does nothing.
    Sorting here also satisfies §3.1: nothing whose order reaches a result may
    depend on insertion order.
    """
    labels: list[str] = []
    family = str(row.get("family", "") or "")
    if family:
        labels.append(f"family={family}")
    for name, value in sorted((row.get("features") or {}).items()):
        if isinstance(value, bool):
            if value:
                labels.append(name)
        elif isinstance(value, (int, float, str)):
            labels.append(f"{name}={value}")
        elif isinstance(value, (list, tuple)):
            labels.extend(f"op:{item}" for item in value)
    state = str(row.get("state", "") or "")
    if state:
        # Grouping by system state as well as by task shape (M6.6): the same
        # problem attempted while out of energy is a different situation, and a
        # weakness that only appears there is a real one.
        labels.append(f"state={state}")
    return tuple(sorted(dict.fromkeys(labels)))


@dataclass
class _Group:
    support: int = 0
    fails: int = 0
    examples: list[str] = field(default_factory=list)
    families: set[str] = field(default_factory=set)


class WeaknessDetector:
    """Scans attempt records and reports where the system is losing."""

    def __init__(self, *, alpha: float | None = None,
                 min_support: int = MIN_SUPPORT, max_combo: int = MAX_COMBO):
        self.alpha = float(cfg.DISC_ALPHA if alpha is None else alpha)
        self.min_support = int(min_support)
        self.max_combo = max(1, int(max_combo))
        self.scans = 0
        self.tested = 0
        self.last: list[Weakness] = []

    def scan(self, rows, window: int | None = None) -> list[Weakness]:
        """Report the weaknesses in the most recent ``window`` attempts.

        Sources are folded in by the caller — reasoning attempts, environment
        misses, failed experiences, prediction errors — because each knows how
        to describe its own outcome and none of them should have to know about
        this. What arrives here is a uniform row: features, state, and whether
        it worked.
        """
        rows = list(rows or [])
        if window:
            rows = rows[-int(window):]
        self.scans += 1
        if len(rows) < self.min_support:
            self.last = []
            return []

        total = len(rows)
        total_fails = sum(1 for row in rows if not row.get("solved"))
        base_rate = total_fails / total
        if total_fails == 0:
            self.last = []
            return []                   # nothing failed; nothing to explain

        groups = self._group(rows)
        candidates, p_values = [], []
        for combo, group in sorted(groups.items()):
            if group.support < self.min_support or group.support >= total:
                continue
            fail_rate = group.fails / group.support
            if fail_rate <= base_rate:
                continue
            # Against the *rest* of the data rather than against everything:
            # a large group is part of its own base rate, and comparing it with
            # a total it dominates hides exactly the biggest weaknesses.
            rest_support = total - group.support
            rest_fails = total_fails - group.fails
            if rest_support <= 0:
                continue
            p_value = two_proportion_z(group.fails, group.support,
                                       rest_fails, rest_support).p_value
            rest_rate = rest_fails / rest_support
            candidates.append((combo, group, fail_rate, rest_rate, p_value))
            p_values.append(p_value)

        self.tested += len(candidates)
        if not candidates:
            self.last = []
            return []

        keep = benjamini_hochberg(p_values, self.alpha)
        found = []
        for (combo, group, fail_rate, rest_rate, p_value), significant in zip(
                candidates, keep):
            if not significant:
                continue
            excess = fail_rate - rest_rate
            lower, _ = wilson_interval(group.fails, group.support)
            found.append(Weakness(
                combo=combo, fail_rate=fail_rate, base_rate=rest_rate,
                support=group.support, fails=group.fails, lower=lower,
                excess=excess, p_value=p_value,
                rank=group.support * excess,
                family=sorted(group.families)[0] if len(group.families) == 1 else "",
                examples=tuple(group.examples[:EXAMPLES])))

        found = self._prune(found)
        found.sort(key=lambda weakness: (-weakness.rank, weakness.label))
        self.last = found
        return found

    # ── internals ────────────────────────────────────────────────────

    def _group(self, rows) -> dict[tuple[str, ...], _Group]:
        groups: dict[tuple[str, ...], _Group] = {}
        for row in rows:
            labels = labels_of(row)
            solved = bool(row.get("solved"))
            for size in range(1, self.max_combo + 1):
                for combo in combinations(labels, size):
                    group = groups.setdefault(combo, _Group())
                    group.support += 1
                    if not solved:
                        group.fails += 1
                        if len(group.examples) < EXAMPLES:
                            group.examples.append(str(row.get("task", "")))
                    family = str(row.get("family", "") or "")
                    if family:
                        group.families.add(family)
        return groups

    def _prune(self, found: list[Weakness]) -> list[Weakness]:
        """Drop a combination that a shorter one already explains.

        Kept only if it fails *more* than every combination it contains. A
        specialisation that merely repeats its parent would send the synthesiser
        after one problem twice and halve the evidence behind each attempt.
        """
        by_combo = {weakness.combo: weakness for weakness in found}
        kept = []
        for weakness in found:
            general = [by_combo[tuple(sorted(smaller))]
                       for size in range(1, len(weakness.combo))
                       for smaller in combinations(weakness.combo, size)
                       if tuple(sorted(smaller)) in by_combo]
            if any(parent.fail_rate >= weakness.fail_rate - 1e-9
                   for parent in general):
                continue
            kept.append(weakness)
        return kept

    def status(self) -> dict:
        return {"scans": self.scans, "tested": self.tested,
                "found": len(self.last), "alpha": self.alpha,
                "min_support": self.min_support}
