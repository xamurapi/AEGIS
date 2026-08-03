"""The closed vocabulary of mechanisms and the credit that orders synthesis
(spec M11.5.3, M11.6.3).

Free text cannot be counted and cannot be tested, so "why a strategy won" is an
enumeration: eight mechanisms, each earned by a particular kind of edit to a
particular operation. The mapping is declarative and **total** over every
``(op, kind)`` pair the diff can produce — a pair without a mechanism would be
an edit whose confirmation could never be stated, and the invariant of M11.4
(``mechanism`` non-empty iff at least one edit is confirmed) would quietly
break.

The credit table is where the arrow *experience → changed behaviour* closes on
the meta level. Every candidate the arena judges is an attempt for its
mechanism on the weakness's features; every acceptance is a win; and
:meth:`CreditTable.order` turns that record into the order in which the
synthesiser tries transformations and skeletons — deterministic UCB, untried
first, canonical tie-break, no RNG anywhere (§3.1).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import aegis.config as cfg
from aegis.layers.reasoning.dsl import OPS

logger = logging.getLogger("aegis.metacognition")

#: The eight mechanisms, canonically ordered. Closed: code may compare against
#: these, never invent a ninth.
MECHANISMS: tuple[str, ...] = (
    "abstention_avoided_confident_error",
    "computation_replaced_guess",
    "decomposition_shortened_chain",
    "prediction_pruned_branch",
    "reflection_reordered_work",
    "retrieval_supplied_missing_fact",
    "verification_caught_error",
    "voting_reduced_variance",
)

#: The kinds an :class:`~aegis.layers.metacognition.attribution.Edit` may have.
EDIT_KINDS: tuple[str, ...] = ("insert", "remove", "param", "wrap", "reorder")

#: Which mechanism an edit to this operation expresses (M11.5.3). Total over
#: the vocabulary: the spec's table names the pairs that matter, and the
#: remaining operations are assigned to the mechanism whose hypothesis their
#: edit expresses — an extra SOLVE is an extra attempt at the answer (variance),
#: an LLM_STEP edit moves the guess/computation boundary, a LOOP is the
#: machinery decomposition runs on, a REFLECT edit is reflection by name.
_OP_MECHANISM: dict[str, str] = {
    "DECOMPOSE": "decomposition_shortened_chain",
    "LOOP": "decomposition_shortened_chain",
    "VERIFY": "verification_caught_error",
    "ABSTAIN": "abstention_avoided_confident_error",
    "VOTE": "voting_reduced_variance",
    "SOLVE": "voting_reduced_variance",
    "COMPUTE": "computation_replaced_guess",
    "LLM_STEP": "computation_replaced_guess",
    "PREDICT": "prediction_pruned_branch",
    "BRANCH": "prediction_pruned_branch",
    "RETRIEVE": "retrieval_supplied_missing_fact",
    "REFLECT": "reflection_reordered_work",
}


def mechanism_for(op: str, kind: str) -> str:
    """The one mechanism a ``(op, kind)`` edit expresses.

    ``wrap`` and ``reorder`` are about the shape of the work, whatever
    operation they touch; everything else is about the operation edited.
    Raises on a pair outside the vocabulary — an unknown operation here means
    the diff produced something the DSL does not contain, which is a bug, not
    a case.
    """
    if kind not in EDIT_KINDS:
        raise KeyError(f"unknown edit kind {kind!r}")
    if kind in ("wrap", "reorder"):
        return "reflection_reordered_work"
    mechanism = _OP_MECHANISM.get(str(op))
    if mechanism is None:
        raise KeyError(f"no mechanism for operation {op!r}")
    return mechanism


def reverse_mapping() -> dict[str, list[tuple[str, str]]]:
    """mechanism -> every ``(op, kind)`` pair that produces it.

    Non-empty for every mechanism, or the mechanism is decorative — the test
    the spec demands.
    """
    table: dict[str, list[tuple[str, str]]] = {name: [] for name in MECHANISMS}
    for op in sorted(OPS):
        for kind in EDIT_KINDS:
            table[mechanism_for(op, kind)].append((op, kind))
    return table


#: What each of the six M6.7 transformations expresses, and what each skeleton
#: expresses (M11.6.2). This is what lets the credit table order both.
TRANSFORM_MECHANISM: dict[str, str] = {
    "add_abstain": "abstention_avoided_confident_error",
    "add_decompose": "decomposition_shortened_chain",
    "add_verify": "verification_caught_error",
    "add_predict": "prediction_pruned_branch",
    "raise_vote": "voting_reduced_variance",
    "compute_instead_of_llm": "computation_replaced_guess",
    "skeleton:decompose_solve_verify": "decomposition_shortened_chain",
    "skeleton:predict_branch_abstain": "prediction_pruned_branch",
    "skeleton:retrieve_compute_verify": "retrieval_supplied_missing_fact",
    "skeleton:vote_of_alternatives": "voting_reduced_variance",
    "skeleton:verify_first_then_solve": "verification_caught_error",
    "skeleton:reflect_retry_bounded": "reflection_reordered_work",
}

#: The fixed order M6 used before this module existed — the baseline that
#: ``order_delta`` is measured against.
FIXED_TRANSFORM_ORDER: tuple[str, ...] = (
    "add_abstain", "add_decompose", "add_verify", "add_predict",
    "raise_vote", "compute_instead_of_llm",
)


def mechanism_of_transform(transform: str) -> str:
    return TRANSFORM_MECHANISM.get(str(transform), "")


@dataclass
class MechanismCredit:
    """How often a mechanism led to an accepted strategy, per weakness feature."""

    mechanism: str
    feature: str
    attempts: int = 0
    accepted: int = 0
    total_effect: float = 0.0

    def as_dict(self) -> dict:
        return {"mechanism": self.mechanism, "feature": self.feature,
                "attempts": self.attempts, "accepted": self.accepted,
                "total_effect": round(self.total_effect, 6)}


class CreditTable:
    """The record that orders synthesis (M11.6.3). Deterministic throughout."""

    def __init__(self, mechanism_c: float | None = None):
        self.mechanism_c = float(
            cfg.META_MECHANISM_C if mechanism_c is None else mechanism_c)
        #: (mechanism, feature) -> MechanismCredit
        self.rows: dict[tuple[str, str], MechanismCredit] = {}

    # ── recording ────────────────────────────────────────────────────

    def _row(self, mechanism: str, feature: str) -> MechanismCredit:
        key = (str(mechanism), str(feature))
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = MechanismCredit(mechanism=key[0],
                                                   feature=key[1])
        return row

    def note_attempt(self, mechanism: str, features) -> None:
        if mechanism not in MECHANISMS:
            return
        for feature in sorted(set(features or ())):
            self._row(mechanism, feature).attempts += 1

    def note_accepted(self, mechanism: str, features, effect: float = 0.0) -> None:
        if mechanism not in MECHANISMS:
            return
        for feature in sorted(set(features or ())):
            row = self._row(mechanism, feature)
            row.accepted += 1
            row.total_effect += float(effect)

    def note_confirmed_effect(self, mechanism: str, features,
                              effect: float) -> None:
        """A confirmed ablation effect, credited without an extra acceptance."""
        if mechanism not in MECHANISMS:
            return
        for feature in sorted(set(features or ())):
            self._row(mechanism, feature).total_effect += float(effect)

    # ── ordering ─────────────────────────────────────────────────────

    def score(self, mechanism: str, features) -> float:
        """Deterministic UCB of one mechanism over a feature set.

        ``attempts == 0`` on any feature is ``+inf``: an untried mechanism is
        tried, always. Any finite default would let one lucky mechanism crowd
        the untried out for ever (M11.6.3).
        """
        features = sorted(set(features or ()))
        if not features:
            return float("inf")
        total = sum(row.attempts for (name, feature), row in self.rows.items()
                    if feature in features) or 1
        scores = []
        for feature in features:
            row = self.rows.get((mechanism, feature))
            if row is None or row.attempts == 0:
                return float("inf")
            scores.append(row.accepted / row.attempts
                          + self.mechanism_c
                          * math.sqrt(math.log(total + 1) / row.attempts))
        return sum(scores) / len(scores)

    def order(self, features, names=None) -> tuple[str, ...]:
        """Transformations and skeletons, best-scoring mechanism first.

        ``names`` defaults to every known generator. Ties break on the
        canonical name, so two runs order identically (§3.1).
        """
        names = list(names if names is not None else sorted(TRANSFORM_MECHANISM))
        scored = []
        for name in names:
            mechanism = mechanism_of_transform(name)
            value = self.score(mechanism, features) if mechanism else 0.0
            scored.append((name, value))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(name for name, _ in scored)

    def order_differs(self, features, names) -> bool:
        """Whether credit actually changed the order for this weakness."""
        names = list(names)
        return self.order(features, names) != tuple(names)

    # ── reporting and persistence ────────────────────────────────────

    def win_rates(self) -> dict[str, dict]:
        """Per mechanism, across features — the dashboard table."""
        table: dict[str, dict] = {}
        for (mechanism, _feature), row in sorted(self.rows.items()):
            entry = table.setdefault(mechanism, {"attempts": 0, "accepted": 0,
                                                 "total_effect": 0.0})
            entry["attempts"] += row.attempts
            entry["accepted"] += row.accepted
            entry["total_effect"] += row.total_effect
        for mechanism, entry in table.items():
            entry["win_rate"] = (entry["accepted"] / entry["attempts"]
                                 if entry["attempts"] else 0.0)
            entry["total_effect"] = round(entry["total_effect"], 6)
        return table

    def to_dict(self) -> dict:
        return {"mechanism_c": self.mechanism_c,
                "rows": [row.as_dict()
                         for _, row in sorted(self.rows.items())]}

    @classmethod
    def from_dict(cls, data: dict | None) -> "CreditTable":
        data = data or {}
        table = cls(mechanism_c=data.get("mechanism_c"))
        for row in data.get("rows") or []:
            if not isinstance(row, dict):
                continue
            mechanism = str(row.get("mechanism", ""))
            feature = str(row.get("feature", ""))
            if mechanism not in MECHANISMS or not feature:
                continue
            try:
                table.rows[(mechanism, feature)] = MechanismCredit(
                    mechanism=mechanism, feature=feature,
                    attempts=max(0, int(row.get("attempts", 0))),
                    accepted=max(0, int(row.get("accepted", 0))),
                    total_effect=float(row.get("total_effect", 0.0)))
            except (TypeError, ValueError):
                continue
        return table

    def status(self) -> dict:
        return {"rows": len(self.rows), "mechanism_c": self.mechanism_c,
                "mechanisms": self.win_rates()}
