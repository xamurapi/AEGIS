"""The skeleton catalogue: structurally different ways to think (spec M11.6.2).

A skeleton is the shape of a strategy without its parameters. The catalogue is
**closed and declared**: the system fills skeletons deterministically, it never
invents new ones — free generation of program structure is exactly what M11.2.4
rules out, and what it would take to lift that is written down in M11.20.

Six shapes, pairwise ``distance ≥ META_FAR`` (a test holds that line, because a
catalogue that drifts into a pile of near-duplicates is a neighbourhood
generator wearing a costume).

Filling is deterministic (§3.1): every parameter slot has a closed option
list, and the choice is hash-indexed from the skeleton, the slot, the
weakness's features and the credit rank of the skeleton's mechanism — so the
credit table steers the fill, ties fall to canonical order, and no PRNG is
consulted anywhere.

Failure memory (M11.6.5): a skeleton that failed ``META_RETIRE_AFTER`` times on
one feature set is retired **for that feature set, for ever** — the same rule
as M7's permanent refutations, and for the same reason: without it every cycle
re-invents the same attractive structure. Retirement is per ``(skeleton,
features)``, never per skeleton — a shape useless on incomplete data may be
right on long chains.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.layers.metacognition.mechanism import (
    CreditTable, mechanism_of_transform,
)
from aegis.layers.reasoning.dsl import validate
from aegis.util.quasirandom import hash_index

logger = logging.getLogger("aegis.metacognition")


def _step(op: str, **fields_) -> dict:
    return {"op": op, **fields_}


#: name -> (template, parameter slots). A slot is ``(path, key, options)``:
#: ``path`` walks nested bodies by (index, body_field, index, ...), and the
#: options are a closed canonical list — the fill picks one, never writes a
#: free value.
_CATALOG: dict[str, tuple[list, tuple]] = {
    "decompose_solve_verify": (
        [_step("DECOMPOSE", max_parts="$gene:reason_decompose_parts"),
         _step("LOOP", body=[_step("SOLVE")], max_iter=2),
         _step("VERIFY", checker="type")],
        # max_iter starts at 1: the loop's declared cost multiplies its body,
        # and a skeleton whose cheapest fill is twice the incumbent would be
        # priced out of the arena before its structure was ever measured.
        (((1,), "max_iter", (1, 2, 3)),
         ((2,), "checker", ("type", "consistency", "confidence"))),
    ),
    "predict_branch_abstain": (
        [_step("PREDICT", horizon=1),
         _step("BRANCH", cond="p_success_below:0.25",
               then=[_step("ABSTAIN", reason="predicted failure")],
               **{"else": [_step("SOLVE")]})],
        (((1,), "cond", ("p_success_below:0.25", "p_success_below:0.35",
                         "p_success_below:0.5")),),
    ),
    "retrieve_compute_verify": (
        [_step("RETRIEVE", source="memory", k=5),
         _step("COMPUTE", expr="$last"),
         _step("VERIFY", checker="type")],
        (((0,), "source", ("memory", "graph", "skills")),
         ((2,), "checker", ("type", "consistency"))),
    ),
    "vote_of_alternatives": (
        [_step("VOTE", n=3, agg="majority",
               body=[_step("SOLVE"), _step("COMPUTE", expr="$last")])],
        (((0,), "n", (3, 5)),
         ((0,), "agg", ("majority", "first"))),
    ),
    "verify_first_then_solve": (
        [_step("VERIFY", checker="consistency"),
         _step("BRANCH", cond="verify_failed",
               then=[_step("DECOMPOSE",
                           max_parts="$gene:reason_decompose_parts"),
                     _step("SOLVE")],
               **{"else": [_step("SOLVE")]})],
        (((0,), "checker", ("consistency", "confidence")),),
    ),
    "reflect_retry_bounded": (
        [_step("SOLVE"),
         _step("REFLECT"),
         _step("BRANCH", cond="insufficient",
               then=[_step("ABSTAIN", reason="the answer would be a guess")],
               **{"else": [_step("LOOP", body=[_step("SOLVE")], max_iter=2)]})],
        (((2,), "cond", ("insufficient", "verify_failed")),
         ((2, "else", 0), "max_iter", (1, 2))),
    ),
}

SKELETON_NAMES: tuple[str, ...] = tuple(sorted(_CATALOG))


def skeleton_template(name: str) -> list:
    """The unfilled shape, a fresh copy."""
    template, _slots = _CATALOG[str(name)]
    return _copy(template)


def _copy(tree):
    if isinstance(tree, list):
        return [_copy(item) for item in tree]
    if isinstance(tree, dict):
        return {key: _copy(value) for key, value in tree.items()}
    return tree


def _walk(steps: list, path: tuple):
    """The step a slot path points at."""
    node = steps[path[0]]
    index = 1
    while index < len(path):
        node = node[path[index]][path[index + 1]]
        index += 2
    return node


def fill(name: str, features, credit: CreditTable) -> list:
    """A skeleton with its parameters chosen — deterministically (M11.6.2).

    The choice folds in the credit rank of the skeleton's mechanism on this
    weakness's features, so accumulating credit genuinely changes what gets
    proposed; everything else in the seed is canonical, so two runs with the
    same table propose the same strategy.
    """
    template, slots = _CATALOG[str(name)]
    steps = _copy(template)
    features = tuple(sorted(set(features or ())))
    mechanism = mechanism_of_transform(f"skeleton:{name}")
    order = credit.order(features) if credit is not None else ()
    rank = order.index(f"skeleton:{name}") if f"skeleton:{name}" in order else 0
    for path, key, options in slots:
        choice = hash_index(len(options), "meta_skeleton", name, key,
                            rank, *features)
        _walk(steps, path)[key] = options[choice]
    problems = validate(steps)
    if problems:                    # a catalogue entry that fails is a bug here
        raise ValueError(f"skeleton {name!r}: {problems}")
    return steps


@dataclass
class _Retirement:
    skeleton: str
    features: str               # "|".join(sorted(combo))
    fails: int = 0
    retired: bool = False

    def as_dict(self) -> dict:
        return {"skeleton": self.skeleton, "features": self.features,
                "fails": self.fails, "retired": self.retired}


class SkeletonCatalog:
    """The catalogue plus its permanent failure memory."""

    def __init__(self, retire_after: int | None = None):
        self.retire_after = int(
            cfg.META_RETIRE_AFTER if retire_after is None else retire_after)
        #: (skeleton, features_key) -> _Retirement
        self.retirements: dict[tuple[str, str], _Retirement] = {}

    @staticmethod
    def _key(features) -> str:
        return "|".join(sorted(set(str(f) for f in features or ())))

    def is_retired(self, name: str, features) -> bool:
        row = self.retirements.get((str(name), self._key(features)))
        return bool(row and row.retired)

    def note_failure(self, name: str, features) -> None:
        key = (str(name), self._key(features))
        row = self.retirements.get(key)
        if row is None:
            row = self.retirements[key] = _Retirement(skeleton=key[0],
                                                      features=key[1])
        row.fails += 1
        if row.fails >= self.retire_after and not row.retired:
            row.retired = True
            logger.info("Skeleton %r retired for features %r after %d failures",
                        key[0], key[1], row.fails)

    def note_success(self, name: str, features) -> None:
        """A win resets nothing — retirement is permanent — but is recorded."""
        key = (str(name), self._key(features))
        row = self.retirements.get(key)
        if row is not None and not row.retired:
            row.fails = 0

    def available(self, features) -> tuple[str, ...]:
        """Skeletons not retired for this feature set, canonical order."""
        return tuple(name for name in SKELETON_NAMES
                     if not self.is_retired(name, features))

    def retired_report(self) -> list[dict]:
        return [row.as_dict() for _, row in sorted(self.retirements.items())
                if row.retired]

    def to_dict(self) -> dict:
        return {"retire_after": self.retire_after,
                "retirements": [row.as_dict()
                                for _, row in sorted(self.retirements.items())]}

    @classmethod
    def from_dict(cls, data: dict | None) -> "SkeletonCatalog":
        data = data or {}
        catalog = cls(retire_after=data.get("retire_after"))
        for row in data.get("retirements") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("skeleton", ""))
            features = str(row.get("features", ""))
            if not name:
                continue
            try:
                catalog.retirements[(name, features)] = _Retirement(
                    skeleton=name, features=features,
                    fails=max(0, int(row.get("fails", 0))),
                    retired=bool(row.get("retired")))
            except (TypeError, ValueError):
                continue
        return catalog

    def status(self) -> dict:
        return {"skeletons": list(SKELETON_NAMES),
                "retire_after": self.retire_after,
                "retired": self.retired_report()}
