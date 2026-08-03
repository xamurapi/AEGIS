"""Canonical form and distance between strategies (spec M11.6.1).

A distance without canonicalisation would measure formatting: two spellings of
one strategy — fields in a different order, ``3`` against ``3.0``, a trailing
``REFLECT`` nobody reads — would come out "different". Canonicalisation first,
then two half-weighted measures over the canonical trees:

* **Edit distance** over a preorder token rendering of the tree — sensitive to
  structure, blind to composition (replacing one operation in an otherwise
  identical shape moves it only slightly).
* **Jaccard distance** over the multisets of operations — sensitive to
  composition, blind to structure (a permutation of the same operations scores
  zero).

Each alone is gameable; the average of the two is what the spec fixes, and the
tests pin the properties that make it usable: ``distance(a, a) == 0``,
symmetry, strictly positive for strategies with different canonical hashes,
invariance to the order independent parameters happen to be written in.

:class:`StrategyArchive` is the M5 ``NoveltyArchive`` pattern over strategies:
it remembers everything that was evaluated and cuts near-duplicates *before*
evaluation, because an arena run costs seconds and a neighbourhood generator
produces near-duplicates constantly.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter

import aegis.config as cfg
from aegis.layers.reasoning.dsl import OPS, normalise

logger = logging.getLogger("aegis.metacognition")


# ── canonical form ───────────────────────────────────────────────────

def canonicalize(steps) -> list:
    """The canonical tree of a strategy.

    Built on the DSL's ``normalise`` (sorted keys, unknown fields dropped,
    bodies normalised recursively), plus the two normalisations distance needs
    and admission does not: numeric literals collapse (``3.0`` is ``3``), and a
    ``REFLECT`` with no step after it in its block is removed — nothing can
    read its result, so it is not part of what the strategy does.
    """
    return _strip_dead_reflect(_normalise_numbers(normalise(steps)))


def _normalise_numbers(tree):
    if isinstance(tree, list):
        return [_normalise_numbers(item) for item in tree]
    if isinstance(tree, dict):
        return {key: _normalise_numbers(value) for key, value in tree.items()}
    if isinstance(tree, float) and not isinstance(tree, bool) \
            and float(tree).is_integer():
        return int(tree)
    return tree


def _strip_dead_reflect(tree):
    if not isinstance(tree, list):
        return tree
    out = []
    for step in tree:
        if isinstance(step, dict):
            step = {key: (_strip_dead_reflect(value) if isinstance(value, list)
                          else value)
                    for key, value in step.items()}
        out.append(step)
    # A trailing REFLECT has no observable effect: nothing after it can read
    # what it noted. Repeated, in case several stacked up.
    while out and isinstance(out[-1], dict) and out[-1].get("op") == "REFLECT":
        out.pop()
    return out


def canonical_json(steps) -> str:
    return json.dumps(canonicalize(steps), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def canonical_hash(steps) -> str:
    """Stable identity of the canonical tree, ``blake2b`` as the spec names."""
    return hashlib.blake2b(canonical_json(steps).encode("utf-8"),
                           digest_size=16).hexdigest()


# ── the two half-measures ────────────────────────────────────────────

def _tokens(tree, out: list | None = None) -> list[str]:
    """Preorder token rendering of a canonical tree.

    Structure markers are tokens too, so nesting depth participates in the
    edit distance — ``VOTE(body=[SOLVE])`` and ``VOTE, SOLVE`` differ.
    """
    if out is None:
        out = []
    for step in tree if isinstance(tree, list) else []:
        if not isinstance(step, dict):
            continue
        out.append(f"op:{step.get('op')}")
        spec = OPS.get(str(step.get("op")))
        bodies = set(spec.bodies) if spec else set()
        for key in sorted(step):
            if key == "op":
                continue
            if key in bodies:
                out.append(f"({key}")
                _tokens(step[key], out)
                out.append(f"){key}")
            else:
                out.append(f"{key}={step[key]!r}")
    return out


def _edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein over token sequences, two rows of memory."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + cost)
        previous = current
    return previous[-1]


def op_multiset(steps) -> Counter:
    """Every operation in the strategy, counted, through every nesting."""
    counts: Counter = Counter()

    def walk(tree):
        for step in tree if isinstance(tree, list) else []:
            if not isinstance(step, dict):
                continue
            counts[str(step.get("op"))] += 1
            for value in step.values():
                if isinstance(value, list):
                    walk(value)

    walk(canonicalize(steps))
    return counts


def _jaccard_distance(a: Counter, b: Counter) -> float:
    union = sum((a | b).values())
    if union == 0:
        return 0.0
    return 1.0 - sum((a & b).values()) / union


def distance(a, b) -> float:
    """Distance between two strategies, in ``[0.0, 1.0]`` (spec M11.6.1).

    Accepts step lists or anything with a ``steps`` attribute. Identical
    canonical hashes are exactly zero — the equality the archive dedups on and
    the distance must agree about.
    """
    steps_a = getattr(a, "steps", a)
    steps_b = getattr(b, "steps", b)
    canon_a, canon_b = canonicalize(steps_a), canonicalize(steps_b)
    if canonical_hash(canon_a) == canonical_hash(canon_b):
        return 0.0
    tokens_a, tokens_b = _tokens(canon_a), _tokens(canon_b)
    longest = max(len(tokens_a), len(tokens_b), 1)
    structural = _edit_distance(tokens_a, tokens_b) / longest
    compositional = _jaccard_distance(op_multiset(canon_a), op_multiset(canon_b))
    return min(1.0, 0.5 * structural + 0.5 * compositional)


# ── the archive ──────────────────────────────────────────────────────

class StrategyArchive:
    """Everything that has been evaluated, so near-duplicates are cut early.

    The M5 ``NoveltyArchive`` pattern over strategies rather than genomes. Two
    questions, two thresholds: :meth:`is_novel` asks "is this worth an arena
    run at all" against ``META_NEAR``; :meth:`min_distance` is what the far
    quota compares against ``META_FAR``.
    """

    def __init__(self, near: float | None = None, capacity: int = 500):
        self.near = float(cfg.META_NEAR if near is None else near)
        self.capacity = int(capacity)
        self.entries: list[dict] = []
        self.skips = 0

    def seen(self, steps) -> bool:
        """Exact canonical identity — already evaluated as-is."""
        mark = canonical_hash(getattr(steps, "steps", steps))
        return any(entry["hash"] == mark for entry in self.entries)

    def min_distance(self, steps) -> float:
        """Distance to the nearest archived strategy; ``inf`` when empty."""
        steps = getattr(steps, "steps", steps)
        if not self.entries:
            return float("inf")
        return min(distance(steps, entry["steps"]) for entry in self.entries)

    def is_novel(self, steps) -> bool:
        return self.min_distance(steps) >= self.near

    def note_skip(self) -> None:
        self.skips += 1

    def add(self, name: str, steps) -> None:
        steps = getattr(steps, "steps", steps)
        mark = canonical_hash(steps)
        if any(entry["hash"] == mark for entry in self.entries):
            return
        self.entries.append({"name": str(name), "hash": mark,
                             "steps": canonicalize(steps)})
        if len(self.entries) > self.capacity:
            # Oldest first: the archive describes where the search has been
            # recently, the same rule the M5 archive uses.
            self.entries = self.entries[-self.capacity:]

    def to_dict(self) -> dict:
        return {"near": self.near, "skips": self.skips,
                "entries": [dict(entry) for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: dict | None) -> "StrategyArchive":
        data = data or {}
        archive = cls(near=data.get("near"))
        try:
            archive.skips = max(0, int(data.get("skips", 0)))
        except (TypeError, ValueError):
            archive.skips = 0
        for entry in data.get("entries") or []:
            if isinstance(entry, dict) and isinstance(entry.get("steps"), list):
                archive.entries.append({
                    "name": str(entry.get("name", "")),
                    "hash": str(entry.get("hash", "")
                                or canonical_hash(entry["steps"])),
                    "steps": canonicalize(entry["steps"]),
                })
        return archive

    def status(self) -> dict:
        return {"size": len(self.entries), "skips": self.skips,
                "near": self.near}
