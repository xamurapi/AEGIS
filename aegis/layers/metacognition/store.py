"""Persistence for the metacognition contour (spec M11.7.8, §3.2).

Three files under ``data/metacognition/``: explanations, the mechanism credit
table (with the strategy archive beside it — both are the invention loop's
memory), and the skeleton retirements. All at ``schema_version = 1``, written
atomically (tmp + ``os.replace`` inside :func:`write_store`), read through
``store/migrations.py::read_store`` so a file from a future build is refused
rather than half-read. When the schema moves to 2, the migration registers in
``MIGRATIONS`` like every other store — registering a placeholder now would be
dead code wearing a coverage costume, the exact thing the migrations module
warns against.

Eviction: explanations are capped at ``META_MAX_EXPLANATIONS`` and evicted by
age — but ``unsupported`` first, whatever its age, because it is the cheapest
kind to reconstruct: it carries no confirmed mechanism anyone reasons from.
"""
from __future__ import annotations

import logging
from pathlib import Path

import aegis.config as cfg
from aegis.layers.metacognition.attribution import Explanation
from aegis.layers.metacognition.distance import StrategyArchive
from aegis.layers.metacognition.mechanism import CreditTable
from aegis.layers.metacognition.skeletons import SkeletonCatalog
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.metacognition")

SCHEMA_VERSION = 1


class MetaStore:
    """Reads and writes the contour's three files. Never raises on load."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or cfg.META_DIR)

    # ── paths ────────────────────────────────────────────────────────

    @property
    def explanations_path(self) -> Path:
        return self.root / "explanations.json"

    @property
    def mechanisms_path(self) -> Path:
        return self.root / "mechanisms.json"

    @property
    def retired_path(self) -> Path:
        return self.root / "retired.json"

    # ── explanations ─────────────────────────────────────────────────

    def load_explanations(self) -> list[Explanation]:
        data = read_store(self.explanations_path, store="meta_explanations",
                          target_version=SCHEMA_VERSION)
        out = []
        for row in data.get("explanations") or []:
            if not isinstance(row, dict):
                continue
            try:
                out.append(Explanation.from_dict(row))
            except (TypeError, ValueError):
                logger.debug("Ignoring a malformed stored explanation")
        return out

    def save_explanations(self, explanations: list[Explanation]) -> bool:
        return write_store(
            self.explanations_path,
            {"explanations": [e.as_dict() for e in
                              evict(explanations,
                                    cap=int(cfg.META_MAX_EXPLANATIONS))]},
            target_version=SCHEMA_VERSION)

    # ── credit and archive ───────────────────────────────────────────

    def load_credit(self) -> tuple[CreditTable, StrategyArchive]:
        data = read_store(self.mechanisms_path, store="meta_mechanisms",
                          target_version=SCHEMA_VERSION)
        return (CreditTable.from_dict(data.get("credit")),
                StrategyArchive.from_dict(data.get("archive")))

    def save_credit(self, credit: CreditTable,
                    archive: StrategyArchive) -> bool:
        return write_store(self.mechanisms_path,
                           {"credit": credit.to_dict(),
                            "archive": archive.to_dict()},
                           target_version=SCHEMA_VERSION)

    # ── retirements ──────────────────────────────────────────────────

    def load_retired(self) -> SkeletonCatalog:
        data = read_store(self.retired_path, store="meta_retired",
                          target_version=SCHEMA_VERSION)
        return SkeletonCatalog.from_dict(data.get("skeletons"))

    def save_retired(self, catalog: SkeletonCatalog) -> bool:
        return write_store(self.retired_path,
                           {"skeletons": catalog.to_dict()},
                           target_version=SCHEMA_VERSION)


def evict(explanations: list[Explanation], cap: int) -> list[Explanation]:
    """Keep at most ``cap``, oldest ``unsupported`` leaving first.

    Order of departure: ``unsupported`` by age, then everything else by age —
    ``supported`` and ``contested`` carry a verdict someone may still act on,
    and ``contested`` in particular records a live disagreement.
    """
    cap = max(1, int(cap))
    if len(explanations) <= cap:
        return list(explanations)
    cheap_first = sorted(
        explanations,
        key=lambda e: (e.status != "unsupported", e.created_tick, e.strategy))
    surplus = len(explanations) - cap
    evicted = set(id(e) for e in cheap_first[:surplus])
    return [e for e in explanations if id(e) not in evicted]
