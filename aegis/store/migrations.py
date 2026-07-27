"""Schema versions and migrations for every persistent store (spec §3.2, Appendix I).

Three rules the whole system depends on, implemented once here instead of
thirteen times across the contours:

1. **A file with no ``schema_version`` is version 1.** Every store that existed
   before this spec was written falls into that case, so old data keeps loading.
2. **A version from the future is not read.** A newer build may have written
   fields this one would silently drop and then write back — losing them. So an
   unknown version logs a warning and yields empty state; it never crashes and
   never half-reads.
3. **Writes are atomic.** Temp file in the same directory, then replace. A crash
   mid-write leaves either the old complete file or the new one.

Registering a migration is one entry in :data:`MIGRATIONS`: a function from the
old payload to the new one, keyed by ``(store, from_version)``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from aegis._atomic import atomic_write_text

logger = logging.getLogger("aegis.store")

# Every store the spec introduces or upgrades lands on version 2.
CURRENT_VERSION = 2

VERSION_KEY = "schema_version"


class MigrationError(Exception):
    """A registered migration could not complete."""


def version_of(data: dict) -> int:
    """Schema version of a loaded payload; absent means version 1."""
    if not isinstance(data, dict):
        return 1
    try:
        return int(data.get(VERSION_KEY, 1))
    except (TypeError, ValueError):
        return 1


# ── migrations ───────────────────────────────────────────────────────
# Each takes the payload at version N and returns it at version N+1. They must
# be pure and total: given anything the previous version could have written,
# they produce something the new version can read.


def _v1_to_v2_passthrough(data: dict) -> dict:
    """Data whose shape did not change — only the version stamp is added."""
    return dict(data)


def _v1_to_v2_world_model(data: dict) -> dict:
    """Causal links survive verbatim; the predictive tables start empty.

    The transition and outcome tables cannot be reconstructed from cause→effect
    strings — they are keyed by encoded system state, which v1 never recorded.
    Starting them empty is honest; inventing them would seed the new model with
    fiction and every calibration number after that would be wrong.
    """
    out = dict(data)
    out.setdefault("links", {})
    out.setdefault("chains", [])
    out.setdefault("total_observations", 0)
    return out


def _v1_to_v2_evolution(data: dict) -> dict:
    """Genome v1 (LoRA hyper-parameters) is dropped; genome v2 is seeded.

    The old genes (``learning_rate``, ``dropout``, ``attention_heads``…) do not
    influence the measured benchmark at all — that is precisely why the genome
    is being replaced (§M5.3). Carrying their values into the new genome would
    carry the old problem with them. Lineage, counters and generation number
    are history and are kept; a pending candidate in the old format is dropped
    because there is no way to finish judging it correctly.
    """
    from aegis.layers.evolution.genome import default_genome  # local: avoids a cycle

    out = dict(data)
    champion = out.get("champion")
    if isinstance(champion, dict):
        out["champion"] = {
            **champion,
            "genome": default_genome(),
            "migrated_from": "genome_v1",
        }
    out["candidate"] = None
    out.setdefault("generation", 0)
    out.setdefault("accepted", 0)
    out.setdefault("rejected", 0)
    out.setdefault("lineage", [])
    return out


# (store name, from_version) -> migration
MIGRATIONS: dict[tuple[str, int], Callable[[dict], dict]] = {
    ("world_model", 1): _v1_to_v2_world_model,
    ("evolution", 1): _v1_to_v2_evolution,
    ("checkpoint", 1): _v1_to_v2_passthrough,
    ("goal_intelligence", 1): _v1_to_v2_passthrough,
    ("cognitive_graph", 1): _v1_to_v2_passthrough,
    ("skills", 1): _v1_to_v2_passthrough,
    ("eval_history", 1): _v1_to_v2_passthrough,
}


def migrate(data: dict, from_v: int, to_v: int, store: str = "") -> dict:
    """Bring ``data`` from ``from_v`` up to ``to_v``, one step at a time.

    A store with no registered migration for a step is passed through with only
    its version stamp changed: most upgrades are purely additive, and demanding
    a no-op function for each of them would be ceremony, not safety.
    """
    if from_v > to_v:
        raise MigrationError(
            f"cannot migrate {store or 'store'} backwards ({from_v} -> {to_v})")
    current = dict(data)
    for step in range(int(from_v), int(to_v)):
        handler = MIGRATIONS.get((store, step), _v1_to_v2_passthrough)
        try:
            current = handler(current)
        except Exception as exc:  # a broken migration must not take the boot down
            raise MigrationError(
                f"migration of {store or 'store'} {step} -> {step + 1} failed: {exc}"
            ) from exc
        # The version is stamped once, after the whole chain: stamping it per
        # step was dead work — the final assignment below overwrote every
        # intermediate value, so no migration could ever observe it.
    current[VERSION_KEY] = int(to_v)
    return current


def read_store(path: Path, store: str = "", *, default: dict | None = None,
               target_version: int = CURRENT_VERSION) -> dict:
    """Load a versioned store, migrating it forward. Never raises.

    Returns ``default`` (or an empty dict) when the file is missing, unreadable,
    corrupt, or written by a newer build. Every one of those is a normal thing
    to survive: the system must keep ticking with empty state rather than
    refuse to start.
    """
    fallback = dict(default or {})
    fallback.setdefault(VERSION_KEY, target_version)
    path = Path(path)
    if not path.exists():
        return fallback
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Unreadable store %s — continuing with empty state",
                       path, exc_info=True)
        return fallback
    if not isinstance(raw, dict):
        logger.warning("Store %s is not an object — continuing with empty state", path)
        return fallback

    found = version_of(raw)
    if found > target_version:
        logger.warning(
            "Store %s has schema_version %d but this build understands %d. "
            "Refusing to read it (a newer build may use fields this one would "
            "drop on the next write); continuing with empty state.",
            path, found, target_version,
        )
        return fallback
    if found == target_version:
        return raw
    try:
        migrated = migrate(raw, found, target_version, store=store)
    except MigrationError:
        logger.warning("Migration of %s failed — continuing with empty state",
                       path, exc_info=True)
        return fallback
    logger.info("Migrated %s from schema_version %d to %d", path, found, target_version)
    return migrated


def write_store(path: Path, data: dict, *,
                target_version: int = CURRENT_VERSION) -> bool:
    """Write a versioned store atomically. Returns False on failure, never raises."""
    payload = dict(data)
    payload[VERSION_KEY] = int(target_version)
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        logger.warning("Failed to write store %s", path, exc_info=True)
        return False
