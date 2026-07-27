"""Response cache (spec §M8.6).

The cache is not primarily a cost saving — it is a *correctness* requirement.
Comparative runs (the A/B planner harness, the reasoning arena, evolution's
generation scoring) compare metrics between configurations, and if the same
prompt returned different text each time, the difference in the metric would
partly be the model's sampling variance rather than the change under test.
Caching by an exact key makes a repeated run reproduce.

Keyed on ``blake2b(provider | model | messages | params)``: everything that
could change the answer, and nothing that could not. Bounded by count and by
age, persisted so the reproducibility survives a restart.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from aegis.clock import CLOCK
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.cortex.cache")


@dataclass
class CacheEntry:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    stored_at: float
    hits: int = 0

    def to_dict(self) -> dict:
        return {"text": self.text, "provider": self.provider, "model": self.model,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "stored_at": self.stored_at, "hits": self.hits}

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry | None:
        try:
            return cls(
                text=str(data["text"]),
                provider=str(data.get("provider", "")),
                model=str(data.get("model", "")),
                tokens_in=int(data.get("tokens_in", 0)),
                tokens_out=int(data.get("tokens_out", 0)),
                stored_at=float(data.get("stored_at", 0.0)),
                hits=int(data.get("hits", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None      # one malformed row must not discard the cache


def cache_key(provider: str, model: str, messages: list[dict], params_part: str) -> str:
    """Stable key for a call.

    Messages are canonicalised through ``json.dumps(sort_keys=True)`` so two
    equivalent requests that differ only in key order share an entry.
    """
    payload = json.dumps(
        {"provider": provider, "model": model, "params": params_part,
         "messages": [{"role": m.get("role", ""), "content": str(m.get("content", ""))}
                      for m in (messages or [])]},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


class ResponseCache:
    """Bounded, persistent, time-limited cache of model responses."""

    def __init__(self, path: Path | None = None, *, ttl: float = 3600.0,
                 max_entries: int = 2000):
        self._path = Path(path) if path is not None else None
        self.ttl = max(0.0, float(ttl))
        self.max_entries = max(1, int(max_entries))
        self._entries: dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        data = read_store(self._path, store="cortex_cache")
        for key, row in (data.get("entries") or {}).items():
            entry = CacheEntry.from_dict(row) if isinstance(row, dict) else None
            if entry is not None:
                self._entries[str(key)] = entry
        self._expire()

    def save(self) -> None:
        if self._path is None:
            return
        self._expire()
        write_store(self._path,
                    {"entries": {k: e.to_dict() for k, e in sorted(self._entries.items())}})

    # ── access ───────────────────────────────────────────────────────

    def get(self, key: str) -> CacheEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if self._is_expired(entry):
            del self._entries[key]
            self.expirations += 1
            self.misses += 1
            return None
        entry.hits += 1
        self.hits += 1
        return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        self._entries[key] = entry
        self._evict_if_needed()

    def clear(self) -> None:
        self._entries.clear()

    # ── retention ────────────────────────────────────────────────────

    def _is_expired(self, entry: CacheEntry) -> bool:
        return self.ttl > 0 and (CLOCK.now() - entry.stored_at) > self.ttl

    def _expire(self) -> None:
        stale = [k for k, e in self._entries.items() if self._is_expired(e)]
        for key in stale:
            del self._entries[key]
        self.expirations += len(stale)

    def _evict_if_needed(self) -> None:
        self._expire()
        excess = len(self._entries) - self.max_entries
        if excess <= 0:
            return
        # Evict the least useful first: never-reused entries, oldest among them.
        # Plain LRU would keep a one-off answer that happened to arrive late
        # ahead of a prompt the arena replays on every single run.
        ranked = sorted(self._entries.items(),
                        key=lambda kv: (kv[1].hits, kv[1].stored_at))
        for key, _ in ranked[:excess]:
            del self._entries[key]
        self.evictions += excess

    # ── reporting ────────────────────────────────────────────────────

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def status(self) -> dict:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
            "evictions": self.evictions,
            "expirations": self.expirations,
        }

    def __len__(self) -> int:
        return len(self._entries)
