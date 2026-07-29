"""Metric time-series on disk (spec M9.2).

Until now every metric lived only inside ``full_status()`` — a snapshot with no
history. Nothing could answer "is the Brier score improving?", and the discovery
engine (M7) would have had nothing to look for laws in. This module is that
history: one append-only JSONL per metric, buffered, bounded, and readable back
as columns.

Two design points worth stating:

* **Downsampling, not truncation.** When a metric outgrows its budget the older
  half is averaged into buckets instead of deleted. Cutting the head off the
  series would remove precisely the long-range shape a law is fitted to.
* **Columnar reads.** ``series()`` returns parallel lists rather than a list of
  dicts, because every consumer (regression, correlation scan, charts) wants
  columns and would otherwise transpose it again.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from aegis.clock import CLOCK
from aegis.config import (
    TELEMETRY_DIR, TELEMETRY_FLUSH_ROWS, TELEMETRY_FLUSH_SECONDS,
    TELEMETRY_MAX_ROWS,
)

logger = logging.getLogger("aegis.telemetry")

SCHEMA_VERSION = 2

# A metric name becomes a file name, so it must not be able to escape the
# telemetry directory or collide after sanitisation.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME = 120


@dataclass
class Series:
    """A metric read back as columns."""

    metric: str
    ticks: list[int] = field(default_factory=list)   # window START (== tick for raw points)
    ends: list[int] = field(default_factory=list)    # window END   (== tick for raw points)
    values: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)  # 1 = raw point, >1 = bucket
    tags: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    def mean(self) -> float | None:
        """Count-weighted mean, so downsampled buckets carry their real weight.

        With no usable weights — a series read from an older schema, or one
        built by hand — this falls back to the plain mean. Falling back on the
        denominator alone (the obvious version) leaves a numerator of zero and
        reports every such series as 0.0.
        """
        if not self.values:
            return None
        weights = self.counts if len(self.counts) == len(self.values) else []
        total_n = sum(weights)
        if total_n <= 0:
            return sum(self.values) / len(self.values)
        return sum(v * n for v, n in zip(self.values, weights)) / total_n

    def rows(self) -> list[dict]:
        return [
            {"tick": t, "tick_end": e, "value": v, "t": ts, "n": n, "tags": tg}
            for t, e, v, ts, n, tg in zip(
                self.ticks, self.ends, self.values, self.times,
                self.counts, self.tags,
            )
        ]


class Telemetry:
    """Buffered writer + reader for metric series."""

    def __init__(self, directory: Path | None = None,
                 max_rows: int = TELEMETRY_MAX_ROWS,
                 flush_rows: int = TELEMETRY_FLUSH_ROWS,
                 flush_seconds: float = TELEMETRY_FLUSH_SECONDS):
        self._dir = Path(directory) if directory is not None else TELEMETRY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_rows = max(2, int(max_rows))
        self._flush_rows = max(1, int(flush_rows))
        self._flush_seconds = max(0.0, float(flush_seconds))
        self._buffer: dict[str, list[dict]] = {}
        self._pending = 0
        self._last_flush = CLOCK.now()
        self.records_written = 0
        self.flushes = 0
        self.downsamples = 0
        self.dropped = 0
        #: Rows currently on disk, per file. Counted once when a file is first
        #: touched and maintained from then on.
        #:
        #: This exists because the compaction check used to answer "is this
        #: series over its cap" by reading the whole file — on every flush, for
        #: every metric. Compaction almost never fires (the cap is two hundred
        #: thousand rows) so the read was almost always wasted, and it grew with
        #: the length of the run: fifty-seven files, read end to end, inside a
        #: cognitive phase. It was 86% of the tick's measured cost and the
        #: reason ACT and PERCEIVE sat over their §3.4 budgets.
        self._rows_on_disk: dict[str, int] = {}

    # ── naming ───────────────────────────────────────────────────────

    @staticmethod
    def safe_name(metric: str) -> str:
        """File-safe metric name. Rejects nothing — sanitises everything — so a
        caller can never write outside the telemetry directory via a name like
        ``../../etc/passwd``."""
        name = _SAFE_NAME.sub("_", str(metric).strip())[:_MAX_NAME]
        name = name.strip(".") or "unnamed"
        return name

    def path_for(self, metric: str) -> Path:
        return self._dir / f"{self.safe_name(metric)}.jsonl"

    # ── writing ──────────────────────────────────────────────────────

    def record(self, metric: str, value, tick: int = 0, tags: dict | None = None) -> bool:
        """Buffer one observation. Returns False if the value is not a number.

        Non-numeric values are dropped rather than raising: telemetry is called
        from inside cognitive phases, and a bad metric must never abort a tick.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            self.dropped += 1
            return False
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            # NaN/inf would poison every downstream mean and fit.
            self.dropped += 1
            return False

        row = {
            "tick": int(tick),
            "value": numeric,
            "t": round(CLOCK.now(), 3),
            "n": 1,
        }
        if tags:
            row["tags"] = {str(k): tags[k] for k in sorted(tags)}
        self._buffer.setdefault(self.safe_name(metric), []).append(row)
        self._pending += 1

        if (self._pending >= self._flush_rows
                or CLOCK.now() - self._last_flush >= self._flush_seconds):
            self.flush()
        return True

    def record_many(self, metrics: dict, tick: int = 0, tags: dict | None = None) -> int:
        """Record a whole batch; returns how many were accepted."""
        return sum(1 for name, value in sorted(metrics.items())
                   if self.record(name, value, tick, tags))

    def flush(self) -> int:
        """Write buffered rows to disk. Never raises."""
        if not self._buffer:
            self._last_flush = CLOCK.now()
            return 0
        written = 0
        for name, rows in sorted(self._buffer.items()):
            path = self._dir / f"{name}.jsonl"
            try:
                with path.open("a", encoding="utf-8") as fh:
                    # A kill during a previous append can leave the last line
                    # without its newline. Appending straight onto it would
                    # glue the first new row to the torn one and lose both —
                    # the tear is already unreadable, and this stops it taking
                    # a good record with it.
                    # Checked only on the first flush that touches a file. Past
                    # that this store wrote the last line itself and knows it
                    # ended in a newline; asking the file system again on every
                    # flush costs a stat and a seek per metric per tick — the
                    # same per-flush IO the row counter exists to avoid, put
                    # straight back.
                    if name not in self._rows_on_disk and self._needs_newline(path):
                        fh.write("\n")
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += len(rows)
            except Exception:
                logger.warning("Failed to write telemetry for %s", name, exc_info=True)
                continue
            known = self._rows_on_disk.get(name)
            # The rows are already in the file by now, so a first count reads
            # the new total directly; afterwards it is simple arithmetic.
            self._rows_on_disk[name] = (self._count_file(path) if known is None
                                        else known + len(rows))
            # A cheap pre-filter on the in-memory count, deliberately looser
            # than the real threshold: it only has to be right about when *not*
            # to look at the file. The exact decision — over the cap or not — is
            # made inside, from the file itself, because the count kept here is
            # an estimate and compacting a series that did not need it would
            # throw away resolution for nothing.
            if self._rows_on_disk[name] > self._max_rows:
                if self._compact_if_needed(path):
                    self._rows_on_disk[name] = self._count_file(path)
        self._buffer.clear()
        self._pending = 0
        self._last_flush = CLOCK.now()
        self.records_written += written
        self.flushes += 1
        return written

    # ── retention ────────────────────────────────────────────────────

    @staticmethod
    def _needs_newline(path: Path) -> bool:
        """Whether the file ends mid-line, which a kill during an append does.

        Reads one byte from the end rather than the file: this runs on every
        flush, and the whole point of the surrounding work is that the flush
        does not read files.
        """
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == 0:
            return False
        try:
            with path.open("rb") as handle:
                handle.seek(-1, 2)
                return handle.read(1) != b"\n"
        except OSError:
            return False

    @staticmethod
    def _count_file(path: Path) -> int:
        """Lines in a file, counted without holding it in memory.

        Called once per file per process — on the first flush that touches it,
        which is the only moment the store cannot know the answer, because the
        file may have been left by an earlier run — and again after a
        compaction rewrites it.
        """
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle)
        except Exception:
            return 0

    def _compact_if_needed(self, path: Path) -> bool:
        """Downsample the older half once a series grows past twice its budget.

        Deleting old rows is the obvious move and the wrong one: the discovery
        engine fits models over long windows, and a series that always starts
        "recently" can never show a slow trend. Averaging preserves the trend at
        lower resolution, and ``n`` keeps the weighting honest.
        """
        try:
            with path.open("r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception:
            logger.warning("Failed to read telemetry file %s", path, exc_info=True)
            return False
        if len(lines) <= self._max_rows * 2:
            return False

        rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # one torn line must not discard the whole history
            if isinstance(row, dict) and "value" in row:
                rows.append(row)
        if len(rows) <= self._max_rows:
            return False

        keep_raw = self._max_rows // 2
        old, recent = rows[:-keep_raw], rows[-keep_raw:]
        bucket = max(2, (len(old) + keep_raw - 1) // max(1, keep_raw))
        compacted = [self._merge(old[i:i + bucket]) for i in range(0, len(old), bucket)]

        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                          for r in compacted + recent)
        try:
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(path)
        except Exception:
            logger.warning("Failed to compact telemetry file %s", path, exc_info=True)
            return False
        self.downsamples += 1
        return True

    @staticmethod
    def _merge(rows: list[dict]) -> dict:
        """Average a bucket of rows into one point, preserving total weight.

        The point is labelled with the window it covers (``tick`` = start,
        ``tick_end`` = end), not with a single edge. Labelling a bucket by its
        last tick made the series look as though its oldest data began later
        and later after each round of compaction — the history was there, but
        unreadable.
        """
        total_n = sum(int(r.get("n", 1)) for r in rows) or len(rows)
        value = sum(float(r.get("value", 0.0)) * int(r.get("n", 1)) for r in rows) / total_n
        return {
            "tick": int(rows[0].get("tick", 0)),
            "tick_end": int(rows[-1].get("tick_end", rows[-1].get("tick", 0))),
            "value": round(value, 6),
            "t": round(float(rows[0].get("t", 0.0)), 3),
            "n": total_n,
        }

    # ── reading ──────────────────────────────────────────────────────

    def series(self, metric: str, window: int | None = None,
               include_buffer: bool = True) -> Series:
        """Read a metric back as columns, newest last.

        ``include_buffer`` folds in rows not yet flushed, so a caller never sees
        a metric it just recorded go missing for up to a flush interval.
        """
        name = self.safe_name(metric)
        out = Series(metric=name)
        rows: list[dict] = []
        path = self._dir / f"{name}.jsonl"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(row, dict) and "value" in row:
                            rows.append(row)
            except Exception:
                logger.warning("Failed to read telemetry for %s", name, exc_info=True)
        if include_buffer:
            rows.extend(self._buffer.get(name, []))
        if window is not None and window > 0:
            rows = rows[-window:]

        for row in rows:
            try:
                out.values.append(float(row["value"]))
            except (TypeError, ValueError, KeyError):
                continue
            tick = int(row.get("tick", 0))
            out.ticks.append(tick)
            out.ends.append(int(row.get("tick_end", tick)))
            out.times.append(float(row.get("t", 0.0)))
            out.counts.append(int(row.get("n", 1)))
            out.tags.append(row.get("tags", {}))
        return out

    def metrics(self) -> list[str]:
        """Every metric known on disk or in the buffer, sorted."""
        names = {p.stem for p in self._dir.glob("*.jsonl")}
        names |= set(self._buffer)
        return sorted(names)

    def status(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "directory": str(self._dir),
            "metrics": len(self.metrics()),
            "records_written": self.records_written,
            "pending": self._pending,
            "flushes": self.flushes,
            "downsamples": self.downsamples,
            "dropped": self.dropped,
            "max_rows_per_metric": self._max_rows,
        }
