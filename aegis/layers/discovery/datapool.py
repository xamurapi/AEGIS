"""The data a discovery is made out of (spec M7.3).

Two things live here: a **column store** with a declared schema and provenance,
and :class:`Frame`, the minimal table the rest of the engine computes on.

``Frame`` is columnar and hand-written rather than a pandas import. Not
austerity — the engine has to give the same answer on a machine where pandas is
absent as on one where it is present, and "the discovery held up" is not a claim
that may depend on what happened to be installed. The operations here are the
ones a hypothesis actually needs: pick columns, drop rows that are not numbers,
shift a column back in time, standardise it, and line two series up by tick.

The **lag** operation is the one worth reading twice. A hypothesis in this system
is almost always about something that happened *earlier* — surprise at tick t
against reward at tick t+1 — and lagging is what makes that expressible without
letting a predictor quietly see its own future. ``lag`` shifts a column forward
and shortens the frame from the front, so every surviving row still has all of
its columns and none of them come from the future.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import aegis.config as cfg
from aegis.clock import CLOCK
from aegis.store.migrations import read_store, write_store
from aegis.util.stats import mean

logger = logging.getLogger("aegis.discovery")

#: Rows kept per dataset. Beyond this the oldest are dropped: the engine looks
#: for relationships that hold *now*, and a series that reaches back far enough
#: to span several champions is a series describing several different systems.
MAX_ROWS = 20_000

#: Column types the schema may declare. Anything else is refused at registration
#: rather than discovered halfway through a regression.
TYPES = ("float", "int", "bool", "str")


@dataclass(frozen=True)
class VariableSpec:
    """One column, and what may be done with it.

    ``controllable`` is a safety property, not a description: it is what decides
    whether an interventional experiment may set this variable. It is set from
    the whitelist of Appendix F and nowhere else — a variable cannot make itself
    controllable by declaring so.
    """

    name: str
    kind: str = "float"
    unit: str = ""
    source: str = ""
    controllable: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "unit": self.unit,
                "source": self.source, "controllable": self.controllable}


@dataclass
class Frame:
    """A small columnar table. Every column has the same length."""

    columns: dict[str, list] = field(default_factory=dict)

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def from_rows(cls, rows, columns=None) -> "Frame":
        rows = list(rows or [])
        if columns is None:
            names: list[str] = []
            for row in rows:
                for key in row:
                    if key not in names:
                        names.append(key)
            columns = sorted(names)
        data = {name: [row.get(name) for row in rows] for name in columns}
        return cls(columns=data)

    @property
    def names(self) -> list[str]:
        return sorted(self.columns)

    def __len__(self) -> int:
        for values in self.columns.values():
            return len(values)
        return 0

    def column(self, name: str) -> list:
        return list(self.columns.get(name, []))

    def rows(self) -> list[dict]:
        names = self.names
        return [{name: self.columns[name][index] for name in names}
                for index in range(len(self))]

    # ── operations ───────────────────────────────────────────────────

    def select(self, *names: str) -> "Frame":
        """Keep only these columns, in the order asked for."""
        return Frame(columns={name: list(self.columns.get(name, []))
                              for name in names})

    def filter(self, predicate) -> "Frame":
        """Keep rows where ``predicate(row)`` holds."""
        keep = [index for index, row in enumerate(self.rows()) if predicate(row)]
        return Frame(columns={name: [values[index] for index in keep]
                              for name, values in self.columns.items()})

    def numeric(self, *names: str) -> "Frame":
        """Drop every row where any named column is not a finite number.

        Missing telemetry is the normal case, not an error: a metric that is
        only written every tenth tick leaves gaps, and a correlation computed
        over ``None`` would be a correlation with the recording schedule.
        """
        names = names or tuple(self.names)

        def _usable(row) -> bool:
            for name in names:
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False
                if value != value or value in (float("inf"), float("-inf")):
                    return False
            return True

        return self.filter(_usable)

    def lag(self, name: str, periods: int = 1, as_name: str | None = None) -> "Frame":
        """Add ``name`` shifted ``periods`` rows into the past.

        The frame shortens from the front by ``periods``, so every remaining row
        carries a complete set of values and the lagged column genuinely holds
        an *earlier* reading. Dropping the rows is what keeps a predictor from
        being padded with a value it could not have known.
        """
        periods = max(0, int(periods))
        as_name = as_name or f"{name}_lag{periods}"
        source = self.column(name)
        if periods == 0:
            out = dict(self.columns)
            out[as_name] = list(source)
            return Frame(columns=out)
        if periods >= len(self):
            return Frame(columns={key: [] for key in list(self.columns) + [as_name]})
        out = {key: values[periods:] for key, values in self.columns.items()}
        out[as_name] = source[:-periods]
        return Frame(columns=out)

    def zscore(self, name: str, as_name: str | None = None) -> "Frame":
        """Standardise a column. A constant column becomes zeros, not NaNs."""
        as_name = as_name or f"{name}_z"
        values = [float(v) for v in self.column(name)]
        out = dict(self.columns)
        if not values:
            out[as_name] = []
            return Frame(columns=out)
        average = mean(values)
        spread = (sum((v - average) ** 2 for v in values) / len(values)) ** 0.5
        out[as_name] = [0.0] * len(values) if spread <= 0 else \
            [(v - average) / spread for v in values]
        return Frame(columns=out)

    def join_on_tick(self, other: "Frame", tick: str = "tick") -> "Frame":
        """Inner join two frames on their tick column.

        Inner rather than outer, and that is the safe direction: an outer join
        invents rows where one side has no reading, and a model fitted through
        invented rows is fitted to the join.
        """
        mine = {row.get(tick): row for row in self.rows()}
        merged = []
        for row in other.rows():
            key = row.get(tick)
            if key in mine:
                combined = dict(mine[key])
                combined.update(row)
                merged.append(combined)
        merged.sort(key=lambda row: row.get(tick, 0))
        return Frame.from_rows(merged)


class DataPool:
    """Named datasets with declared schemas, and the frames built from them."""

    def __init__(self, directory: Path | None = None, *,
                 max_rows: int = MAX_ROWS):
        self.directory = Path(directory) if directory else \
            Path(cfg.DATA_DIR) / "discovery" / "datasets"
        self.max_rows = int(max_rows)
        self.schemas: dict[str, dict[str, VariableSpec]] = {}
        self.sources: dict[str, str] = {}
        self.data: dict[str, list[dict]] = {}
        self.rejected = 0
        self._load()

    # ── registration ─────────────────────────────────────────────────

    def register(self, name: str, columns: dict, source: str = "",
                 controllable=()) -> bool:
        """Declare a dataset. Returns False if the schema is not usable.

        A schema is mandatory (M7.3) and it is the only thing that makes a
        column a variable: an association engine that accepted undeclared
        columns would happily correlate a timestamp with a version string.
        """
        name = str(name)
        if not name or not isinstance(columns, dict) or not columns:
            self.rejected += 1
            return False
        controllable = set(controllable or ())
        schema: dict[str, VariableSpec] = {}
        for column, declared in sorted(columns.items()):
            kind = str(declared if isinstance(declared, str) else
                       (declared or {}).get("kind", "float"))
            if kind not in TYPES:
                self.rejected += 1
                logger.warning("Dataset %s declares column %s as %r, which is "
                               "not one of %s", name, column, kind, list(TYPES))
                return False
            unit = "" if isinstance(declared, str) else \
                str((declared or {}).get("unit", ""))
            schema[str(column)] = VariableSpec(
                name=str(column), kind=kind, unit=unit, source=str(source),
                controllable=str(column) in controllable)
        self.schemas[name] = schema
        self.sources[name] = str(source)
        self.data.setdefault(name, [])
        return True

    def append(self, name: str, row: dict) -> bool:
        """Add one observation. Unknown datasets and stray columns are refused.

        Refusing rather than widening the schema on the fly: the count of
        variables is what the false-discovery correction is computed against
        (M7.4), and a schema that grows by itself makes that count a fiction.
        """
        name = str(name)
        schema = self.schemas.get(name)
        if schema is None or not isinstance(row, dict):
            self.rejected += 1
            return False
        clean = {column: row[column] for column in schema if column in row}
        if not clean:
            self.rejected += 1
            return False
        bucket = self.data.setdefault(name, [])
        bucket.append(clean)
        if len(bucket) > self.max_rows:
            del bucket[:len(bucket) - self.max_rows]
        return True

    def ingest_telemetry(self, telemetry, metrics, name: str = "telemetry",
                         window: int | None = None) -> int:
        """Build one dataset out of several telemetry series, aligned by tick.

        The engine's main source (M7.3). Series are recorded independently and
        at different cadences, so they are joined on tick and only ticks where
        every requested metric has a reading survive — the alternative is
        carrying a stale value forward, which manufactures the autocorrelation
        the scan is looking for.
        """
        metrics = sorted({str(metric) for metric in metrics})
        if not metrics:
            return 0
        by_tick: dict[int, dict] = {}
        present: dict[int, set] = {}
        for metric in metrics:
            try:
                series = telemetry.series(metric, window=window)
            except Exception:
                logger.warning("Telemetry series %s unavailable", metric,
                               exc_info=True)
                continue
            for row in getattr(series, "rows", lambda: [])():
                tick = int(row.get("tick", 0))
                value = row.get("value")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                by_tick.setdefault(tick, {"tick": tick})[metric] = float(value)
                present.setdefault(tick, set()).add(metric)

        self.register(name, {"tick": "int",
                             **{metric: "float" for metric in metrics}},
                      source="telemetry")
        # Only ticks after the newest one already held. Ingestion is periodic
        # and the telemetry series it reads is cumulative, so without this every
        # ingest re-appends everything: the pool grows without bound and, far
        # worse, each observation is counted several times. Duplicated rows
        # shrink every p-value in the scan without adding any evidence, which
        # is precisely the failure the false-discovery control exists to
        # prevent — and it would look like the engine getting better at finding
        # laws the longer it ran.
        newest = max((int(row.get("tick", 0))
                      for row in self.data.get(name, [])), default=None)
        complete = [by_tick[tick] for tick in sorted(by_tick)
                    if present.get(tick, set()) >= set(metrics)
                    and (newest is None or tick > newest)]
        added = 0
        for row in complete:
            if self.append(name, row):
                added += 1
        return added

    # ── reading ──────────────────────────────────────────────────────

    def frame(self, name: str, window: int | None = None) -> Frame:
        rows = self.data.get(str(name), [])
        if window:
            rows = rows[-int(window):]
        columns = ["tick"] if "tick" in self.schemas.get(str(name), {}) else []
        columns += [column for column in sorted(self.schemas.get(str(name), {}))
                    if column != "tick"]
        return Frame.from_rows(rows, columns=columns or None)

    def variables(self, name: str | None = None) -> list[VariableSpec]:
        """Every declared variable, sorted. The candidate set for a scan."""
        out: list[VariableSpec] = []
        for dataset in ([str(name)] if name else sorted(self.schemas)):
            out.extend(self.schemas.get(dataset, {}).values())
        return sorted(out, key=lambda spec: (spec.source, spec.name))

    def datasets(self) -> list[str]:
        return sorted(self.data)

    def row_count(self, name: str) -> int:
        return len(self.data.get(str(name), []))

    # ── persistence ──────────────────────────────────────────────────

    def _path(self) -> Path:
        return self.directory / "pool.json"

    def save(self) -> bool:
        payload = {
            "saved": CLOCK.now(),
            "sources": dict(self.sources),
            "schemas": {name: {column: spec.as_dict()
                               for column, spec in sorted(schema.items())}
                        for name, schema in sorted(self.schemas.items())},
            "data": {name: rows[-self.max_rows:]
                     for name, rows in sorted(self.data.items())},
        }
        return write_store(self._path(), payload)

    def _load(self) -> None:
        payload = read_store(self._path(), store="discovery_datapool")
        schemas = payload.get("schemas")
        if not isinstance(schemas, dict):
            return
        for name, schema in schemas.items():
            if not isinstance(schema, dict):
                continue
            self.schemas[str(name)] = {
                str(column): VariableSpec(
                    name=str(spec.get("name", column)),
                    kind=str(spec.get("kind", "float")),
                    unit=str(spec.get("unit", "")),
                    source=str(spec.get("source", "")),
                    controllable=bool(spec.get("controllable", False)))
                for column, spec in schema.items() if isinstance(spec, dict)}
        sources = payload.get("sources")
        if isinstance(sources, dict):
            self.sources = {str(key): str(value) for key, value in sources.items()}
        data = payload.get("data")
        if isinstance(data, dict):
            for name, rows in data.items():
                if isinstance(rows, list):
                    self.data[str(name)] = [row for row in rows
                                            if isinstance(row, dict)][-self.max_rows:]

    def status(self) -> dict:
        return {"datasets": len(self.data), "rejected": self.rejected,
                "rows": {name: len(rows) for name, rows in sorted(self.data.items())},
                "variables": len(self.variables())}
