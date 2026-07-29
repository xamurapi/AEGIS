"""The data a discovery is made out of (spec M7.3, M7.11).

Two claims carry the weight here. **A schema is mandatory** — an engine that
correlated undeclared columns would correlate whatever happened to be in the
row, and the count of variables is what the false-discovery correction is
computed against. And **lagging must not leak the future** — a lagged frame
whose rows were padded rather than dropped hands a predictor a value it could
not have known, which is the one mistake that makes every time series look
predictable.
"""
import pytest

from aegis.layers.discovery.datapool import (
    MAX_ROWS, DataPool, Frame, VariableSpec,
)


@pytest.fixture
def pool(tmp_path):
    return DataPool(tmp_path / "datasets")


def _rows(count, **extra):
    return [{"tick": index, "x": float(index), "y": float(index) * 2, **extra}
            for index in range(count)]


# ── Frame ────────────────────────────────────────────────────────────

def test_a_frame_built_from_rows_keeps_every_column():
    frame = Frame.from_rows(_rows(5))
    assert frame.names == ["tick", "x", "y"]
    assert len(frame) == 5


def test_an_empty_frame_has_no_length_and_no_columns():
    frame = Frame.from_rows([])
    assert len(frame) == 0 and frame.names == []


def test_a_missing_value_becomes_none_rather_than_a_shorter_column():
    """Columns must stay the same length or every row-wise operation below
    silently pairs the wrong observations."""
    frame = Frame.from_rows([{"a": 1}, {"a": 2, "b": 3}])
    assert frame.column("b") == [None, 3]
    assert len(frame.column("a")) == len(frame.column("b"))


def test_selecting_keeps_only_what_was_asked_for():
    frame = Frame.from_rows(_rows(4)).select("x", "y")
    assert frame.names == ["x", "y"]


def test_selecting_an_absent_column_gives_an_empty_one():
    assert Frame.from_rows(_rows(3)).select("nope").column("nope") == []


def test_filtering_keeps_whole_rows():
    frame = Frame.from_rows(_rows(10)).filter(lambda row: row["x"] >= 5)
    assert frame.column("x") == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert frame.column("y") == [10.0, 12.0, 14.0, 16.0, 18.0]


def test_rows_round_trip_through_a_frame():
    rows = _rows(6)
    assert Frame.from_rows(rows).rows() == rows


@pytest.mark.parametrize("bad", [None, "text", float("nan"), float("inf"), True])
def test_a_row_with_an_unusable_number_is_dropped(bad):
    """Missing telemetry is the normal case. A correlation computed through
    ``None`` would be a correlation with the recording schedule."""
    frame = Frame.from_rows([{"x": 1.0}, {"x": bad}, {"x": 3.0}]).numeric("x")
    assert frame.column("x") == [1.0, 3.0]


def test_a_boolean_is_not_a_measurement():
    """``True`` is an ``int`` in Python and would arrive as 1.0 in a regression,
    where it would be indistinguishable from a real reading of one."""
    assert Frame.from_rows([{"x": True}, {"x": 2.0}]).numeric("x").column("x") == [2.0]


# ── lag: the operation that must not see the future ──────────────────

def test_lagging_shifts_the_column_back_in_time():
    frame = Frame.from_rows(_rows(5)).lag("x", 1)
    assert frame.column("x") == [1.0, 2.0, 3.0, 4.0]
    assert frame.column("x_lag1") == [0.0, 1.0, 2.0, 3.0]


def test_every_lagged_row_pairs_a_value_with_its_own_past():
    """The property the whole design rests on: at each surviving row the lagged
    column holds the value from ``periods`` rows earlier — never later."""
    frame = Frame.from_rows(_rows(20)).lag("x", 3)
    for current, past in zip(frame.column("x"), frame.column("x_lag3")):
        assert past == current - 3


def test_lagging_shortens_the_frame_from_the_front():
    """Not padded. A padded frame would hand the model a value that did not
    exist yet, which is how a series is made to look predictable."""
    frame = Frame.from_rows(_rows(10)).lag("x", 4)
    assert len(frame) == 6
    assert frame.column("tick")[0] == 4


def test_lagging_by_zero_copies_the_column():
    frame = Frame.from_rows(_rows(5)).lag("x", 0)
    assert frame.column("x_lag0") == frame.column("x")
    assert len(frame) == 5


def test_lagging_past_the_end_leaves_nothing_rather_than_wrapping():
    frame = Frame.from_rows(_rows(3)).lag("x", 10)
    assert len(frame) == 0
    assert "x_lag10" in frame.columns


def test_a_lagged_column_can_be_named():
    frame = Frame.from_rows(_rows(5)).lag("x", 2, as_name="earlier")
    assert frame.column("earlier") == [0.0, 1.0, 2.0]


# ── zscore and join ──────────────────────────────────────────────────

def test_standardising_centres_and_scales():
    frame = Frame.from_rows([{"x": value} for value in (2.0, 4.0, 6.0)]).zscore("x")
    values = frame.column("x_z")
    assert values[1] == pytest.approx(0.0)
    assert values[0] == pytest.approx(-values[2])


def test_a_constant_column_standardises_to_zeros_not_to_nothing():
    """Division by a zero spread would put NaN into every downstream fit."""
    frame = Frame.from_rows([{"x": 5.0}] * 4).zscore("x")
    assert frame.column("x_z") == [0.0, 0.0, 0.0, 0.0]


def test_standardising_an_empty_column_is_empty():
    assert Frame.from_rows([]).zscore("x").column("x_z") == []


def test_joining_on_tick_keeps_only_ticks_present_in_both():
    """Inner, not outer: an outer join invents rows where one side has no
    reading, and a model fitted through invented rows is fitted to the join."""
    left = Frame.from_rows([{"tick": 1, "a": 1.0}, {"tick": 2, "a": 2.0}])
    right = Frame.from_rows([{"tick": 2, "b": 20.0}, {"tick": 3, "b": 30.0}])
    joined = left.join_on_tick(right)
    assert joined.column("tick") == [2]
    assert joined.column("a") == [2.0] and joined.column("b") == [20.0]


def test_a_join_comes_back_in_tick_order():
    left = Frame.from_rows([{"tick": 3, "a": 1.0}, {"tick": 1, "a": 2.0}])
    right = Frame.from_rows([{"tick": 1, "b": 1.0}, {"tick": 3, "b": 2.0}])
    assert left.join_on_tick(right).column("tick") == [1, 3]


# ── the pool: a schema is mandatory ──────────────────────────────────

def test_a_dataset_must_declare_its_columns(pool):
    assert pool.register("metrics", {"x": "float"}) is True
    assert pool.register("nothing", {}) is False
    assert pool.register("", {"x": "float"}) is False


def test_a_column_type_outside_the_declared_set_is_refused(pool):
    assert pool.register("metrics", {"x": "complex"}) is False
    assert "metrics" not in pool.schemas


def test_appending_to_an_unregistered_dataset_is_refused(pool):
    assert pool.append("ghost", {"x": 1.0}) is False
    assert pool.rejected == 1


def test_a_column_nobody_declared_does_not_enter_the_data(pool):
    """The count of variables is what the correction is computed against, so a
    schema that widened itself would make that count a fiction."""
    pool.register("metrics", {"x": "float"})
    pool.append("metrics", {"x": 1.0, "smuggled": 99.0})
    assert pool.frame("metrics").names == ["x"]


def test_a_row_with_nothing_declared_in_it_is_refused(pool):
    pool.register("metrics", {"x": "float"})
    assert pool.append("metrics", {"other": 1.0}) is False


def test_a_dataset_is_capped_and_drops_the_oldest(tmp_path):
    pool = DataPool(tmp_path / "d", max_rows=10)
    pool.register("metrics", {"x": "float"})
    for index in range(25):
        pool.append("metrics", {"x": float(index)})
    assert pool.row_count("metrics") == 10
    assert pool.frame("metrics").column("x")[0] == 15.0


def test_the_default_cap_is_the_documented_one(tmp_path):
    assert DataPool(tmp_path / "d").max_rows == MAX_ROWS


# ── variables and controllability ────────────────────────────────────

def test_a_variable_is_controllable_only_if_it_was_declared_so(pool):
    """Controllability decides whether an experiment may *set* the variable.
    A variable that could declare itself controllable would be a safety gate
    written by the thing it constrains."""
    pool.register("genes", {"explore_bonus": "float", "brier": "float"},
                  controllable=["explore_bonus"])
    by_name = {spec.name: spec for spec in pool.variables("genes")}
    assert by_name["explore_bonus"].controllable is True
    assert by_name["brier"].controllable is False


def test_variables_come_back_in_a_stable_order(pool):
    pool.register("b", {"z": "float", "a": "float"}, source="two")
    pool.register("a", {"m": "float"}, source="one")
    names = [(spec.source, spec.name) for spec in pool.variables()]
    assert names == sorted(names)


def test_a_variable_spec_serialises(pool):
    spec = VariableSpec("x", "float", "ms", "telemetry", True)
    assert spec.as_dict()["controllable"] is True


# ── telemetry ingestion ──────────────────────────────────────────────

class _Series:
    def __init__(self, rows):
        self._rows = rows

    def rows(self):
        return list(self._rows)


class _Telemetry:
    def __init__(self, data):
        self.data = data

    def series(self, metric, window=None):
        return _Series(self.data.get(metric, []))


def test_ingesting_telemetry_aligns_series_by_tick(pool):
    telemetry = _Telemetry({
        "a": [{"tick": 1, "value": 1.0}, {"tick": 2, "value": 2.0}],
        "b": [{"tick": 1, "value": 10.0}, {"tick": 2, "value": 20.0}],
    })
    assert pool.ingest_telemetry(telemetry, ["a", "b"]) == 2
    frame = pool.frame("telemetry")
    assert frame.column("a") == [1.0, 2.0] and frame.column("b") == [10.0, 20.0]


def test_a_tick_missing_one_metric_is_dropped_not_carried_forward(pool):
    """Carrying a stale value forward manufactures exactly the autocorrelation
    the scan is looking for."""
    telemetry = _Telemetry({
        "a": [{"tick": 1, "value": 1.0}, {"tick": 2, "value": 2.0}],
        "b": [{"tick": 1, "value": 10.0}],
    })
    pool.ingest_telemetry(telemetry, ["a", "b"])
    assert pool.frame("telemetry").column("tick") == [1]


def test_a_non_numeric_reading_is_not_ingested(pool):
    telemetry = _Telemetry({"a": [{"tick": 1, "value": "up"},
                                  {"tick": 2, "value": 2.0}]})
    pool.ingest_telemetry(telemetry, ["a"])
    assert pool.frame("telemetry").column("a") == [2.0]


def test_a_telemetry_source_that_raises_does_not_stop_the_ingest(pool):
    class _Broken:
        def series(self, metric, window=None):
            if metric == "bad":
                raise RuntimeError("the store is gone")
            return _Series([{"tick": 1, "value": 1.0}])

    assert pool.ingest_telemetry(_Broken(), ["bad"]) == 0


def test_ingesting_no_metrics_does_nothing(pool):
    assert pool.ingest_telemetry(_Telemetry({}), []) == 0


# ── persistence ──────────────────────────────────────────────────────

def test_a_pool_reloads_its_schema_and_its_data(tmp_path):
    pool = DataPool(tmp_path / "d")
    pool.register("metrics", {"x": "float"}, source="test",
                  controllable=["x"])
    pool.append("metrics", {"x": 1.5})
    assert pool.save() is True

    reloaded = DataPool(tmp_path / "d")
    assert reloaded.frame("metrics").column("x") == [1.5]
    assert reloaded.variables("metrics")[0].controllable is True
    assert reloaded.sources["metrics"] == "test"


def test_a_missing_store_loads_as_an_empty_pool(tmp_path):
    assert DataPool(tmp_path / "absent").datasets() == []


def test_a_corrupt_store_does_not_stop_the_pool(tmp_path):
    directory = tmp_path / "d"
    directory.mkdir(parents=True)
    (directory / "pool.json").write_text("{not json", encoding="utf-8")
    assert DataPool(directory).datasets() == []


def test_a_store_whose_schemas_are_not_an_object_is_ignored(tmp_path):
    import json

    directory = tmp_path / "d"
    directory.mkdir(parents=True)
    (directory / "pool.json").write_text(
        json.dumps({"schema_version": 2, "schemas": ["not", "a", "map"]}),
        encoding="utf-8")
    assert DataPool(directory).datasets() == []


def test_the_status_counts_what_is_held(pool):
    pool.register("metrics", {"x": "float"})
    pool.append("metrics", {"x": 1.0})
    status = pool.status()
    assert status["datasets"] == 1 and status["rows"]["metrics"] == 1
    assert status["variables"] == 1


def test_ingesting_twice_does_not_duplicate_a_single_row(pool):
    """Ingestion is periodic and the series it reads is cumulative.

    Without a watermark every ingest re-appends everything. The pool would grow
    without bound, and — far worse — each observation would be counted several
    times: duplicated rows shrink every p-value in the scan without adding any
    evidence, which is exactly the failure the false-discovery control exists to
    prevent. It would look like the engine getting better at finding laws the
    longer it ran.
    """
    telemetry = _Telemetry({
        "a": [{"tick": tick, "value": float(tick)} for tick in range(10)],
        "b": [{"tick": tick, "value": float(tick) * 2} for tick in range(10)],
    })
    assert pool.ingest_telemetry(telemetry, ["a", "b"]) == 10
    assert pool.ingest_telemetry(telemetry, ["a", "b"]) == 0
    assert pool.row_count("telemetry") == 10
    assert pool.frame("telemetry").column("tick") == list(range(10))


def test_ingesting_picks_up_only_what_is_new(pool):
    rows_a = [{"tick": tick, "value": float(tick)} for tick in range(5)]
    rows_b = [{"tick": tick, "value": float(tick)} for tick in range(5)]
    telemetry = _Telemetry({"a": rows_a, "b": rows_b})
    pool.ingest_telemetry(telemetry, ["a", "b"])

    rows_a.extend({"tick": tick, "value": float(tick)} for tick in range(5, 8))
    rows_b.extend({"tick": tick, "value": float(tick)} for tick in range(5, 8))
    assert pool.ingest_telemetry(telemetry, ["a", "b"]) == 3
    assert pool.frame("telemetry").column("tick") == list(range(8))


# ── the schema declaration is read, not assumed ──────────────────────

def test_a_declared_type_and_unit_are_kept(pool):
    """A column may be declared as a bare type string or as a mapping. The
    mapping form is the only way a unit reaches the variable list, and a unit is
    what tells an operator whether "duration 40" is milliseconds or ticks."""
    pool.register("metrics", {"latency": {"kind": "int", "unit": "ms"},
                              "score": "float"}, source="test")
    by_name = {spec.name: spec for spec in pool.variables("metrics")}
    assert by_name["latency"].kind == "int"
    assert by_name["latency"].unit == "ms"
    assert by_name["score"].kind == "float" and by_name["score"].unit == ""


def test_a_variable_spec_is_frozen():
    """It is a declaration. One that could be edited afterwards would let a
    variable become controllable after the whitelist had already been checked
    against it."""
    spec = VariableSpec("x", "float", "ms", "telemetry", False)
    with pytest.raises(Exception):
        spec.controllable = True


def test_a_variable_is_not_controllable_unless_something_says_so():
    """The default has to be the safe one: a variable nobody has thought about
    is a variable no experiment may set."""
    assert VariableSpec("x").controllable is False


def test_a_stored_spec_without_the_flag_loads_as_not_controllable(tmp_path):
    """Same default, one layer down. A store written before the flag existed
    must not come back granting permission nobody wrote."""
    import json

    directory = tmp_path / "d"
    directory.mkdir(parents=True)
    (directory / "pool.json").write_text(json.dumps({
        "schema_version": 2,
        "schemas": {"metrics": {"x": {"name": "x", "kind": "float"}}},
        "data": {"metrics": [{"x": 1.0}]},
    }), encoding="utf-8")
    assert DataPool(directory).variables("metrics")[0].controllable is False


def test_the_default_directory_is_under_the_discovery_store(tmp_path, monkeypatch):
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=False)
    pool = DataPool()
    assert pool.directory.name == "datasets"
    assert pool.directory.parent.name == "discovery"


# ── numeric() looks only where it was told to ────────────────────────

def test_only_the_named_columns_decide_whether_a_row_is_usable():
    """A frame carries every metric that was ingested; a hypothesis is about
    two of them. Dropping rows because some unrelated column had a gap would
    throw away most of the data for a reason that has nothing to do with the
    question being asked."""
    frame = Frame.from_rows([
        {"a": 1.0, "b": 2.0, "unrelated": None},
        {"a": 3.0, "b": 4.0, "unrelated": "text"},
    ]).numeric("a", "b")
    assert frame.column("a") == [1.0, 3.0]


def test_with_no_columns_named_every_column_has_to_be_usable():
    frame = Frame.from_rows([
        {"a": 1.0, "b": 2.0},
        {"a": 3.0, "b": None},
    ]).numeric()
    assert frame.column("a") == [1.0]


# ── zscore, to the number ────────────────────────────────────────────

def test_standardising_matches_the_definition():
    """Over (2, 4, 6): mean 4, deviations (−2, 0, 2), squares summing to 8, and
    a *population* spread of √(8/3) = 1.63299…, so the z-scores are ∓1.224745.

    Population and not sample spread, and the difference is the point of
    checking against arithmetic rather than a shape: the sample version would
    give exactly ∓1 here, which looks tidier and is a different statistic.
    """
    frame = Frame.from_rows([{"x": v} for v in (2.0, 4.0, 6.0)]).zscore("x")
    spread = (8.0 / 3.0) ** 0.5
    assert frame.column("x_z") == pytest.approx([-2.0 / spread, 0.0, 2.0 / spread])
    assert frame.column("x_z")[0] == pytest.approx(-1.224745, abs=1e-6)


def test_standardising_uses_the_population_spread():
    """Four values one apart: mean 2.5, population sd = √1.25 = 1.118…"""
    frame = Frame.from_rows([{"x": v} for v in (1.0, 2.0, 3.0, 4.0)]).zscore("x")
    expected = [(v - 2.5) / (1.25 ** 0.5) for v in (1.0, 2.0, 3.0, 4.0)]
    assert frame.column("x_z") == pytest.approx(expected)


# ── the frame a dataset comes back as ────────────────────────────────

def test_a_frame_carries_every_declared_column_even_when_unfilled(pool):
    """The schema is what says which variables exist. A frame that only showed
    the columns that happened to have values would make the scan's variable
    count depend on which rows arrived."""
    pool.register("metrics", {"tick": "int", "a": "float", "b": "float"})
    pool.append("metrics", {"tick": 1, "a": 1.0})
    frame = pool.frame("metrics")
    assert frame.names == ["a", "b", "tick"]
    assert frame.column("b") == [None]


def test_the_tick_column_comes_first(pool):
    """Everything downstream joins and lags on it."""
    pool.register("metrics", {"zeta": "float", "tick": "int", "alpha": "float"})
    pool.append("metrics", {"tick": 1, "zeta": 1.0, "alpha": 2.0})
    assert list(pool.frame("metrics").columns)[0] == "tick"
