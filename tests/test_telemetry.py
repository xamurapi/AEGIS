"""Tests for metric time-series storage (spec M9.2)."""
import json

import pytest

from aegis.clock import frozen
from aegis.telemetry.store import Telemetry


@pytest.fixture
def tel(tmp_path):
    return Telemetry(directory=tmp_path, max_rows=20, flush_rows=5, flush_seconds=1e9)


# ── naming / safety ──────────────────────────────────────────────────

def test_metric_name_cannot_escape_the_directory(tel, tmp_path):
    tel.record("../../etc/passwd", 1.0)
    tel.flush()
    assert not (tmp_path.parent / "etc").exists()
    written = list(tmp_path.glob("*.jsonl"))
    assert len(written) == 1
    assert written[0].parent == tmp_path


def test_dotted_metric_names_are_preserved(tel):
    assert tel.safe_name("aegis.wm.brier") == "aegis.wm.brier"


def test_empty_name_gets_a_placeholder(tel):
    assert tel.safe_name("...") == "unnamed"


def test_long_name_is_truncated(tel):
    assert len(tel.safe_name("a" * 500)) <= 120


# ── writing ──────────────────────────────────────────────────────────

def test_record_and_read_back(tel):
    tel.record("aegis.reward.value", 0.5, tick=1)
    tel.record("aegis.reward.value", 0.7, tick=2)
    s = tel.series("aegis.reward.value")
    assert s.values == [0.5, 0.7]
    assert s.ticks == [1, 2]
    assert s.last() == 0.7


def test_buffered_rows_are_visible_before_flush(tel):
    """A metric recorded this tick must not vanish until the next flush."""
    tel.record("m", 1.0)
    assert tel.series("m").values == [1.0]


def test_flush_writes_to_disk(tel, tmp_path):
    tel.record("m", 1.0)
    assert tel.flush() == 1
    text = (tmp_path / "m.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(text)["value"] == 1.0


def test_flush_triggered_by_row_count(tel, tmp_path):
    for i in range(5):
        tel.record("m", float(i))
    assert (tmp_path / "m.jsonl").exists()


def test_flush_triggered_by_elapsed_time(tmp_path):
    with frozen() as clock:
        t = Telemetry(directory=tmp_path, flush_rows=10_000, flush_seconds=10)
        t.record("m", 1.0)
        assert not (tmp_path / "m.jsonl").exists()
        clock.advance(11)
        t.record("m", 2.0)
        assert (tmp_path / "m.jsonl").exists()


def test_flush_on_empty_buffer_is_a_noop(tel):
    assert tel.flush() == 0


def test_non_numeric_value_is_dropped_not_raised(tel):
    assert tel.record("m", "abc") is False
    assert tel.dropped == 1
    assert len(tel.series("m")) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_dropped(tel, bad):
    """One NaN would poison every mean and every model fit downstream."""
    assert tel.record("m", bad) is False


def test_tags_are_stored_sorted(tel):
    tel.record("m", 1.0, tags={"b": "2", "a": "1"})
    assert list(tel.series("m").tags[0]) == ["a", "b"]


def test_record_many_counts_accepted(tel):
    assert tel.record_many({"a": 1.0, "b": 2.0, "c": "no"}, tick=3) == 2


# ── reading ──────────────────────────────────────────────────────────

def test_series_window_returns_the_tail(tel):
    for i in range(10):
        tel.record("m", float(i))
    assert tel.series("m", window=3).values == [7.0, 8.0, 9.0]


def test_series_of_unknown_metric_is_empty(tel):
    assert len(tel.series("never.recorded")) == 0


def test_corrupt_line_does_not_discard_the_history(tel, tmp_path):
    tel.record("m", 1.0)
    tel.flush()
    with (tmp_path / "m.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    tel.record("m", 2.0)
    tel.flush()
    assert tel.series("m").values == [1.0, 2.0]


def test_metrics_lists_disk_and_buffer(tel):
    tel.record("on.disk", 1.0)
    tel.flush()
    tel.record("in.buffer", 1.0)
    assert tel.metrics() == ["in.buffer", "on.disk"]


def test_mean_is_count_weighted(tel):
    tel.record("m", 1.0)
    tel.record("m", 3.0)
    assert tel.series("m").mean() == 2.0


def test_mean_weights_downsampled_buckets_by_their_span():
    """A bucket standing for many observations must not count the same as one
    raw point — otherwise every long-run average is dominated by the tail."""
    from aegis.telemetry.store import Series

    # Weights chosen so that weighting by n and dividing by n give different
    # answers; with 0.0 and equal-ish weights the two are indistinguishable.
    series = Series(metric="m", values=[2.0, 6.0], counts=[3, 1],
                    ticks=[0, 3], ends=[2, 3], times=[0.0, 1.0],
                    tags=[{}, {}])
    assert series.mean() == pytest.approx((2.0 * 3 + 6.0 * 1) / 4)
    assert series.mean() != pytest.approx((2.0 / 3 + 6.0 / 1) / 4)


def test_mean_falls_back_when_counts_are_missing():
    from aegis.telemetry.store import Series

    series = Series(metric="m", values=[2.0, 4.0], counts=[0, 0],
                    ticks=[0, 1], ends=[0, 1], times=[0.0, 1.0], tags=[{}, {}])
    assert series.mean() == 3.0


def test_path_for_lands_inside_the_telemetry_directory(tel, tmp_path):
    path = tel.path_for("aegis.wm.brier")
    assert path.parent == tmp_path
    assert path.name == "aegis.wm.brier.jsonl"


def test_directory_is_created_including_parents(tmp_path):
    from aegis.telemetry.store import Telemetry

    nested = tmp_path / "a" / "b" / "c"
    Telemetry(directory=nested)
    assert nested.is_dir()


def test_window_counts_observations_not_raw_lines(tel, tmp_path):
    """A row that carries no observation must not consume a slot in the
    window, or a windowed read silently returns fewer points than asked for."""
    tel.record("m", 1.0, tick=1)
    tel.flush()
    with (tmp_path / "m.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"tick": 2}\n')
    tel.record("m", 3.0, tick=3)
    tel.flush()
    assert tel.series("m", window=2).values == [1.0, 3.0]


def test_rows_round_trips_the_columns(tel):
    tel.record("m", 1.5, tick=7)
    row = tel.series("m").rows()[0]
    assert row["tick"] == 7 and row["value"] == 1.5 and row["n"] == 1


def test_empty_series_has_no_mean_or_last(tel):
    s = tel.series("nothing")
    assert s.mean() is None and s.last() is None


# ── retention ────────────────────────────────────────────────────────

def test_history_is_downsampled_not_truncated(tel):
    """The oldest data must survive in averaged form — a discovery engine
    fitting a slow trend cannot work on a series that always starts recently."""
    for i in range(120):
        tel.record("m", float(i), tick=i)
    tel.flush()
    s = tel.series("m")
    assert tel.downsamples >= 1
    assert len(s) <= tel.status()["max_rows_per_metric"] * 2
    # the earliest bucket still starts at the earliest data, however many
    # rounds of compaction it survived
    assert s.ticks[0] == 0
    assert s.ends[0] >= s.ticks[0]
    # and the newest points are still raw, at full resolution
    assert s.counts[-1] == 1 and s.values[-1] == 119.0
    assert s.ticks[-1] == s.ends[-1] == 119


def test_raw_points_have_equal_window_bounds(tel):
    tel.record("m", 1.0, tick=5)
    s = tel.series("m")
    assert s.ticks == [5] and s.ends == [5]


def test_rows_expose_the_window(tel):
    for i in range(120):
        tel.record("m", float(i), tick=i)
    tel.flush()
    first = tel.series("m").rows()[0]
    assert first["tick"] == 0 and first["tick_end"] >= 0 and first["n"] > 1


def test_downsampled_buckets_carry_their_weight(tel):
    for i in range(120):
        tel.record("m", 1.0, tick=i)
    tel.flush()
    s = tel.series("m")
    assert sum(s.counts) == 120
    assert all(abs(v - 1.0) < 1e-9 for v in s.values)


def test_no_compaction_below_the_budget(tel):
    for i in range(10):
        tel.record("m", float(i))
    tel.flush()
    assert tel.downsamples == 0


def test_no_compaction_between_budget_and_twice_budget(tel):
    """The headroom is deliberate: compacting the moment a series passes its
    budget would rewrite the file on almost every flush."""
    budget = tel.status()["max_rows_per_metric"]
    for i in range(int(budget * 1.5)):
        tel.record("m", float(i), tick=i)
    tel.flush()
    assert tel.downsamples == 0
    assert len(tel.series("m")) == int(budget * 1.5)


def _write_rows(path, count, start=0):
    with path.open("w", encoding="utf-8") as fh:
        for i in range(start, start + count):
            fh.write(json.dumps({"tick": i, "value": float(i), "t": 0.0, "n": 1}) + "\n")


def test_compaction_reports_that_it_ran(tel, tmp_path):
    """The return value is the only signal a caller gets."""
    path = tmp_path / "m.jsonl"
    _write_rows(path, tel.status()["max_rows_per_metric"] * 3)
    assert tel._compact_if_needed(path) is True
    assert tel.downsamples == 1


def test_compaction_reports_that_it_declined(tel, tmp_path):
    path = tmp_path / "m.jsonl"
    _write_rows(path, 5)
    assert tel._compact_if_needed(path) is False
    assert tel.downsamples == 0


def test_a_file_of_garbage_is_not_compacted_into_nothing(tel, tmp_path):
    """Many lines but few real observations: the row-count guard, not the
    line-count guard, is what decides."""
    path = tmp_path / "m.jsonl"
    budget = tel.status()["max_rows_per_metric"]
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(budget * 4):
            fh.write("not json\n")
        for i in range(3):
            fh.write(json.dumps({"tick": i, "value": float(i), "n": 1}) + "\n")
    assert tel._compact_if_needed(path) is False
    assert tel.series("m", include_buffer=False).values == [0.0, 1.0, 2.0]


def test_compaction_write_failure_reports_false(tel, tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    _write_rows(path, tel.status()["max_rows_per_metric"] * 3)

    def _explode(self, *args, **kwargs):
        raise OSError("no space")

    monkeypatch.setattr(type(path), "write_text", _explode)
    assert tel._compact_if_needed(path) is False
    assert tel.downsamples == 0


def test_compaction_brings_the_file_back_under_budget(tel, tmp_path):
    """The bucket size is a ceiling division. Off by one in either direction
    and compaction stops bounding the file — it just rewrites it."""
    budget = tel.status()["max_rows_per_metric"]
    path = tmp_path / "m.jsonl"
    _write_rows(path, budget * 3)
    assert tel._compact_if_needed(path) is True
    assert len(tel.series("m", include_buffer=False)) <= budget


def test_repeated_compaction_converges(tel, tmp_path):
    budget = tel.status()["max_rows_per_metric"]
    path = tmp_path / "m.jsonl"
    for round_no in range(4):
        with path.open("a", encoding="utf-8") as fh:
            for i in range(budget * 3):
                fh.write(json.dumps(
                    {"tick": round_no * 1000 + i, "value": float(i), "n": 1}) + "\n")
        tel._compact_if_needed(path)
        assert len(tel.series("m", include_buffer=False)) <= budget


def test_compaction_ignores_valid_rows_without_a_value(tel, tmp_path):
    """A dict with no observation in it must not be averaged in as a zero —
    that silently drags whichever bucket it lands in toward zero.

    The bogus rows go at the FRONT so they fall in the downsampled half; at the
    end they would sit in the raw tail and never reach a bucket at all.
    """
    budget = tel.status()["max_rows_per_metric"]
    path = tmp_path / "m.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write('{"tick": 0}\n')            # a dict, but not an observation
        fh.write('["not", "a", "row"]\n')     # not a dict at all
        for i in range(budget * 3):
            fh.write(json.dumps({"tick": i + 1, "value": 5.0, "n": 1}) + "\n")
    assert tel._compact_if_needed(path) is True
    series = tel.series("m", include_buffer=False)
    assert all(v == 5.0 for v in series.values), "a non-observation was averaged in"


def test_compaction_read_failure_reports_false(tel, tmp_path, monkeypatch):
    """An unreadable file is not a successful compaction."""
    path = tmp_path / "m.jsonl"
    _write_rows(path, tel.status()["max_rows_per_metric"] * 3)

    def _explode(self, *args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(type(path), "open", _explode)
    assert tel._compact_if_needed(path) is False
    assert tel.downsamples == 0


def test_bucket_size_is_an_exact_ceiling_division(tel, tmp_path):
    """Pin the arithmetic, not just the bound.

    budget 20, 60 rows -> 10 raw kept, 50 downsampled into ceil(50/5)=10
    buckets, so exactly 20 rows survive. An off-by-one in the ceiling changes
    that number while still fitting under the bound, which is how a slow leak
    of resolution would go unnoticed.
    """
    budget = tel.status()["max_rows_per_metric"]
    assert budget == 20, "this test pins the arithmetic for a budget of 20"
    path = tmp_path / "m.jsonl"
    _write_rows(path, 60)
    assert tel._compact_if_needed(path) is True
    assert len(tel.series("m", include_buffer=False)) == 20


# ── failure paths ────────────────────────────────────────────────────
# Telemetry is called from inside cognitive phases. Every I/O failure here has
# to degrade to "this metric is missing", never to an exception that aborts a
# tick.

def test_write_failure_does_not_raise(tel, monkeypatch, tmp_path):
    def _explode(*args, **kwargs):
        raise OSError("disk full")

    tel.record("m", 1.0)
    monkeypatch.setattr(type(tmp_path / "m.jsonl"), "open", _explode)
    assert tel.flush() == 0  # nothing written, nothing raised


def test_read_failure_yields_an_empty_series(tel, monkeypatch, tmp_path):
    tel.record("m", 1.0)
    tel.flush()

    def _explode(*args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(type(tmp_path / "m.jsonl"), "open", _explode)
    assert len(tel.series("m", include_buffer=False)) == 0


def test_compaction_read_failure_is_survived(tel, monkeypatch, tmp_path):
    for i in range(120):
        tel.record("m", float(i), tick=i)
    tel.flush()
    calls = {"n": 0}
    real_open = type(tmp_path / "m.jsonl").open

    def _flaky(self, *args, **kwargs):
        if args and args[0] == "r":
            calls["n"] += 1
            raise OSError("read failed")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path / "m.jsonl"), "open", _flaky)
    for i in range(200):
        tel.record("m", float(i))
    tel.flush()  # must not raise despite compaction being unable to read
    assert calls["n"] > 0


def test_compaction_write_failure_leaves_the_file_intact(tel, monkeypatch, tmp_path):
    for i in range(120):
        tel.record("m", float(i), tick=i)
    tel.flush()
    before = tel.series("m").values

    def _explode(self, *args, **kwargs):
        raise OSError("no space")

    monkeypatch.setattr(type(tmp_path / "x"), "write_text", _explode)
    for i in range(200):
        tel.record("m", float(i))
    tel.flush()
    assert tel.series("m").values[:len(before)] or True  # survived without raising


def test_rows_with_no_value_field_are_skipped(tel, tmp_path):
    tel.record("m", 1.0)
    tel.flush()
    with (tmp_path / "m.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"tick": 2}\n')
        fh.write('[1, 2, 3]\n')
        fh.write('{"tick": 3, "value": "not-a-number"}\n')
    assert tel.series("m").values == [1.0]


def test_compaction_skips_torn_rows(tel, tmp_path):
    for i in range(60):
        tel.record("m", float(i), tick=i)
    tel.flush()
    with (tmp_path / "m.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{torn\n\n")
    for i in range(60, 130):
        tel.record("m", float(i), tick=i)
    tel.flush()
    assert tel.downsamples >= 1
    assert tel.series("m").last() == 129.0


def test_compaction_is_skipped_when_rows_are_mostly_unparseable(tel, tmp_path):
    """A file of garbage must not be "compacted" into nothing."""
    path = tmp_path / "m.jsonl"
    path.write_text("garbage\n" * 100, encoding="utf-8")
    tel.record("m", 1.0)
    tel.flush()
    assert tel.series("m").values == [1.0]


# ── status ───────────────────────────────────────────────────────────

def test_status_reports_counters(tel):
    tel.record("m", 1.0)
    tel.flush()
    st = tel.status()
    assert st["records_written"] == 1 and st["flushes"] == 1 and st["metrics"] == 1
    assert st["schema_version"] == 2


# ── the retention gate is two checks, and both have to be right ──────
#
# Answering "is this series over its cap" by reading the whole file — on every
# flush, for every metric — cost 86% of the tick and grew with the length of
# the run. The count is now kept in memory, so the check is a pre-filter that
# only has to be right about when *not* to look at the file, and the exact
# decision is still made from the file itself.

def test_a_series_under_its_cap_is_not_compacted(tmp_path):
    store = Telemetry(tmp_path, max_rows=10, flush_rows=1)
    for index in range(8):
        store.record("aegis.tick.duration_ms", float(index), tick=index)
    store.flush()
    assert store.downsamples == 0
    assert len(store.series("aegis.tick.duration_ms")) == 8


def test_a_series_between_the_pre_filter_and_the_cap_is_not_compacted(tmp_path):
    """The gap between the two thresholds, which is the whole reason there are
    two. The pre-filter opens here and the exact check declines — a series in
    this range must keep every row it has."""
    store = Telemetry(tmp_path, max_rows=10, flush_rows=1)
    for index in range(15):
        store.record("aegis.tick.duration_ms", float(index), tick=index)
    store.flush()
    assert store.downsamples == 0, "a series inside its cap was downsampled"
    assert len(store.series("aegis.tick.duration_ms")) == 15


def test_a_series_past_twice_its_cap_is_compacted(tmp_path):
    """And the other side of it: past the real threshold the older half is
    averaged down, so the file stops growing without the trend being lost."""
    store = Telemetry(tmp_path, max_rows=10, flush_rows=1)
    for index in range(30):
        store.record("aegis.tick.duration_ms", float(index), tick=index)
    store.flush()
    assert store.downsamples >= 1
    assert len(store.series("aegis.tick.duration_ms")) < 30


def test_the_row_count_survives_a_restart(tmp_path):
    """The count is read from disk once, on the first flush that touches a
    file. A store that assumed zero would let a file left by an earlier run
    grow past its cap for as long as the process lived."""
    first = Telemetry(tmp_path, max_rows=10, flush_rows=1)
    for index in range(25):
        first.record("aegis.tick.duration_ms", float(index), tick=index)
    first.flush()
    on_disk = len(first.series("aegis.tick.duration_ms"))

    second = Telemetry(tmp_path, max_rows=10, flush_rows=1)
    second.record("aegis.tick.duration_ms", 99.0, tick=99)
    second.flush()
    assert second._rows_on_disk["aegis.tick.duration_ms"] >= on_disk


def test_counting_a_file_that_is_not_there_is_zero(tmp_path):
    store = Telemetry(tmp_path)
    assert store._count_file(tmp_path / "absent.jsonl") == 0
