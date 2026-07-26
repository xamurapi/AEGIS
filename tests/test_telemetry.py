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


# ── status ───────────────────────────────────────────────────────────

def test_status_reports_counters(tel):
    tel.record("m", 1.0)
    tel.flush()
    st = tel.status()
    assert st["records_written"] == 1 and st["flushes"] == 1 and st["metrics"] == 1
    assert st["schema_version"] == 2
