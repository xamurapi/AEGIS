"""State survives being killed mid-write (spec §VII.5).

The acceptance criterion is blunt: after a ``SIGTERM`` at an arbitrary moment,
**every** state file is valid. A process cannot choose when it is killed, so the
guarantee cannot come from finishing carefully — it has to come from never
having an invalid file on disk in the first place.

Two mechanisms carry it, and this file tests both against the interruption they
exist for:

* **Atomic replace** for whole-file stores. The new content is written to a
  temp file beside the target and renamed over it. A kill before the rename
  leaves the old complete file; a kill after it leaves the new complete file.
  There is no moment at which the target is half-written.
* **Line-at-a-time append** for JSONL logs, where a kill can genuinely tear the
  last line — so the reader is required to skip a bad line and keep the rest,
  rather than treat the file as lost.

The interruption is simulated by making the write fail exactly where a kill
would land. That is the honest way to test it: a real signal cannot be aimed at
a particular instruction, and a test that sent one would be testing the
scheduler.
"""
import json

import pytest

from aegis._atomic import atomic_write_text
from aegis.store.migrations import read_store, write_store


class _Killed(Exception):
    """Stands in for the process ending between two instructions."""


# ── whole-file stores ────────────────────────────────────────────────

def test_a_kill_before_the_rename_leaves_the_old_file_whole(tmp_path, monkeypatch):
    """The window that matters. Everything the new content needs is already on
    disk in the temp file, and the target has not been touched at all."""
    from pathlib import Path

    target = tmp_path / "state.json"
    original = json.dumps({"schema_version": 2, "champion": "keep me"})
    target.write_text(original, encoding="utf-8")

    def _die(self, *args, **kwargs):
        raise _Killed()

    monkeypatch.setattr(Path, "replace", _die)
    with pytest.raises(_Killed):
        atomic_write_text(target, json.dumps({"schema_version": 2,
                                              "champion": "new"}))

    assert target.read_text(encoding="utf-8") == original
    assert json.loads(target.read_text(encoding="utf-8"))["champion"] == "keep me"


def test_a_kill_partway_through_the_temp_file_leaves_the_target_untouched(tmp_path,
                                                                          monkeypatch):
    """The other side of the same window. A truncated temp file is rubbish, and
    it is rubbish that nothing reads — the target still holds the last complete
    state."""
    from pathlib import Path

    target = tmp_path / "state.json"
    original = json.dumps({"schema_version": 2, "generation": 41})
    target.write_text(original, encoding="utf-8")

    real_write = Path.write_text

    def _die_halfway(self, text, *args, **kwargs):
        real_write(self, text[:len(text) // 2], encoding="utf-8")
        raise _Killed()

    monkeypatch.setattr(Path, "write_text", _die_halfway)
    with pytest.raises(_Killed):
        atomic_write_text(target, json.dumps({"schema_version": 2,
                                              "generation": 42}))

    monkeypatch.undo()
    assert json.loads(target.read_text(encoding="utf-8"))["generation"] == 41


def test_a_leftover_temp_file_does_not_break_the_next_read(tmp_path):
    """A kill leaves the temp file behind. The store must not notice."""
    target = tmp_path / "state.json"
    write_store(target, {"champion": "alive"})
    (tmp_path / "state.json.tmp").write_text("{ truncated", encoding="utf-8")

    assert read_store(target)["champion"] == "alive"


def test_a_target_that_was_never_written_reads_as_empty(tmp_path):
    """The first kill can land before anything exists. Starting empty is a
    normal outcome; refusing to start is not."""
    assert read_store(tmp_path / "never_written.json") == {"schema_version": 2}


@pytest.mark.parametrize("corruption", [
    "",                                   # killed before a single byte landed
    "{",                                  # killed after one character
    '{"schema_version": 2, "a":',         # killed mid-value
    '{"schema_version": 2, "a": 1}extra',  # two writes overlapping
    "\x00\x00\x00",                        # the file system's own idea of torn
])
def test_a_corrupt_store_loads_as_empty_rather_than_raising(tmp_path, corruption):
    """Whatever a kill leaves behind, the system keeps ticking with empty state
    rather than refusing to start. That is §3.2's rule, and it is the
    difference between losing one store and losing the process."""
    target = tmp_path / "state.json"
    target.write_text(corruption, encoding="utf-8")
    loaded = read_store(target, store="discovery_ledger")
    assert isinstance(loaded, dict)
    assert loaded.get("entries", []) == []


# ── append-only logs ─────────────────────────────────────────────────

def _torn_jsonl(path, rows, tail):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.write(tail)                # killed mid-line


def test_a_torn_last_line_does_not_cost_the_rest_of_the_history(tmp_path):
    """A JSONL append genuinely can be interrupted mid-line — there is no
    rename to hide behind. So the reader skips the bad line and keeps the rest;
    treating the file as lost would throw away everything for one byte."""
    from aegis.telemetry.store import Telemetry

    store = Telemetry(tmp_path, flush_rows=1)
    path = store.path_for("aegis.tick.duration_ms")
    _torn_jsonl(path,
                [{"tick": index, "value": float(index), "t": 0.0, "n": 1}
                 for index in range(10)],
                '{"tick": 10, "value": 1')

    series = store.series("aegis.tick.duration_ms")
    assert len(series) == 10
    assert series.values[0] == 0.0 and series.values[-1] == 9.0


def test_a_torn_line_in_the_middle_costs_only_itself(tmp_path):
    """A kill during compaction can leave a bad line anywhere, not only last."""
    from aegis.telemetry.store import Telemetry

    store = Telemetry(tmp_path, flush_rows=1)
    path = store.path_for("aegis.reward.value")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"tick": 0, "value": 1.0, "t": 0.0, "n": 1}) + "\n")
        handle.write("{ this line was torn\n")
        handle.write(json.dumps({"tick": 2, "value": 3.0, "t": 0.0, "n": 1}) + "\n")

    series = store.series("aegis.reward.value")
    assert list(series.values) == [1.0, 3.0]


def test_appending_after_a_torn_line_still_works(tmp_path):
    """The next run has to be able to keep writing to the same file."""
    from aegis.telemetry.store import Telemetry

    path_store = Telemetry(tmp_path, flush_rows=1)
    path = path_store.path_for("aegis.reward.value")
    _torn_jsonl(path, [{"tick": 0, "value": 1.0, "t": 0.0, "n": 1}],
                '{"tick": 1, "val')

    store = Telemetry(tmp_path, flush_rows=1)
    store.record("aegis.reward.value", 5.0, tick=2)
    store.flush()
    assert 5.0 in store.series("aegis.reward.value").values


# ── the whole state directory ────────────────────────────────────────

def test_every_persistent_store_survives_an_interrupted_write(tmp_path, monkeypatch):
    """The criterion as stated: *every* state file, not a chosen one.

    Each store is written, then written again with the rename disabled, and
    then read back. Whatever the second write was going to say, the file on
    disk still says what the first one did.
    """
    from pathlib import Path

    stores = {
        "world_model": {"links": {"a": {"n": 3}}},
        "policy": {"rules": [{"id": "r1", "status": "active"}]},
        "evolution": {"generation": 7, "champion": {"fitness": 0.5}},
        "discovery": {"entries": [{"id": "hyp_1", "status": "supported"}]},
        "motivation": {"budgets": {"llm_tokens": {"used": 10}}},
        "reasoning": {"strategies": {"direct": {"version": 1}}},
    }
    for name, payload in stores.items():
        assert write_store(tmp_path / f"{name}.json", payload) is True

    def _die(self, *args, **kwargs):
        raise _Killed()

    monkeypatch.setattr(Path, "replace", _die)
    for name in stores:
        with pytest.raises(_Killed):
            atomic_write_text(tmp_path / f"{name}.json", '{"schema_version": 2}')
    monkeypatch.undo()

    for name, payload in stores.items():
        loaded = read_store(tmp_path / f"{name}.json")
        for key, value in payload.items():
            assert loaded[key] == value, f"{name} lost {key}"


def test_a_write_that_fails_is_reported_rather_than_raised(tmp_path, monkeypatch):
    """A store that cannot be written must not take the tick with it. The
    caller finds out through the return value and carries on — losing a
    checkpoint is survivable, losing the process is not."""
    from pathlib import Path

    def _die(self, *args, **kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr(Path, "write_text", _die)
    assert write_store(tmp_path / "state.json", {"a": 1}) is False


# ── the newline guard, in each of its cases ──────────────────────────

def test_a_file_that_ends_properly_needs_no_separator(tmp_path):
    """The ordinary case, and the one that must not add a blank line. A
    separator written every time would put an empty row between every flush."""
    from aegis.telemetry.store import Telemetry

    path = tmp_path / "aegis.reward.value.jsonl"
    path.write_text('{"tick": 0, "value": 1.0}\n', encoding="utf-8")
    assert Telemetry._needs_newline(path) is False


def test_a_file_that_ends_mid_line_needs_one(tmp_path):
    from aegis.telemetry.store import Telemetry

    path = tmp_path / "aegis.reward.value.jsonl"
    path.write_text('{"tick": 0, "value"', encoding="utf-8")
    assert Telemetry._needs_newline(path) is True


def test_an_empty_file_needs_no_separator(tmp_path):
    """Seeking one byte back from the end of nothing is an error, and a first
    flush into a fresh file is the most ordinary thing there is."""
    from aegis.telemetry.store import Telemetry

    path = tmp_path / "aegis.reward.value.jsonl"
    path.write_text("", encoding="utf-8")
    assert Telemetry._needs_newline(path) is False


def test_a_missing_file_needs_no_separator(tmp_path):
    from aegis.telemetry.store import Telemetry

    assert Telemetry._needs_newline(tmp_path / "absent.jsonl") is False


def test_a_file_that_cannot_be_read_needs_no_separator(tmp_path, monkeypatch):
    """A telemetry write must never abort a tick, so an unreadable file is
    answered rather than raised through."""
    from pathlib import Path

    from aegis.telemetry.store import Telemetry

    path = tmp_path / "aegis.reward.value.jsonl"
    path.write_text('{"tick": 0}\n', encoding="utf-8")

    def _refuse(self, *args, **kwargs):
        raise OSError("locked by another process")

    monkeypatch.setattr(Path, "open", _refuse)
    assert Telemetry._needs_newline(path) is False


def test_the_separator_is_written_once_and_not_on_every_flush(tmp_path):
    """The guard costs a stat and a seek per metric. Doing it on every flush is
    the per-tick file IO the row counter exists to remove — it put ACT back
    over its §3.4 budget by a factor of fourteen when it was unguarded."""
    from aegis.telemetry.store import Telemetry

    store = Telemetry(tmp_path, flush_rows=1)
    path = store.path_for("aegis.reward.value")
    path.write_text('{"tick": 0, "value": 1.0}\n{"torn', encoding="utf-8")

    seen = []
    original = Telemetry._needs_newline
    Telemetry._needs_newline = staticmethod(
        lambda p: seen.append(p) or original(p))
    try:
        for tick in range(1, 6):
            store.record("aegis.reward.value", float(tick), tick=tick)
            store.flush()
    finally:
        Telemetry._needs_newline = staticmethod(original)

    assert len(seen) == 1, f"the file was interrogated {len(seen)} times"
    values = list(store.series("aegis.reward.value").values)
    assert values == [1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
