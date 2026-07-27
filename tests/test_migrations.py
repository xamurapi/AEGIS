"""Versioned stores and their migrations (spec §3.2, Appendix I).

The rules under test are the ones the whole persistence layer rests on:
an unversioned file is v1, a future version is refused rather than half-read,
a corrupt file yields empty state instead of a crash, and writes are atomic.
"""
import json

import pytest

from aegis.store.migrations import (
    CURRENT_VERSION, MIGRATIONS, MigrationError, migrate, read_store,
    version_of, write_store,
)


# ── version detection ────────────────────────────────────────────────

def test_missing_version_is_treated_as_v1():
    assert version_of({"links": {}}) == 1


def test_explicit_version_is_read():
    assert version_of({"schema_version": 2}) == 2


def test_unparseable_version_falls_back_to_v1():
    assert version_of({"schema_version": "not-a-number"}) == 1


def test_non_dict_payload_is_v1():
    assert version_of(["a", "list"]) == 1


# ── migrate() ────────────────────────────────────────────────────────

def test_migrate_stamps_the_target_version():
    out = migrate({"a": 1}, 1, 2, store="goal_intelligence")
    assert out["schema_version"] == 2
    assert out["a"] == 1


def test_migrate_is_a_no_op_when_already_current():
    payload = {"schema_version": 2, "a": 1}
    assert migrate(payload, 2, 2, store="anything") == payload


def test_migrate_refuses_to_go_backwards():
    with pytest.raises(MigrationError) as excinfo:
        migrate({"schema_version": 3}, 3, 2, store="lineage")
    # The message has to name the store and the direction, or an operator
    # reading a boot log cannot tell which file is the problem.
    message = str(excinfo.value)
    assert "lineage" in message
    assert "3 -> 2" in message


def test_backwards_refusal_reads_sensibly_for_an_unnamed_store():
    with pytest.raises(MigrationError) as excinfo:
        migrate({}, 3, 2)
    assert "cannot migrate store backwards" in str(excinfo.value)


def test_unregistered_store_passes_through():
    out = migrate({"kept": "value"}, 1, 2, store="brand_new_store")
    assert out["kept"] == "value"
    assert out["schema_version"] == 2


def test_a_failing_migration_is_reported_not_swallowed(monkeypatch):
    def explode(_data):
        raise RuntimeError("boom")

    monkeypatch.setitem(MIGRATIONS, ("fragile", 1), explode)
    with pytest.raises(MigrationError) as excinfo:
        migrate({}, 1, 2, store="fragile")
    message = str(excinfo.value)
    assert "fragile" in message      # which store
    assert "1 -> 2" in message       # which step of the chain
    assert "boom" in message         # and the underlying cause


def test_a_failing_migration_names_the_step_it_died_on(monkeypatch):
    def explode(_data):
        raise RuntimeError("boom")

    monkeypatch.setitem(MIGRATIONS, ("chained", 2), explode)
    with pytest.raises(MigrationError) as excinfo:
        migrate({}, 1, 3, store="chained")
    # It got through 1 -> 2 and died on 2 -> 3; saying "1 -> 2" would send the
    # reader to the wrong migration.
    assert "2 -> 3" in str(excinfo.value)


def test_multi_step_migration_applies_every_step(monkeypatch):
    monkeypatch.setitem(MIGRATIONS, ("multi", 1), lambda d: {**d, "one": True})
    monkeypatch.setitem(MIGRATIONS, ("multi", 2), lambda d: {**d, "two": True})
    out = migrate({}, 1, 3, store="multi")
    assert out["one"] and out["two"]
    assert out["schema_version"] == 3


def test_world_model_migration_keeps_links_and_seeds_the_rest():
    v1 = {"links": {"a": {"b": {"observations": 3, "successes": 2}}},
          "total_observations": 3}
    out = migrate(v1, 1, 2, store="world_model")
    assert out["links"] == v1["links"]          # causal history survives verbatim
    assert out["total_observations"] == 3
    assert out["chains"] == []
    assert out["schema_version"] == 2


def test_world_model_migration_does_not_invent_transition_data():
    # Transitions are keyed by encoded system state, which v1 never recorded.
    # Fabricating them would poison every calibration number computed after.
    out = migrate({"links": {}}, 1, 2, store="world_model")
    assert not out.get("transitions")
    assert not out.get("outcomes")


# ── read_store ───────────────────────────────────────────────────────

def test_missing_file_yields_the_default(tmp_path):
    out = read_store(tmp_path / "nope.json", store="x", default={"a": 1})
    assert out["a"] == 1
    assert out["schema_version"] == CURRENT_VERSION


def test_corrupt_file_yields_empty_state_without_raising(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    assert read_store(path, store="x") == {"schema_version": CURRENT_VERSION}


def test_non_object_file_yields_empty_state(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_store(path, store="x") == {"schema_version": CURRENT_VERSION}


def test_future_version_is_refused_rather_than_half_read(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 99, "secret": "field"}),
                    encoding="utf-8")
    out = read_store(path, store="x")
    # Reading it would mean writing it back WITHOUT `secret` — silent data loss.
    assert "secret" not in out


def test_legacy_unversioned_file_is_migrated_on_read(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"links": {"c": {"e": {"observations": 1}}}}),
                    encoding="utf-8")
    out = read_store(path, store="world_model")
    assert out["schema_version"] == CURRENT_VERSION
    assert out["links"]["c"]["e"]["observations"] == 1


def test_current_version_file_is_returned_untouched(tmp_path):
    path = tmp_path / "current.json"
    payload = {"schema_version": CURRENT_VERSION, "value": 42}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_store(path, store="x") == payload


def test_read_store_survives_an_unreadable_path(tmp_path):
    # A directory where a file is expected: open() raises, and the caller must
    # still get usable state back.
    directory = tmp_path / "adir"
    directory.mkdir()
    assert read_store(directory, store="x") == {"schema_version": CURRENT_VERSION}


# ── write_store ──────────────────────────────────────────────────────

def test_write_store_stamps_the_version_and_round_trips(tmp_path):
    path = tmp_path / "out.json"
    assert write_store(path, {"a": 1}) is True
    assert read_store(path, store="x") == {"schema_version": CURRENT_VERSION, "a": 1}


def test_write_store_creates_missing_parents(tmp_path):
    path = tmp_path / "deep" / "nested" / "out.json"
    assert write_store(path, {"a": 1}) is True
    assert path.exists()


def test_write_store_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "out.json"
    write_store(path, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_write_store_reports_failure_instead_of_raising(tmp_path):
    # Target is an existing directory — the replace cannot succeed.
    target = tmp_path / "adir"
    target.mkdir()
    assert write_store(target, {"a": 1}) is False


def test_write_store_does_not_mutate_the_callers_dict(tmp_path):
    payload = {"a": 1}
    write_store(tmp_path / "out.json", payload)
    assert payload == {"a": 1}
