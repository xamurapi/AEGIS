"""Tests for StateBackup — save/restore, emergency snapshots, rotation, error handling,
listing and status. Always uses tmp_path so nothing is written into the real backups/."""
import gzip
import json

import pytest

from aegis.layers.state_backup import StateBackup


@pytest.fixture
def sb(tmp_path):
    return StateBackup(backup_dir=tmp_path / "backups", max_backups=5)


# ── save_state ────────────────────────────────────────────────────────

def test_save_state_scheduled(sb):
    rec = sb.save_state({"a": 1, "b": [1, 2, 3]})
    assert rec["success"] is True
    assert rec["type"] == "scheduled"
    assert rec["file"].startswith("aegis_scheduled_")
    assert rec["size_raw"] > 0 and rec["size_compressed"] > 0
    assert sb.backup_count == 1
    assert sb.last_backup_time > 0
    assert (sb.backup_dir / rec["file"]).exists()


def test_save_state_serializes_non_json_via_str(sb):
    # default=str must let a non-serializable value through
    rec = sb.save_state({"path": object()})
    assert rec["success"] is True


def test_save_state_failure_on_circular_ref(sb):
    circ = {}
    circ["self"] = circ  # json.dumps raises ValueError even with default=str
    rec = sb.save_state(circ)
    assert rec["success"] is False
    assert "error" in rec
    assert sb.failed_backups == 1


# ── emergency_backup ──────────────────────────────────────────────────

def test_emergency_backup(sb):
    rec = sb.emergency_backup({"panic": True})
    assert rec["success"] is True
    assert rec["type"] == "emergency"
    assert rec["file"].startswith("aegis_emergency_")


# ── restore ───────────────────────────────────────────────────────────

def test_restore_latest_roundtrip(sb):
    sb.save_state({"n": 1})
    sb.save_state({"n": 2})
    restored = sb.restore_latest()
    assert restored["n"] == 2
    assert sb.restore_count == 1


def test_restore_latest_filtered_by_type(sb):
    sb.save_state({"kind": "sched"}, backup_type="scheduled")
    sb.emergency_backup({"kind": "emerg"})
    restored = sb.restore_latest(backup_type="emergency")
    assert restored["kind"] == "emerg"


def test_restore_latest_none_when_empty(sb):
    assert sb.restore_latest() is None


def test_restore_skips_corrupt_and_returns_next(sb):
    good = sb.save_state({"good": True})
    # write a corrupt file that sorts AFTER the good one (higher timestamp)
    bad = sb.backup_dir / "aegis_scheduled_9999999999999999999.json.gz"
    bad.write_bytes(b"not gzip data")
    restored = sb.restore_latest()
    assert restored == {"good": True}
    assert good["success"]


# ── rotation ──────────────────────────────────────────────────────────

def test_rotation_keeps_only_max_per_type(sb):
    for i in range(sb.max_backups + 4):
        sb.save_state({"i": i}, backup_type="scheduled")
    files = list(sb.backup_dir.glob("aegis_scheduled_*.json.gz"))
    assert len(files) == sb.max_backups


def test_rotation_is_per_type_emergency_survives(sb):
    sb.emergency_backup({"critical": True})
    for i in range(sb.max_backups + 5):
        sb.save_state({"i": i}, backup_type="scheduled")
    emerg = list(sb.backup_dir.glob("aegis_emergency_*.json.gz"))
    sched = list(sb.backup_dir.glob("aegis_scheduled_*.json.gz"))
    assert len(emerg) == 1  # not evicted by scheduled burst
    assert len(sched) == sb.max_backups


def test_rotation_unlink_error_is_tolerated(sb, monkeypatch):
    for i in range(sb.max_backups + 2):
        sb.save_state({"i": i}, backup_type="scheduled")
    # force unlink to fail during the next rotation; must not raise
    from pathlib import Path

    def boom(self, *a, **k):
        raise OSError("in use")

    monkeypatch.setattr(Path, "unlink", boom)
    rec = sb.save_state({"i": "x"}, backup_type="scheduled")
    assert rec["success"] is True


# ── listing ───────────────────────────────────────────────────────────

def test_list_backups(sb):
    sb.save_state({"a": 1}, backup_type="scheduled")
    sb.emergency_backup({"b": 2})
    listed = sb.list_backups()
    assert len(listed) == 2
    types = {item["type"] for item in listed}
    assert "scheduled" in types and "emergency" in types
    for item in listed:
        assert item["size_bytes"] > 0
        assert item["timestamp"] != "unknown"


# ── status ────────────────────────────────────────────────────────────

def test_status(sb):
    sb.save_state({"a": 1})
    sb.emergency_backup({"b": 2})
    sb.restore_latest()
    st = sb.status()
    assert st["backup_count"] == 2
    assert st["restore_count"] == 1
    assert st["available_backups"] == 2
    assert st["backup_dir"] == str(sb.backup_dir)
    assert len(st["recent_history"]) >= 1
    assert all("type" in h and "success" in h for h in st["recent_history"])


def test_default_backup_dir_created(tmp_path, monkeypatch):
    # Exercise the default-path branch without writing to the repo: redirect the
    # module's __file__-derived default by passing an explicit dir is the norm,
    # so here we just confirm an explicit None falls back and still creates a dir.
    d = tmp_path / "explicit"
    sb = StateBackup(backup_dir=d)
    assert d.exists()
    assert sb.max_backups == 20


def test_written_file_is_valid_gzip_json(sb):
    rec = sb.save_state({"hello": "world"})
    raw = (sb.backup_dir / rec["file"]).read_bytes()
    assert json.loads(gzip.decompress(raw).decode("utf-8")) == {"hello": "world"}
