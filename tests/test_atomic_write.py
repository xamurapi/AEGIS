"""Tests for the atomic-write helper and its use in stat persistence (audit A3)."""
import json
import os
import threading
from pathlib import Path

import pytest

from aegis._atomic import atomic_write_text


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, '{"a": 1}')
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_atomic_write_leaves_no_temp(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "hello")
    assert not (tmp_path / "x.json.tmp").exists()


def test_atomic_write_overwrites_completely(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "aaaaaaaaaa")   # 10 chars
    atomic_write_text(p, "bb")            # shorter — must fully replace
    assert p.read_text(encoding="utf-8") == "bb"


def test_atomic_write_preserves_lf_newlines(tmp_path):
    p = tmp_path / "x.txt"
    atomic_write_text(p, "a\nb\nc\n")
    assert p.read_bytes() == b"a\nb\nc\n"  # no CRLF translation


def test_each_write_uses_a_unique_temp_name(tmp_path, monkeypatch):
    """Two writers of the same store must never share a temp file.

    With the old fixed ``<name>.tmp`` both writers truncated and interleaved
    in the SAME temp file and then both replaced — exactly the corruption the
    helper exists to prevent (reachable: two /api/eval runs saving
    eval_history.json from executor threads)."""
    seen = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(str(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    target = tmp_path / "store.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert len(seen) == 2
    assert seen[0] != seen[1], "temp name is fixed — concurrent writers collide"


def test_concurrent_writers_never_corrupt_the_file(tmp_path):
    """Interleave two writers hammering one store: the file must always end up
    as ONE complete payload, parseable, with no temp debris left behind."""
    target = tmp_path / "store.json"
    payloads = [
        json.dumps({"writer": index, "data": ["x" * 40] * 300})
        for index in range(2)
    ]
    errors = []

    def hammer(text):
        try:
            for _ in range(120):
                atomic_write_text(target, text)
        except Exception as exc:      # noqa: BLE001 — the assertion IS the catch
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(payload,))
               for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors[:3]
    content = target.read_text(encoding="utf-8")
    assert content in payloads, "writers interleaved inside one file"
    json.loads(content)
    assert not list(tmp_path.glob("*.tmp"))


def test_a_failed_write_leaves_no_temp_debris(tmp_path):
    """The temp file must be cleaned up when the write itself fails — store
    directories are long-lived and would slowly fill otherwise."""
    target = tmp_path / "adir"
    target.mkdir()          # replacing a directory fails on every platform
    try:
        atomic_write_text(target, "content")
    except Exception:
        pass
    assert not list(tmp_path.glob("*.tmp"))


def test_llm_stats_save_is_atomic(tmp_path, monkeypatch):
    # LLMEngine._save_lifetime_stats routes through atomic_write_text now.
    import aegis.llm as llm
    monkeypatch.setattr(llm, "TOKEN_STATS_FILE", tmp_path / "token_stats.json")
    e = llm.LLMEngine()
    e.lifetime_calls = 42
    e._save_lifetime_stats()
    data = json.loads((tmp_path / "token_stats.json").read_text(encoding="utf-8"))
    assert data["lifetime_calls"] == 42
    assert not (tmp_path / "token_stats.json.tmp").exists()


# ── the paths that only run on one platform, or only on failure ──────
#
# These branches decide whether the module's two promises — "no debris" and
# "the rename itself survives a power cut" — hold. On Windows the directory
# fsync is a no-op by construction, so without simulation the POSIX arm would
# never be executed by this suite and could rot unnoticed.

def test_the_rename_is_retried_when_the_platform_refuses_it_transiently(tmp_path, monkeypatch):
    """Windows can refuse a replace for an instant while another writer renames
    onto the same target. Each attempt renames a COMPLETE file, so the retry
    costs nothing in atomicity — and without it the concurrency this helper
    exists to survive would surface as a spurious save failure."""
    from aegis import _atomic
    target = tmp_path / "store.json"
    calls = []
    real_replace = os.replace

    def flaky_replace(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("transient")
        return real_replace(src, dst)

    monkeypatch.setattr(_atomic.os, "replace", flaky_replace)
    monkeypatch.setattr(_atomic.time, "sleep", lambda _s: None)
    atomic_write_text(target, "payload")

    assert len(calls) == 3
    assert target.read_text(encoding="utf-8") == "payload"
    assert not list(tmp_path.glob("*.tmp"))


def test_a_permanently_refused_rename_raises_and_still_cleans_up(tmp_path, monkeypatch):
    from aegis import _atomic
    target = tmp_path / "store.json"
    monkeypatch.setattr(_atomic.os, "replace",
                        lambda src, dst: (_ for _ in ()).throw(PermissionError("held")))
    monkeypatch.setattr(_atomic.time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        atomic_write_text(target, "payload")
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_cleanup_that_cannot_delete_the_temp_file_does_not_mask_the_real_error(tmp_path, monkeypatch):
    """The caller has to see WHY the write failed. An unlink that fails during
    cleanup must not replace that with its own exception."""
    from aegis import _atomic
    target = tmp_path / "store.json"
    monkeypatch.setattr(_atomic.os, "replace",
                        lambda src, dst: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    monkeypatch.setattr(_atomic.os, "unlink",
                        lambda name: (_ for _ in ()).throw(OSError("locked")))

    with pytest.raises(RuntimeError, match="disk on fire"):
        atomic_write_text(target, "payload")


def test_the_directory_entry_is_fsynced_where_the_platform_supports_it(tmp_path, monkeypatch):
    """POSIX persists a rename only after the DIRECTORY is fsynced. Windows
    cannot open a directory for sync, so there the helper returns early — this
    test drives the POSIX arm on either platform."""
    from aegis import _atomic
    target = tmp_path / "store.json"
    synced, closed = [], []
    real_open, real_fsync, real_close = _atomic.os.open, _atomic.os.fsync, _atomic.os.close

    # `_atomic.os` IS the os module, so this patch is global — tempfile's own
    # open goes through it too. Intercept only the directory handle and pass
    # everything else (including mkstemp's mode argument) straight through.
    def fake_open(name, flags, *rest):
        if Path(name) == tmp_path:
            return 4242
        return real_open(name, flags, *rest)

    monkeypatch.setattr(_atomic.os, "open", fake_open)
    monkeypatch.setattr(_atomic.os, "fsync",
                        lambda fd: synced.append(fd) if fd == 4242 else real_fsync(fd))
    monkeypatch.setattr(_atomic.os, "close",
                        lambda fd: closed.append(fd) if fd == 4242 else real_close(fd))

    atomic_write_text(target, "payload")
    assert synced == [4242], "the directory entry was never fsynced"
    assert closed == [4242], "the directory handle leaked"


def test_a_directory_that_cannot_be_synced_is_not_an_error(tmp_path, monkeypatch):
    """The data is already fsynced and the rename has happened; a platform that
    refuses the directory sync must not turn a completed write into a failure."""
    from aegis import _atomic
    target = tmp_path / "store.json"
    real_open, real_fsync, real_close = _atomic.os.open, _atomic.os.fsync, _atomic.os.close
    closed = []

    monkeypatch.setattr(_atomic.os, "open",
                        lambda name, flags, *rest: 4243 if Path(name) == tmp_path
                        else real_open(name, flags, *rest))
    monkeypatch.setattr(_atomic.os, "fsync",
                        lambda fd: (_ for _ in ()).throw(OSError("no dir sync"))
                        if fd == 4243 else real_fsync(fd))
    monkeypatch.setattr(_atomic.os, "close",
                        lambda fd: closed.append(fd) if fd == 4243 else real_close(fd))

    atomic_write_text(target, "payload")
    assert target.read_text(encoding="utf-8") == "payload"
    assert closed == [4243], "the directory handle leaked on the failure path"
