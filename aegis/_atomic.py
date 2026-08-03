"""Atomic file write helper (audit A3).

A crash or interrupt in the middle of a plain ``path.write_text(...)`` leaves a
truncated/corrupt file. For the stat/store JSON files (token stats, training
stats, eval history, skill library) that means the load path catches the decode
error and silently resets accumulated history to zero. Writing to a temp file
in the same directory and then ``os.replace``-ing it makes the update atomic:
readers see either the old complete file or the new complete file, never a
half-written one.
"""
import os
import tempfile
import time
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temp + replace.

    Same-directory is required so the final ``replace`` is a rename on one
    filesystem (atomic) rather than a cross-device copy. Preserves ``\\n`` line
    endings (newline="") so round-tripping a file does not rewrite them.

    The temp file gets a UNIQUE name per call (``tempfile.mkstemp``), not a
    fixed ``<name>.tmp``: with a fixed name, two concurrent writers of the same
    store truncate and interleave in the SAME temp file and then both replace —
    producing exactly the corruption this helper exists to prevent. That race
    is reachable in production (two /api/eval runs on executor threads both
    calling Evaluator._save()). With unique temp files each writer renames its
    own complete snapshot; the loser's data is dropped whole, never mixed.

    The data and its directory entry are fsync'ed before/after the replace so
    the "crash safety" the module docstring promises actually holds: without
    the flush+fsync the rename can land on disk before the data does, and a
    power cut then reveals an empty file under the final name.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows, ReplaceFile can transiently refuse (PermissionError)
        # when another writer is renaming onto the same target in the same
        # instant, or an indexer/AV briefly holds the file. Each attempt
        # renames a COMPLETE file, so a short bounded retry costs nothing in
        # atomicity — without it, the very concurrency this helper exists to
        # survive turns into a spurious save failure.
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except BaseException:
        # Never leave an orphaned temp file behind on failure — the store
        # directories are long-lived and would slowly fill with debris.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    # Persist the rename itself. POSIX requires fsync on the directory for
    # that; Windows has no equivalent (directories cannot be opened for sync),
    # so there this is a best-effort no-op.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
