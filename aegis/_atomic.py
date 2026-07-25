"""Atomic file write helper (audit A3).

A crash or interrupt in the middle of a plain ``path.write_text(...)`` leaves a
truncated/corrupt file. For the stat/store JSON files (token stats, training
stats, eval history, skill library) that means the load path catches the decode
error and silently resets accumulated history to zero. Writing to a temp file
in the same directory and then ``os.replace``-ing it makes the update atomic:
readers see either the old complete file or the new complete file, never a
half-written one.
"""
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temp + replace.

    Same-directory is required so the final ``replace`` is a rename on one
    filesystem (atomic) rather than a cross-device copy. Preserves ``\\n`` line
    endings (newline="") so round-tripping a file does not rewrite them.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="")
    tmp.replace(path)
