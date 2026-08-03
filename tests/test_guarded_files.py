"""Appendix H: the five files that may not change without a security audit.

`ethics_core`, `self_preservation`, `sandbox`, `_atomic` and `event_bus` are the
system's fuses. The spec puts them out of bounds for ordinary work, and this
test is what makes that a rule rather than an intention — a routine lint sweep
already edited two of them once.

The digest is over the source text. Changing one of these files is allowed, but
it has to be a deliberate act that updates this expectation in the same commit,
which is exactly the review checkpoint Appendix H asks for.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GUARDED = (
    "aegis/layers/ethics_core.py",
    "aegis/layers/self_preservation.py",
    "aegis/eval/sandbox.py",
    "aegis/_atomic.py",
    "aegis/event_bus.py",
)


def _digest(relative: str) -> str:
    data = (ROOT / relative).read_bytes()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


#: Filled in below from the current tree — see the module docstring for how to
#: update it legitimately.
EXPECTED = {
    "aegis/layers/ethics_core.py":
        "459e972f620d5bbf3537f11cfdbb3365",
    # Updated for the current audit round's deliberate fixes to these files
    # (sandbox hardening; self-preservation changes owned by the layers work).
    "aegis/layers/self_preservation.py":
        "e5eb8cb9cdd198e2b5d50fa6fafb6511",
    "aegis/eval/sandbox.py":
        "b926e924fd8a41169f44eff1a2bf5a3e",
    # Updated deliberately for the audit fix "concurrent writers corrupt the
    # store": unique temp names (tempfile.mkstemp) instead of a fixed
    # <name>.tmp, plus fsync of data and directory so the promised crash
    # safety actually holds. See aegis/_atomic.py and
    # tests/test_atomic_write.py::test_concurrent_writers_never_corrupt_the_file.
    "aegis/_atomic.py":
        "35d8a311871f9708da3686cecbfaf5c0",
    "aegis/event_bus.py":
        "049df184d2c5084691194df7bda9018a",
}


def test_every_guarded_file_is_listed():
    assert set(EXPECTED) == set(GUARDED)


def test_the_fuses_are_unchanged():
    changed = [name for name in GUARDED if _digest(name) != EXPECTED[name]]
    assert not changed, (
        "Appendix H forbids changing these without a separate security audit: "
        + ", ".join(changed)
        + ". If the change is deliberate, update EXPECTED in this file in the "
          "same commit and say why in the message.")
