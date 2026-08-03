"""Security regression tests for CodeModifier.validate_safety.

Covers audit finding A1: `open(..., 'w')` arbitrary-file-write was not blocked.
"""
import pytest

from aegis.layers.code_modifier import CodeModifier


@pytest.fixture
def cm(tmp_path):
    base = tmp_path / "pkg"
    (base / "layers").mkdir(parents=True)
    (base / "layers" / "toy.py").write_text("x = 1\n", encoding="utf-8")
    return CodeModifier(base_dir=base, backups_dir=tmp_path / "backups")


def _blocked(cm, code):
    safe, warnings = cm.validate_safety(code, "layers/toy.py")
    return (not safe), warnings


def test_open_write_mode_blocked(cm):
    blocked, w = _blocked(cm, "def f():\n    open('anything.txt', 'w').write('x')\n")
    assert blocked
    assert any("open()" in m for m in w)


def test_open_append_mode_blocked(cm):
    blocked, _ = _blocked(cm, "def f():\n    open('log.txt', 'a')\n")
    assert blocked


def test_open_read_plus_mode_blocked(cm):
    blocked, _ = _blocked(cm, "def f():\n    open('f', 'r+')\n")
    assert blocked


def test_open_mode_via_keyword_blocked(cm):
    blocked, _ = _blocked(cm, "def f():\n    open('f', mode='wb')\n")
    assert blocked


def test_open_dynamic_mode_blocked(cm):
    # Non-literal mode cannot be verified read-only -> blocked.
    blocked, _ = _blocked(cm, "def f(m):\n    open('f', m)\n")
    assert blocked


def test_open_read_mode_allowed(cm):
    safe, _ = cm.validate_safety("def f():\n    open('f').read()\n", "layers/toy.py")
    assert safe
    safe2, _ = cm.validate_safety("def f():\n    open('f', 'r').read()\n", "layers/toy.py")
    assert safe2


def test_open_read_binary_allowed(cm):
    safe, _ = cm.validate_safety("def f():\n    open('f', 'rb').read()\n", "layers/toy.py")
    assert safe


# ── audit C4: the process-execution family was not in the blocklist ──
# os.system/popen were blocked while os.execv & co. — which run (or become)
# an arbitrary program just the same — sailed through.

@pytest.mark.parametrize("call", [
    "os.execv('/bin/sh', ['sh'])",
    "os.execl('/bin/sh', 'sh')",
    "os.execvp('sh', ['sh'])",
    "os.spawnv(0, '/bin/sh', ['sh'])",
    "os.spawnl(0, '/bin/sh', 'sh')",
    "os.posix_spawn('/bin/sh', ['sh'], {})",
    "os.fork()",
    "os.startfile('evil.exe')",
])
def test_process_execution_family_blocked(cm, call):
    blocked, w = _blocked(cm, f"import os\ndef f():\n    {call}\n")
    assert blocked, w


def test_process_execution_blocked_through_import_alias(cm):
    blocked, _ = _blocked(cm, "import os as o\ndef f():\n    o.execv('/bin/sh', ['sh'])\n")
    assert blocked


def test_process_execution_blocked_through_from_import(cm):
    blocked, _ = _blocked(cm, "from os import execv\ndef f():\n    execv('/bin/sh', ['sh'])\n")
    assert blocked


# ── audit C4: write APIs reached through a non-Name receiver ─────────
# open() was only checked as a bare Name, so Path(x).write_text(y) — the
# same arbitrary-file-write — was never reached by the gate.

def test_path_write_text_blocked(cm):
    blocked, w = _blocked(
        cm, "from pathlib import Path\ndef f():\n    Path('x').write_text('pwn')\n")
    assert blocked
    assert any("write_text" in m for m in w)


def test_path_write_bytes_blocked(cm):
    blocked, _ = _blocked(
        cm, "from pathlib import Path\ndef f():\n    Path('x').write_bytes(b'p')\n")
    assert blocked


def test_path_open_write_mode_blocked(cm):
    blocked, _ = _blocked(
        cm, "from pathlib import Path\ndef f():\n    Path('x').open('w')\n")
    assert blocked


def test_variable_path_open_append_mode_blocked(cm):
    blocked, _ = _blocked(
        cm, "from pathlib import Path\ndef f(p):\n    p.open('a')\n")
    assert blocked


def test_path_open_read_mode_allowed(cm):
    safe, w = cm.validate_safety(
        "from pathlib import Path\ndef f():\n    Path('x').open('r').read()\n",
        "layers/toy.py")
    assert safe, w
    safe2, _ = cm.validate_safety(
        "from pathlib import Path\ndef f():\n    Path('x').open().read()\n",
        "layers/toy.py")
    assert safe2


def test_non_file_open_method_with_variable_arg_allowed(cm):
    # world_model.py really contains `self.scorer.open(prediction)` — an .open
    # method that is not a file API. A whole-file rewrite must stay possible.
    safe, w = cm.validate_safety(
        "def f(self, prediction):\n    return self.scorer.open(prediction)\n",
        "layers/toy.py")
    assert safe, w
