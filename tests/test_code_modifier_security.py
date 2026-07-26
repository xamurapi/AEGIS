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
