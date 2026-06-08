"""Tests for CodeModifier: safety validation, immutability, apply + rollback."""
import pytest
from aegis.layers.code_modifier import CodeModifier, IMMUTABLE_FILES


@pytest.fixture
def cm(tmp_path):
    base = tmp_path / "pkg"
    base.mkdir()
    (base / "sample.py").write_text("x = 1\n\n\ndef f():\n    return x\n", encoding="utf-8")
    backups = tmp_path / "backups"
    return CodeModifier(base, backups)


def test_valid_syntax_accepted(cm):
    ok, _ = cm.validate_syntax("a = 1\n")
    assert ok


def test_invalid_syntax_rejected(cm):
    ok, msg = cm.validate_syntax("def (:\n")
    assert not ok and "Syntax" in msg


def test_eval_call_blocked_even_with_spaces(cm):
    # AST-based detection cannot be fooled by spacing.
    safe, warnings = cm.validate_safety("y = eval ( '1+1' )\n", "sample.py")
    assert not safe
    assert any("eval" in w for w in warnings)


def test_os_system_call_blocked(cm):
    safe, warnings = cm.validate_safety("import os\nos.system('ls')\n", "sample.py")
    assert not safe


def test_forbidden_import_blocked(cm):
    safe, warnings = cm.validate_safety("import subprocess\n", "sample.py")
    assert not safe


def test_dunder_import_blocked(cm):
    safe, _ = cm.validate_safety("m = __import__('os')\n", "sample.py")
    assert not safe


def test_benign_code_allowed(cm):
    safe, warnings = cm.validate_safety("z = sum([1, 2, 3])\n", "sample.py")
    assert safe


def test_immutable_file_cannot_be_modified(cm):
    target = next(iter(IMMUTABLE_FILES))
    safe, warnings = cm.validate_safety("x = 1\n", target)
    assert not safe


def test_apply_and_rollback(cm):
    new_code = "x = 2\n\n\ndef f():\n    return x + 1\n"
    res = cm.apply_modification("sample.py", new_code, "tweak")
    assert res["status"] == "applied_pending_restart"
    assert (cm.base_dir / "sample.py").read_text(encoding="utf-8") == new_code

    rb = cm.rollback_last()
    assert rb["success"] is True
    assert "x = 1" in (cm.base_dir / "sample.py").read_text(encoding="utf-8")


def test_apply_rejects_broken_syntax_without_writing(cm):
    before = (cm.base_dir / "sample.py").read_text(encoding="utf-8")
    res = cm.apply_modification("sample.py", "def (:\n", "broken")
    assert res["status"] == "syntax_error"
    assert (cm.base_dir / "sample.py").read_text(encoding="utf-8") == before


def test_rollback_to_unknown_id_is_noop(cm):
    res = cm.rollback_to("does_not_exist")
    assert res["success"] is False
