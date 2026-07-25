"""Extended tests for CodeModifier — path safety, validation branches, apply/rollback,
compile-failure rollback, analysis and stats persistence. All work inside tmp_path so
the real aegis/ package and data/ are never touched."""
import json
from pathlib import Path

import pytest

from aegis.layers.code_modifier import (
    CodeModifier,
    IMMUTABLE_FILES,
    MAX_MODIFICATION_SIZE,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def pkg(tmp_path):
    """A fake mini-package under tmp_path/pkg with a couple of .py files and
    a fake immutable config.py, plus a layers/ subpackage."""
    base = tmp_path / "pkg"
    base.mkdir()
    (base / "sample.py").write_text("x = 1\n\n\ndef f():\n    return x\n", encoding="utf-8")
    (base / "config.py").write_text("SETTING = 1\n", encoding="utf-8")
    layers = base / "layers"
    layers.mkdir()
    (layers / "ethics_core.py").write_text(
        "AXIOMS = ['E-001', 'E-002', 'E-003', 'E-004']\n", encoding="utf-8"
    )
    (layers / "helper.py").write_text("y = 2\n", encoding="utf-8")
    # a __pycache__ file that list_sources must skip
    cache = base / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("noop = 0\n", encoding="utf-8")
    return base


@pytest.fixture
def cm(pkg, tmp_path):
    return CodeModifier(pkg, tmp_path / "backups")


# ── Path resolution ───────────────────────────────────────────────────

def test_resolve_rejects_absolute_path(cm):
    with pytest.raises(ValueError, match="Absolute paths"):
        cm._resolve_path(str(Path.cwd() / "x.py"))


def test_resolve_rejects_traversal_escape(cm):
    with pytest.raises(ValueError, match="escapes package"):
        cm._resolve_path("../outside.py")


def test_resolve_normalizes_relative(cm):
    resolved, norm = cm._resolve_path("layers/../sample.py")
    assert norm == "sample.py"
    assert resolved.name == "sample.py"


# ── Read / list ───────────────────────────────────────────────────────

def test_read_source_ok(cm):
    assert "def f()" in cm.read_source("sample.py")


def test_read_source_missing_raises(cm):
    with pytest.raises(FileNotFoundError):
        cm.read_source("nope.py")


def test_read_source_non_py_rejected(cm):
    (cm.base_dir / "data.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="Only .py"):
        cm.read_source("data.txt")


def test_list_sources_skips_pycache_and_marks_immutable(cm):
    srcs = cm.list_sources()
    paths = {s["path"] for s in srcs}
    assert "sample.py" in paths
    assert not any("__pycache__" in p for p in paths)
    ec = next(s for s in srcs if s["path"] == "layers/ethics_core.py")
    assert ec["immutable"] is True
    sm = next(s for s in srcs if s["path"] == "sample.py")
    assert sm["immutable"] is False
    assert sm["lines"] >= 1 and sm["size"] > 0


# ── Syntax validation ─────────────────────────────────────────────────

def test_validate_syntax_ok(cm):
    ok, msg = cm.validate_syntax("a = 1\n")
    assert ok and msg == "OK"


def test_validate_syntax_error(cm):
    ok, msg = cm.validate_syntax("def (:\n")
    assert not ok and "Syntax error" in msg


# ── Safety validation branches ────────────────────────────────────────

def test_safety_blocks_immutable(cm):
    safe, warns = cm.validate_safety("SETTING = 2\n", "config.py")
    assert not safe
    assert any("immutable" in w for w in warns)


def test_safety_blocks_traversal_path(cm):
    safe, warns = cm.validate_safety("x = 1\n", "../evil.py")
    assert not safe
    assert any("escapes package" in w for w in warns)


def test_safety_blocks_immutable_via_traversal(cm):
    # resolved path check must catch layers/../config.py -> config.py
    safe, warns = cm.validate_safety("SETTING = 9\n", "layers/../config.py")
    assert not safe
    assert any("immutable" in w for w in warns)


def test_safety_forbidden_substring_pattern(cm):
    # 'subprocess' substring appears even in a comment -> FORBIDDEN pattern
    safe, warns = cm.validate_safety("# uses subprocess here\nx = 1\n", "sample.py")
    assert not safe
    assert any("FORBIDDEN" in w for w in warns)


def test_safety_blocks_oversize_modification(cm):
    big = "x = 1\n" + ("# pad\n" * ((MAX_MODIFICATION_SIZE // 6) + 100))
    safe, warns = cm.validate_safety(big, "sample.py")
    assert not safe
    assert any("too large" in w for w in warns)


def test_safety_new_file_ok(cm):
    # target doesn't exist -> FileNotFoundError path in size check is swallowed
    safe, warns = cm.validate_safety("new_val = 5\n", "brandnew.py")
    assert safe and warns == []


def test_safety_non_py_target_blocked(cm):
    (cm.base_dir / "note.txt").write_text("hi", encoding="utf-8")
    safe, warns = cm.validate_safety("x = 1\n", "note.txt")
    assert not safe
    assert any("Only .py" in w for w in warns)


def test_safety_dangerous_bare_call(cm):
    safe, warns = cm.validate_safety("exec('x=1')\n", "sample.py")
    assert not safe
    assert any("exec" in w for w in warns)


def test_safety_dangerous_attr_call_with_alias(cm):
    # import os as o ; o.kill(...) must be resolved back to os.kill
    code = "import os as o\no.kill(1, 9)\n"
    safe, warns = cm.validate_safety(code, "sample.py")
    assert not safe
    assert any("os.kill" in w for w in warns)


def test_safety_forbidden_import_from(cm):
    safe, warns = cm.validate_safety("from importlib import import_module\n", "sample.py")
    assert not safe
    assert any("importlib" in w for w in warns)


def test_safety_forbidden_plain_import(cm):
    safe, warns = cm.validate_safety("import ctypes\n", "sample.py")
    assert not safe
    assert any("ctypes" in w for w in warns)


def test_safety_syntax_error_swallowed(cm):
    # broken syntax -> AST block hits SyntaxError branch (pass); no crash.
    safe, warns = cm.validate_safety("def (:\n", "sample.py")
    # nothing forbidden/blocked flagged
    assert safe is True


def test_safety_benign_ok(cm):
    safe, warns = cm.validate_safety("z = sum([1, 2, 3])\n", "sample.py")
    assert safe and warns == []


# ── Ethics preservation ───────────────────────────────────────────────

def test_ethics_not_relevant_returns_true(cm):
    assert cm.validate_ethics_preserved("x = 1\n", "sample.py") is True


def test_ethics_missing_axiom_returns_false(cm):
    # path mentions ethics -> axioms required, but they are absent
    assert cm.validate_ethics_preserved("nothing here\n", "layers/ethics_stuff.py") is False


def test_ethics_all_axioms_present_true(cm):
    code = "axiom set: E-001 E-002 E-003 E-004\n"
    assert cm.validate_ethics_preserved(code, "layers/ethics_stuff.py") is True


# ── apply_modification: all outcome branches ──────────────────────────

def test_apply_path_blocked(cm):
    res = cm.apply_modification("../evil.py", "x = 1\n", "bad")
    assert res["status"] == "path_blocked"
    assert cm.blocked_mods == 1


def test_apply_syntax_error(cm):
    before = cm.read_source("sample.py")
    res = cm.apply_modification("sample.py", "def (:\n", "broken")
    assert res["status"] == "syntax_error"
    assert cm.read_source("sample.py") == before
    assert cm.failed_mods == 1


def test_apply_safety_blocked(cm):
    res = cm.apply_modification("sample.py", "import subprocess\n", "danger")
    assert res["status"] == "safety_blocked"
    assert "warnings" in res
    assert cm.blocked_mods == 1


def test_apply_ethics_blocked(cm):
    # modifying ethics_core is immutable, so use a non-immutable ethics-named file
    (cm.base_dir / "layers" / "ethics_notes.py").write_text(
        "E-001 E-002 E-003 E-004\n", encoding="utf-8"
    )
    res = cm.apply_modification("layers/ethics_notes.py", "removed = True\n", "strip axioms")
    assert res["status"] == "ethics_blocked"
    assert cm.blocked_mods == 1


def test_apply_success_creates_backup(cm):
    new_code = "x = 2\n\n\ndef f():\n    return x + 1\n"
    res = cm.apply_modification("sample.py", new_code, "tweak")
    assert res["status"] == "applied_pending_restart"
    assert res["lines_before"] > 0 and res["lines_after"] > 0
    assert "backup" in res and Path(res["backup"]).exists()
    assert cm.read_source("sample.py") == new_code
    assert cm.successful_mods == 1


def test_apply_new_file_marked(cm):
    res = cm.apply_modification("created.py", "brand = 1\n", "add file")
    assert res["status"] == "applied_pending_restart"
    assert res.get("is_new_file") is True
    assert res["lines_before"] == 0
    assert (cm.base_dir / "created.py").exists()


def test_apply_compile_failure_rolls_back_existing(cm):
    before = cm.read_source("sample.py")
    # 'return' outside a function parses via ast but fails full compile.
    res = cm.apply_modification("sample.py", "return 5\n", "bad compile")
    assert res["status"] == "compile_failed_rolled_back"
    assert "error" in res
    assert cm.read_source("sample.py") == before  # restored
    assert cm.failed_mods == 1


def test_apply_compile_failure_removes_new_file(cm):
    res = cm.apply_modification("ghost.py", "return 5\n", "bad new file")
    assert res["status"] == "compile_failed_rolled_back"
    assert not (cm.base_dir / "ghost.py").exists()


def test_apply_write_failure(cm, monkeypatch):
    orig_write = Path.write_text

    def failing_write(self, data, *a, **k):
        if self.name == "sample.py":
            raise OSError("disk full")
        return orig_write(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", failing_write)
    res = cm.apply_modification("sample.py", "x = 42\n", "will fail")
    assert res["status"] == "write_failed"
    assert "disk full" in res["error"]
    assert cm.failed_mods == 1


# ── Rollback ──────────────────────────────────────────────────────────

def test_rollback_last_empty(cm):
    res = cm.rollback_last()
    assert res["success"] is False


def test_rollback_last_restores(cm):
    cm.apply_modification("sample.py", "x = 99\n", "m1")
    res = cm.rollback_last()
    assert res["success"] is True
    assert "x = 1" in cm.read_source("sample.py")


def test_rollback_last_new_file_deletes(cm):
    cm.apply_modification("temp_new.py", "brand = 1\n", "new")
    assert (cm.base_dir / "temp_new.py").exists()
    res = cm.rollback_last()
    assert res["success"] is True
    assert not (cm.base_dir / "temp_new.py").exists()


def test_rollback_last_failure_reappends(cm, monkeypatch):
    cm.apply_modification("sample.py", "x = 7\n", "m")
    orig_write = Path.write_text

    def boom(self, data, *a, **k):
        if self.name == "sample.py":
            raise OSError("locked")
        return orig_write(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)
    res = cm.rollback_last()
    assert res["success"] is False
    assert len(cm.rollback_stack) == 1  # entry restored for retry


def test_rollback_to_unknown_id_noop(cm):
    res = cm.rollback_to("cmod_9999")
    assert res["success"] is False
    assert res["rolled_back"] == []


def test_rollback_to_walks_stack(cm):
    r1 = cm.apply_modification("sample.py", "x = 10\n", "m1")
    cm.apply_modification("layers/helper.py", "y = 20\n", "m2")
    res = cm.rollback_to(r1["id"])
    assert res["success"] is True
    # both mods rolled back down to and including m1
    assert r1["file"] in res["rolled_back"]
    assert "x = 1" in cm.read_source("sample.py")


# ── Analysis ──────────────────────────────────────────────────────────

def test_analyze_file_structure(cm):
    (cm.base_dir / "mod.py").write_text(
        "import os\nfrom collections import deque\n\n"
        "class A:\n    def m(self):\n        return 1\n\n"
        "def top():\n    return 2\n",
        encoding="utf-8",
    )
    info = cm.analyze_file("mod.py")
    assert any(c["name"] == "A" and "m" in c["methods"] for c in info["classes"])
    assert any(f["name"] == "top" for f in info["functions"])
    assert "os" in info["imports"] and "collections" in info["imports"]
    assert info["immutable"] is False


def test_analyze_file_syntax_error(cm):
    (cm.base_dir / "bad.py").write_text("def (:\n", encoding="utf-8")
    info = cm.analyze_file("bad.py")
    assert "error" in info


# ── Status & persistence ──────────────────────────────────────────────

def test_status_reports_counts(cm):
    cm.apply_modification("sample.py", "x = 3\n", "m")
    st = cm.status()
    assert st["total_modifications"] >= 1
    assert st["successful"] >= 1
    assert "success_rate" in st
    assert set(st["immutable_files"]) == set(IMMUTABLE_FILES)
    assert st["source_files"] >= 1


def test_stats_persist_and_reload(cm, pkg, tmp_path):
    cm.apply_modification("sample.py", "x = 5\n", "m")
    # a fresh instance on the same backups dir reloads stats
    cm2 = CodeModifier(pkg, tmp_path / "backups")
    assert cm2.total_mods >= 1
    assert cm2.successful_mods >= 1


def test_load_stats_corrupt_file_is_tolerated(pkg, tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    (backups / "code_mod_stats.json").write_text("{not json", encoding="utf-8")
    cm = CodeModifier(pkg, backups)  # must not raise
    assert cm.total_mods == 0
