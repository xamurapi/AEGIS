"""Tests for the completeness gate (spec §VII.1)."""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_no_stubs.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_no_stubs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load()


def write(tmp_path: Path, source: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


# ── marker detection ─────────────────────────────────────────────────

@pytest.mark.parametrize("marker", ["TO" + "DO", "FIX" + "ME", "XX" + "X", "HA" + "CK"])
def test_deferral_markers_are_found(tmp_path, marker):
    path = write(tmp_path, f"def f():\n    return 1  # {marker}: finish this\n")
    kinds = [f.kind for f in gate.scan_file(path)]
    assert "marker" in kinds


def test_marker_is_case_insensitive(tmp_path):
    path = write(tmp_path, "x = 1  # to" + "do later\n")
    assert gate.scan_file(path)


def test_allow_pragma_suppresses_a_marker(tmp_path):
    path = write(
        tmp_path,
        "x = 1  # to" + "do kept on purpose  # check-no-stubs: allow\n",
    )
    assert gate.scan_file(path) == []


def test_clean_file_yields_no_findings(tmp_path):
    path = write(tmp_path, '"""Doc."""\n\n\ndef f():\n    return 42\n')
    assert gate.scan_file(path) == []


# ── NotImplementedError ──────────────────────────────────────────────

def test_not_implemented_is_flagged(tmp_path):
    path = write(tmp_path, "def f():\n    raise NotImplementedError\n")
    assert [f.kind for f in gate.scan_file(path)] == ["not-implemented"]


def test_not_implemented_call_form_is_flagged(tmp_path):
    path = write(tmp_path, "def f():\n    raise NotImplementedError('soon')\n")
    assert [f.kind for f in gate.scan_file(path)] == ["not-implemented"]


def test_not_implemented_inside_abstract_base_is_allowed(tmp_path):
    """An ABC method is a contract, not an unfinished one."""
    path = write(tmp_path, (
        "from abc import ABC\n\n"
        "class Provider(ABC):\n"
        "    def call(self):\n"
        "        raise NotImplementedError\n"
    ))
    assert gate.scan_file(path) == []


def test_not_implemented_inside_protocol_is_allowed(tmp_path):
    path = write(tmp_path, (
        "from typing import Protocol\n\n"
        "class P(Protocol):\n"
        "    def call(self):\n"
        "        ...\n"
    ))
    assert gate.scan_file(path) == []


def test_other_exceptions_are_not_flagged(tmp_path):
    path = write(tmp_path, "def f():\n    raise ValueError('bad')\n")
    assert gate.scan_file(path) == []


# ── empty bodies ─────────────────────────────────────────────────────

def test_pass_only_function_is_flagged(tmp_path):
    path = write(tmp_path, "def f():\n    pass\n")
    assert [f.kind for f in gate.scan_file(path)] == ["empty-body"]


def test_ellipsis_only_function_is_flagged(tmp_path):
    path = write(tmp_path, "def f():\n    ...\n")
    assert [f.kind for f in gate.scan_file(path)] == ["empty-body"]


def test_documented_empty_function_is_allowed(tmp_path):
    """An empty body with a docstring explaining why is a decision, not a gap —
    e.g. a save() that persists incrementally and has nothing to flush."""
    path = write(tmp_path, 'def save():\n    """Nothing to flush: rows are appended."""\n')
    assert gate.scan_file(path) == []


def test_pass_only_class_is_flagged(tmp_path):
    path = write(tmp_path, "class Thing:\n    pass\n")
    assert [f.kind for f in gate.scan_file(path)] == ["empty-body"]


def test_documented_exception_class_is_allowed(tmp_path):
    path = write(tmp_path, 'class MyError(Exception):\n    """Raised when X."""\n')
    assert gate.scan_file(path) == []


def test_pass_inside_except_is_not_a_stub(tmp_path):
    """`except OSError: pass` is deliberate error handling, not an empty body."""
    path = write(tmp_path, (
        "def f(p):\n"
        "    try:\n"
        "        p.unlink()\n"
        "    except OSError:\n"
        "        pass\n"
        "    return True\n"
    ))
    assert gate.scan_file(path) == []


def test_empty_body_allow_pragma(tmp_path):
    path = write(tmp_path, "def f():  # check-no-stubs: allow\n    pass\n")
    assert gate.scan_file(path) == []


# ── robustness ───────────────────────────────────────────────────────

def test_syntax_error_is_reported_not_raised(tmp_path):
    path = write(tmp_path, "def broken(:\n")
    assert [f.kind for f in gate.scan_file(path)] == ["syntax"]


def test_finding_renders_path_line_and_kind(tmp_path):
    path = write(tmp_path, "def f():\n    pass\n")
    text = str(gate.scan_file(path)[0])
    assert "sample.py" in text and ":1:" in text and "empty-body" in text


def test_iter_python_files_skips_pycache(tmp_path):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    found = gate.iter_python_files([str(tmp_path)])
    assert [p.name for p in found] == ["real.py"]


def test_iter_python_files_accepts_a_single_file(tmp_path):
    path = write(tmp_path, "x = 1\n")
    assert gate.iter_python_files([str(path)]) == [path]


# ── the gate itself ──────────────────────────────────────────────────

def test_package_is_free_of_placeholders():
    """The real gate: the shipped package must contain no stubs."""
    assert gate.main(["check_no_stubs.py"]) == 0


def test_gate_fails_on_a_dirty_target(tmp_path):
    write(tmp_path, "def f():\n    pass\n")
    assert gate.main(["check_no_stubs.py", str(tmp_path)]) == 1
