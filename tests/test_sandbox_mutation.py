"""Mutation-driven tests for the sandbox — the module with the highest blast
radius in the tree (it is the only barrier between self-written skill code and
the host).

Each test here was written to kill a specific surviving mutant reported by
``python scripts/mutation_test.py sandbox``. They pin two things the previous
suite left unspecified:

  * the PRECISION of the safety gate (what it must NOT reject), and
  * every failure path of the two subprocess runners (timeout, spawn failure,
    missing result marker, unparseable result) — none of which was exercised,
    so the whole error-handling half of the module was unpinned.
"""
import subprocess

import pytest

from aegis.eval import sandbox
from aegis.eval.sandbox import check_safe, run_skill, run_tests


# ══ check_safe: precision (what must still be ACCEPTED) ═══════════════

def test_allowed_from_import_is_accepted():
    """`from math import sqrt` must pass — the import root is checked against
    SAFE_IMPORTS, not replaced by an empty string."""
    safe, reasons = check_safe("from math import sqrt\ndef solve(p):\n    return sqrt(p)\n")
    assert safe is True, reasons


def test_relative_import_is_rejected():
    """`from . import x` has no module name and must not be treated as safe."""
    safe, _ = check_safe("from . import helper\ndef solve(p):\n    return 1\n")
    assert safe is False


def test_name_mangled_private_is_not_a_dunder():
    """`__cache` starts with two underscores but is NOT a dunder — the gate
    blocks dunders, and must not over-block ordinary private names."""
    safe, reasons = check_safe("__cache = {}\ndef solve(p):\n    return len(__cache)\n")
    assert safe is True, reasons


def test_name_mangled_private_attribute_is_not_a_dunder():
    safe, reasons = check_safe("def solve(p):\n    return p.__cache\n")
    assert safe is True, reasons


def test_trailing_underscore_name_is_not_a_dunder():
    safe, reasons = check_safe("def solve(p):\n    total__ = p\n    return total__\n")
    assert safe is True, reasons


def test_real_dunder_name_is_still_rejected():
    safe, _ = check_safe("def solve(p):\n    return __builtins__\n")
    assert safe is False


def test_real_dunder_attribute_is_still_rejected():
    safe, _ = check_safe("def solve(p):\n    return p.__class__\n")
    assert safe is False


def test_syntactically_invalid_code_is_rejected():
    """A parse failure must be a REJECTION, not an accidental pass."""
    safe, reasons = check_safe("def solve(p)\n    return 1\n")
    assert safe is False
    assert any("syntax error" in r for r in reasons)


# ══ run_skill: every failure path ═════════════════════════════════════

def test_run_skill_rejects_non_identifier_function_name():
    """`func` is interpolated into the runner source — a non-identifier would
    execute arbitrary code in the child, bypassing check_safe entirely."""
    out = run_skill("def solve(p):\n    return 1\n", "__import__('os').getcwd()", None)
    assert out["ok"] is False
    assert "invalid function name" in out["error"]


def test_run_skill_rejects_unsafe_code():
    out = run_skill("def solve(p):\n    return eval('1')\n", "solve", None)
    assert out["ok"] is False
    assert "unsafe code" in out["error"]


def test_run_skill_times_out_on_a_hanging_skill():
    out = run_skill("def solve(p):\n    while True:\n        pass\n", "solve", None, timeout=1.0)
    assert out["ok"] is False
    assert "timeout" in out["error"]


def test_run_skill_reports_a_spawn_failure_without_raising(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("cannot spawn")

    monkeypatch.setattr(sandbox.subprocess, "run", _boom)
    out = run_skill("def solve(p):\n    return 1\n", "solve", None)
    assert out["ok"] is False
    assert "sandbox error" in out["error"]


def test_run_skill_reports_missing_result_marker():
    """The child died before reaching the runner block (module-level error), so
    stdout carries no result marker at all."""
    out = run_skill("1 / 0\ndef solve(p):\n    return 1\n", "solve", None, timeout=10.0)
    assert out["ok"] is False
    assert "no result" in out["error"]


def test_run_skill_reports_unparseable_result():
    """A marker is present but the payload after it is not JSON."""
    code = ('print("__AEGIS__{not json")\n'
            "1 / 0\n"
            "def solve(p):\n    return 1\n")
    out = run_skill(code, "solve", None, timeout=10.0)
    assert out["ok"] is False
    assert out["error"] == "unparseable result"


def test_run_skill_happy_path():
    out = run_skill("def solve(p):\n    return p['a'] + p['b']\n", "solve", {"a": 2, "b": 3})
    assert out == {"ok": True, "result": 5}


# ══ run_tests: the batch runner had NO successful-path coverage ═══════

def test_run_tests_executes_every_case():
    code = "def add(a, b):\n    return a + b\n"
    out = run_tests(code, "add", [[1, 2], [10, 5]])
    assert out["ok"] is True
    assert [r["result"] for r in out["results"]] == [3, 15]


def test_run_tests_reports_per_case_errors_without_failing_the_batch():
    code = "def div(a, b):\n    return a / b\n"
    out = run_tests(code, "div", [[6, 3], [1, 0]])
    assert out["ok"] is True
    assert out["results"][0] == {"ok": True, "result": 2}
    assert out["results"][1]["ok"] is False
    assert "ZeroDivisionError" in out["results"][1]["error"]


def test_run_tests_rejects_non_identifier_function_name():
    out = run_tests("def solve(p):\n    return 1\n", "os.system", [[1]])
    assert out["ok"] is False
    assert "invalid function name" in out["error"]
    assert out["results"] == []


def test_run_tests_rejects_unsafe_code():
    out = run_tests("import os\ndef solve(p):\n    return 1\n", "solve", [[1]])
    assert out["ok"] is False
    assert "unsafe code" in out["error"]
    assert out["results"] == []


def test_run_tests_times_out_on_a_hanging_function():
    out = run_tests("def spin(x):\n    while True:\n        pass\n", "spin", [[1]], timeout=1.0)
    assert out["ok"] is False
    assert "timeout" in out["error"]
    assert out["results"] == []


def test_run_tests_reports_a_spawn_failure_without_raising(monkeypatch):
    def _boom(*a, **kw):
        raise OSError("cannot spawn")

    monkeypatch.setattr(sandbox.subprocess, "run", _boom)
    out = run_tests("def f(x):\n    return x\n", "f", [[1]])
    assert out["ok"] is False
    assert "sandbox error" in out["error"]
    assert out["results"] == []


def test_run_tests_reports_missing_result_marker():
    out = run_tests("1 / 0\ndef f(x):\n    return x\n", "f", [[1]], timeout=10.0)
    assert out["ok"] is False
    assert "no result" in out["error"]
    assert out["results"] == []


def test_run_tests_reports_unparseable_result():
    code = ('print("__AEGIS__{not json")\n'
            "1 / 0\n"
            "def f(x):\n    return x\n")
    out = run_tests(code, "f", [[1]], timeout=10.0)
    assert out["ok"] is False
    assert out["error"] == "unparseable result"
    assert out["results"] == []


# ══ the temp script must never linger on disk ═════════════════════════

def test_temp_script_is_removed_after_a_run(tmp_path, monkeypatch):
    created = []
    real_writer = sandbox._write_temp_script

    def tracking_writer(script):
        path = real_writer(script)
        created.append(path)
        return path

    monkeypatch.setattr(sandbox, "_write_temp_script", tracking_writer)
    run_skill("def solve(p):\n    return 1\n", "solve", None)
    assert created, "no temp script was created"
    assert not created[0].exists(), "sandbox left a generated script on disk"
