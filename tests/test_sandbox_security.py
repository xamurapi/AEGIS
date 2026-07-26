"""Security regression tests for the skill sandbox static gate (check_safe).

Covers audit finding C1: escape via ``__builtins__`` used as a bare name.
"""
from aegis.eval.sandbox import check_safe, run_skill


def test_builtins_name_escape_is_blocked():
    # audit C1: `__builtins__.eval(...)` — __builtins__ is a bare Name (not in
    # FORBIDDEN_CALLS) and `.eval` is not a dunder attribute, so it slipped
    # through before. It must now be rejected.
    code = "def solve(p):\n    return __builtins__.eval(\"1+1\")\n"
    safe, reasons = check_safe(code)
    assert safe is False
    assert any("__builtins__" in r for r in reasons)


def test_builtins_import_escape_is_blocked():
    code = ("def solve(p):\n"
            "    return __builtins__.__import__('os').system('echo hi')\n")
    safe, reasons = check_safe(code)
    assert safe is False


def test_other_dunder_names_blocked():
    for expr in ("__loader__", "__class__.__bases__", "__globals__"):
        code = f"def solve(p):\n    return {expr}\n"
        safe, reasons = check_safe(code)
        assert safe is False, f"{expr} should be rejected"


def test_run_skill_refuses_builtins_escape():
    # End to end: the escape never reaches the subprocess.
    code = "def solve(p):\n    return __builtins__.eval('2+2')\n"
    out = run_skill(code, "solve", None, timeout=3.0)
    assert out["ok"] is False
    assert "unsafe" in out["error"]


def test_legitimate_pure_skill_still_allowed():
    # A normal compute skill must still pass the gate.
    code = ("import math\n"
            "def solve(p):\n"
            "    return sorted(p, key=lambda x: math.sqrt(abs(x)))\n")
    safe, reasons = check_safe(code)
    assert safe is True, reasons


def test_parameter_named_like_builtin_still_allowed():
    # A param shadowing a builtin name is not the builtin — must not false-flag.
    code = "def solve(input):\n    return sorted(input)\n"
    safe, reasons = check_safe(code)
    assert safe is True, reasons
