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


# ── audit R5-1: attribute access spelled as a string ──────────────────
#
# The gate reads names and attributes out of the AST. A dunder handed to
# something that performs the lookup at runtime — `operator.attrgetter`, a dict
# subscript — is invisible to it. The escape below was verified end to end
# before the fix: `check_safe` returned (True, []) and `run_skill` returned the
# working directory from the child process, i.e. full RCE from a self-written
# skill.

_ATTRGETTER_ESCAPE = '''
import json, operator
def solve(payload):
    g = operator.attrgetter("__globals__")(json.dumps)
    b = g["__builtins__"]
    imp = b["__import__"] if isinstance(b, dict) else operator.attrgetter("__import__")(b)
    return operator.attrgetter("getcwd")(imp("os"))()
'''


def test_attrgetter_escape_is_blocked():
    safe, reasons = check_safe(_ATTRGETTER_ESCAPE)
    assert safe is False
    assert any("attrgetter" in r for r in reasons)
    assert any("__globals__" in r for r in reasons)


def test_run_skill_refuses_the_attrgetter_escape():
    out = run_skill(_ATTRGETTER_ESCAPE, "solve", None, timeout=5.0)
    assert out["ok"] is False
    assert "unsafe" in out["error"]
    # Whatever else it returned, it must not be a path from this machine.
    assert "result" not in out


def test_a_dunder_spelled_as_a_string_is_still_a_dunder():
    for expr in ('payload["__builtins__"]', 'payload.get("__globals__")',
                 'operator.methodcaller("__reduce__")(payload)'):
        code = f"import operator\ndef solve(payload):\n    return {expr}\n"
        safe, reasons = check_safe(code)
        assert safe is False, expr


def test_methodcaller_is_refused_even_without_a_dunder():
    code = ("import operator\n"
            "def solve(payload):\n"
            "    return operator.methodcaller('upper')(payload)\n")
    safe, reasons = check_safe(code)
    assert safe is False
    assert any("methodcaller" in r for r in reasons)


def test_ordinary_strings_and_operator_use_are_untouched():
    """The fix must not cost a skill its normal vocabulary: `operator.add`,
    dict keys, and any string that is not a dunder stay allowed."""
    code = ("import operator, functools\n"
            "def solve(payload):\n"
            "    total = functools.reduce(operator.add, payload['values'], 0)\n"
            "    return {'sum': total, 'note': '__ok', 'k': payload['__']}\n")
    safe, reasons = check_safe(code)
    assert safe is True, reasons
    out = run_skill(code, "solve", {"values": [1, 2, 3], "__": "x"}, timeout=5.0)
    assert out["ok"] is True and out["result"]["sum"] == 6
