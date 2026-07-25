"""Point 4 (infrastructure) — execute skill code in an ISOLATED subprocess.

Self-written skills never run in the main process. They are:
  1. AST-checked (no dangerous imports/calls, no dunder access) — code that
     fails this is rejected without ever executing.
  2. Run via ``python -I`` (isolated mode: no env vars, no user site, cwd off
     sys.path) in a child process with a hard wall-clock timeout.

This is defense-in-depth, not a perfect jail (a real deployment should add an
OS-level sandbox / container + network egress block). Skills are meant to be
PURE functions ``solve(payload) -> answer``; only a small stdlib allowlist of
compute modules may be imported.
"""
import ast
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Stdlib modules a pure-compute skill may import.
SAFE_IMPORTS = {
    "math", "cmath", "statistics", "itertools", "functools", "operator",
    "re", "json", "collections", "string", "decimal", "fractions",
    "heapq", "bisect", "datetime", "random", "typing",
}
# Names that must never be called from skill code.
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
}


def _write_temp_script(script: str) -> Path:
    """Create a UNIQUE, private temp file and write the script to it.

    Uses mkstemp (O_EXCL, mode 0600) rather than a predictable, shared path —
    a fixed name derived from hash(code) is both a local symlink/TOCTOU hazard
    and a concurrency race (two runs of the same code would clobber and unlink
    each other's file)."""
    fd, name = tempfile.mkstemp(prefix="aegis_skill_", suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(script)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)


def _param_names(tree: ast.AST) -> set[str]:
    """Collect names bound as function/lambda parameters anywhere in the tree.

    A parameter named ``input`` (or ``vars``, ``open``, …) shadows the builtin
    inside its function, so a reference to it is NOT the dangerous builtin — we
    exempt those to avoid false-positives on ordinary code like
    ``def solve(input): return sorted(input)``. Assignment targets are
    deliberately NOT exempted, so the aliasing escape ``_i = __import__`` (whose
    right-hand side is a genuine builtin reference) is still caught."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                names.add(arg.arg)
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)
    return names


def check_safe(code: str) -> tuple[bool, list[str]]:
    """Static safety gate for skill code. Returns (safe, reasons)."""
    reasons: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e.msg} (line {e.lineno})"]

    params = _param_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in SAFE_IMPORTS:
                    reasons.append(f"forbidden import '{a.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in SAFE_IMPORTS:
                reasons.append(f"forbidden import from '{node.module}'")
        elif isinstance(node, ast.Name):
            # Flag every *reference* to a forbidden name, not just direct calls.
            # This catches aliasing escapes like ``_i = __import__; _i('os')``
            # and ``g = getattr`` that a call-site-only check would miss, while
            # exempting parameters that merely shadow a builtin name.
            if node.id not in params:
                if node.id in FORBIDDEN_CALLS:
                    reasons.append(f"forbidden name '{node.id}'")
                elif node.id.startswith("__") and node.id.endswith("__"):
                    # Block ALL dunder *names*, not just dunder attributes. The
                    # dunder-attribute rule below misses ``__builtins__`` /
                    # ``__loader__`` used as a bare name — e.g.
                    # ``__builtins__.eval("...")`` — because ``.eval`` is not a
                    # dunder attribute and ``__builtins__`` is not a Name in
                    # FORBIDDEN_CALLS. That was a real sandbox-escape (audit C1).
                    reasons.append(f"forbidden dunder name '{node.id}'")
        elif isinstance(node, ast.Attribute):
            # Block dunder attribute escapes like obj.__globals__ / __builtins__.
            if node.attr.startswith("__") and node.attr.endswith("__"):
                reasons.append(f"forbidden dunder attribute '{node.attr}'")
    return (len(reasons) == 0), reasons


_RUNNER = """{code}

if __name__ == "__main__":
    import sys, json
    _raw = sys.stdin.read()
    _payload = json.loads(_raw) if _raw.strip() else None
    try:
        _res = {func}(_payload)
        sys.stdout.write("__AEGIS__" + json.dumps({{"ok": True, "result": _res}}))
    except Exception as _e:
        sys.stdout.write("__AEGIS__" + json.dumps({{"ok": False, "error": repr(_e)}}))
"""


def run_skill(code: str, func: str, payload, timeout: float = 3.0) -> dict:
    """Run ``func(payload)`` from ``code`` in an isolated subprocess.

    Returns {"ok": bool, "result"/"error": ...}. Never raises.
    """
    if not func.isidentifier():
        # ``func`` is interpolated into the runner source; a non-identifier
        # (e.g. "__import__('os').system('...')") would execute arbitrary code
        # in the subprocess, bypassing check_safe (which only inspects ``code``).
        return {"ok": False, "error": f"invalid function name: {func!r}"}
    safe, reasons = check_safe(code)
    if not safe:
        return {"ok": False, "error": f"unsafe code: {reasons}"}

    script = _RUNNER.format(code=code, func=func)
    tmp = _write_temp_script(script)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            input=json.dumps(payload),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": f"sandbox error: {e!r}"}
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    out = proc.stdout
    marker = out.rfind("__AEGIS__")
    if marker == -1:
        return {"ok": False, "error": f"no result (stderr: {proc.stderr[:200]})"}
    try:
        return json.loads(out[marker + len("__AEGIS__"):])
    except json.JSONDecodeError:
        return {"ok": False, "error": "unparseable result"}


_TEST_RUNNER = """{code}

if __name__ == "__main__":
    import sys, json
    _cases = json.loads(sys.stdin.read() or "[]")
    _out = []
    for _args in _cases:
        try:
            _out.append({{"ok": True, "result": {func}(*_args)}})
        except Exception as _e:
            _out.append({{"ok": False, "error": repr(_e)}})
    sys.stdout.write("__AEGIS__" + json.dumps(_out))
"""


def run_tests(code: str, func: str, arg_lists: list, timeout: float = 3.0) -> dict:
    """Run ``func(*args)`` for each arg list in one isolated subprocess.

    Used by the coding benchmark to execute a candidate function against many
    hidden test cases at once. Returns {"ok": bool, "results": [...]}.
    """
    if not func.isidentifier():
        return {"ok": False, "error": f"invalid function name: {func!r}", "results": []}
    safe, reasons = check_safe(code)
    if not safe:
        return {"ok": False, "error": f"unsafe code: {reasons}", "results": []}

    script = _TEST_RUNNER.format(code=code, func=func)
    tmp = _write_temp_script(script)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(tmp)],
            input=json.dumps(arg_lists),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s", "results": []}
    except Exception as e:
        return {"ok": False, "error": f"sandbox error: {e!r}", "results": []}
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    out = proc.stdout
    marker = out.rfind("__AEGIS__")
    if marker == -1:
        return {"ok": False, "error": f"no result (stderr: {proc.stderr[:200]})", "results": []}
    try:
        return {"ok": True, "results": json.loads(out[marker + len("__AEGIS__"):])}
    except json.JSONDecodeError:
        return {"ok": False, "error": "unparseable result", "results": []}
