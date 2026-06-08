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


def check_safe(code: str) -> tuple[bool, list[str]]:
    """Static safety gate for skill code. Returns (safe, reasons)."""
    reasons: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e.msg} (line {e.lineno})"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in SAFE_IMPORTS:
                    reasons.append(f"forbidden import '{a.name}'")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in SAFE_IMPORTS:
                reasons.append(f"forbidden import from '{node.module}'")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                reasons.append(f"forbidden call '{node.func.id}(...)'")
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
    safe, reasons = check_safe(code)
    if not safe:
        return {"ok": False, "error": f"unsafe code: {reasons}"}

    script = _RUNNER.format(code=code, func=func)
    tmp = Path(tempfile.gettempdir()) / f"aegis_skill_{abs(hash(code)) % (10**9)}.py"
    try:
        tmp.write_text(script, encoding="utf-8")
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
