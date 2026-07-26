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

# Stdlib modules a pure-compute skill may import. ``random`` is deliberately
# EXCLUDED — skills solve deterministic, exactly-verified benchmark tasks, and
# the system carries a project-wide "zero randomness" guarantee.
SAFE_IMPORTS = {
    "math", "cmath", "statistics", "itertools", "functools", "operator",
    "re", "json", "collections", "string", "decimal", "fractions",
    "heapq", "bisect", "datetime", "typing",
}
# Names that must never be called from skill code.
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
    "breakpoint",
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


def _arg_names(args: ast.arguments) -> set[str]:
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


class _SafetyVisitor(ast.NodeVisitor):
    """Scope-aware safety gate.

    A parameter named ``input`` shadows a builtin ONLY inside its own function,
    so the exemption must be scoped. The previous flat "all params anywhere"
    set was a sandbox-escape (audit): ``_ = lambda eval: None`` bound ``eval``
    in a throw-away lambda and thereby un-blocked the real builtin ``eval`` at
    module scope, giving RCE via ``eval("__import__('os')...")``. Here the
    exemption only applies to names actually bound as parameters in an
    ENCLOSING function of the reference.
    """

    def __init__(self):
        self.reasons: list[str] = []
        self._scope: list[set[str]] = [set()]  # stack of accumulated bound params

    # ── scope-introducing nodes ──
    def _visit_signature(self, args: ast.arguments):
        """Visit every part of a signature that Python EVALUATES at definition
        time, in the enclosing scope: default values and parameter annotations.

        Annotations are real expressions evaluated when the ``def`` executes
        (there is no ``from __future__ import annotations`` in generated skill
        code), so leaving them unchecked was a sandbox escape (audit R3-1):
        ``def solve(p, _z: __import__('os').system('...')): ...`` passed
        check_safe and then ran arbitrary code in the child process.
        """
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs,
                  args.vararg, args.kwarg):
            if a is not None and a.annotation is not None:
                self.visit(a.annotation)

    def _visit_function(self, node):
        # Decorators, defaults and annotations are evaluated in the OUTER scope.
        for d in node.decorator_list:
            self.visit(d)
        self._visit_signature(node.args)
        if node.returns is not None:
            self.visit(node.returns)          # `def f() -> <expr>` also runs
        for tp in getattr(node, "type_params", ()):   # PEP 695 generics
            self.visit(tp)
        self._scope.append(self._scope[-1] | _arg_names(node.args))
        for stmt in node.body:
            self.visit(stmt)
        self._scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node):
        # A lambda has no annotations, but it CAN carry keyword-only defaults
        # (`lambda *, a=<expr>: ...`) which are evaluated in the outer scope.
        self._visit_signature(node.args)
        self._scope.append(self._scope[-1] | _arg_names(node.args))
        self.visit(node.body)
        self._scope.pop()

    # ── checks ──
    def visit_Import(self, node):
        for a in node.names:
            if a.name.split(".")[0] not in SAFE_IMPORTS:
                self.reasons.append(f"forbidden import '{a.name}'")

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root not in SAFE_IMPORTS:
            self.reasons.append(f"forbidden import from '{node.module}'")

    def visit_Name(self, node):
        nid = node.id
        if nid in self._scope[-1]:
            return  # genuinely shadowed by an enclosing parameter
        if nid in FORBIDDEN_CALLS:
            self.reasons.append(f"forbidden name '{nid}'")
        elif nid.startswith("__") and nid.endswith("__"):
            self.reasons.append(f"forbidden dunder name '{nid}'")

    def visit_Attribute(self, node):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.reasons.append(f"forbidden dunder attribute '{node.attr}'")
        self.generic_visit(node)


def check_safe(code: str) -> tuple[bool, list[str]]:
    """Static safety gate for skill code. Returns (safe, reasons)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax error: {e.msg} (line {e.lineno})"]
    v = _SafetyVisitor()
    v.visit(tree)
    return (len(v.reasons) == 0), v.reasons


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
