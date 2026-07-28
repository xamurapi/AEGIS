"""Undefined-name gate: a name used but never bound anywhere in its module.

This exists because of a defect that reached the audit. ``phases/act.py`` built
an ``Event(...)`` and read ``Layer.GOAL_ENGINE`` without importing either. The
line sat on a *success* branch — a curiosity call that came back with a topic —
so every test that exercised the phase took the failure path instead and the
suite stayed green while the phase aborted halfway through in production.

Coverage would not have caught it (the branch was uncovered, and the module was
still at 88%). Mutation testing would not have caught it (there was nothing to
mutate). What catches it is asking a much simpler question of every module:

    is every name this module reads bound somewhere it can see?

The check is deliberately conservative — it looks for names that appear nowhere
at all in the module: not imported, not assigned, not a parameter, not a
builtin, not a global declaration. That has no false positives worth arguing
about, and it is exactly the failure above.

Run it directly, or through ``tests/test_no_undefined_names.py``:

    python scripts/check_undefined_names.py            # whole aegis package
    python scripts/check_undefined_names.py aegis/api  # one subtree
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BUILTINS = frozenset(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__",
}


class _Bindings(ast.NodeVisitor):
    """Every name the module binds, anywhere, at any nesting depth.

    Scope is deliberately ignored. A name bound in one function and read in
    another is a different (and much rarer) bug than the one this gate is for,
    and pretending to resolve Python scoping properly is how a checker like this
    grows false positives until people switch it off.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()

    # -- imports ------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)
        self.generic_visit(node)

    # -- definitions --------------------------------------------------
    def _function(self, node) -> None:
        self.bound.add(node.name)
        args = node.args
        for arg in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)
        self.generic_visit(node)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        for arg in (list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)):
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    # -- assignment targets and other binding forms -------------------
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.bound.update(node.names)
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.bound.update(node.names)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_MatchMapping(self, node) -> None:
        if node.rest:
            self.bound.add(node.rest)
        self.generic_visit(node)


def _loads(tree: ast.AST) -> list[tuple[str, int]]:
    """Every name read, with the line it was read on."""
    return [(node.id, node.lineno) for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]


def undefined_in(path: Path) -> list[tuple[str, int]]:
    """Names this file reads but never binds. Sorted for a stable report."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:      # pragma: no cover
        return [(f"<unparseable: {exc}>", 0)]

    collector = _Bindings()
    collector.visit(tree)
    known = collector.bound | BUILTINS
    seen: set[str] = set()
    findings = []
    for name, line in _loads(tree):
        if name in known or name in seen:
            continue
        seen.add(name)
        findings.append((name, line))
    return sorted(findings, key=lambda item: (item[1], item[0]))


def scan(*roots: str | Path) -> dict[str, list[tuple[str, int]]]:
    """Findings per file, keyed by a repo-relative path."""
    targets = [Path(root) for root in roots] or [ROOT / "aegis"]
    report: dict[str, list[tuple[str, int]]] = {}
    for target in targets:
        base = target if target.is_absolute() else ROOT / target
        files = sorted(base.rglob("*.py")) if base.is_dir() else [base]
        for path in files:
            if "__pycache__" in path.parts:
                continue
            findings = undefined_in(path)
            if findings:
                report[_label(path)] = findings
    return report


def _label(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    A path outside the tree is a legitimate argument (a scratch file, a single
    module under review); reporting it must not be the thing that fails.
    """
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main(argv: list[str]) -> int:
    report = scan(*argv[1:])
    if not report:
        print("check_undefined_names: OK — every name is bound")
        return 0
    for filename, findings in sorted(report.items()):
        for name, line in findings:
            print(f"{filename}:{line}: undefined name {name!r}")
    total = sum(len(v) for v in report.values())
    print(f"check_undefined_names: FAIL — {total} undefined name(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
