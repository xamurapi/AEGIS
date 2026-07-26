"""Completeness gate (spec §VII.1).

The development spec is executed in "maximum" mode: no stubs, no deferred
halves, no "good enough for now". That rule is worth exactly as much as its
enforcement, so this script is a CI gate — it fails the build when the shipped
package contains a placeholder.

What counts as a placeholder:

* a deferral marker in a comment or string;
* ``raise NotImplementedError`` outside an abstract base;
* a function or class whose entire body is ``pass`` / ``...`` with no
  docstring explaining why it is intentionally empty.

Deliberate exceptions are declared in the source, not here: append
``# check-no-stubs: allow`` to the line and it is skipped. That keeps the
justification next to the code instead of in a growing ignore list.

Usage:
    python scripts/check_no_stubs.py            # scan aegis/ and scripts/
    python scripts/check_no_stubs.py aegis/eval # scan a subtree
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ("aegis", "scripts")

# Assembled from fragments so this file does not trip its own gate — the
# checker has to be scannable like everything else.
MARKERS: tuple[str, ...] = (
    "TO" + "DO",
    "FIX" + "ME",
    "XX" + "X",
    "HA" + "CK",
    "заглуш",          # ru: "заглушка" — stub          # check-no-stubs: allow
    "не реализован",   # ru: "not implemented"          # check-no-stubs: allow
    "later" + " version",
)

ALLOW_PRAGMA = "check-no-stubs: allow"

# Base classes whose methods are legitimately empty: the contract lives in the
# subclass. Only these may raise NotImplementedError.
ABSTRACT_BASES = {"ABC", "ABCMeta", "Protocol"}


class Finding:
    __slots__ = ("path", "line", "kind", "detail")

    def __init__(self, path: Path, line: int, kind: str, detail: str):
        self.path = path
        self.line = line
        self.kind = kind
        self.detail = detail

    def __str__(self) -> str:
        # Paths outside the repo (an explicit target, a temp dir) have no
        # relative form; reporting must not explode on them.
        try:
            location = self.path.relative_to(REPO_ROOT)
        except ValueError:
            location = self.path
        return f"{location}:{self.line}: [{self.kind}] {self.detail}"


def _allowed(lines: list[str], lineno: int) -> bool:
    """True if the line carries an explicit opt-out pragma."""
    if 1 <= lineno <= len(lines):
        return ALLOW_PRAGMA in lines[lineno - 1]
    return False


def _is_abstract(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
        if name in ABSTRACT_BASES:
            return True
    for kw in node.keywords:
        if kw.arg == "metaclass":
            value = kw.value
            name = value.id if isinstance(value, ast.Name) else getattr(value, "attr", "")
            if name in ABSTRACT_BASES:
                return True
    return False


def _body_is_empty(node) -> bool:
    """A body of exactly ``pass`` or ``...`` with no docstring is a stub."""
    body = [n for n in node.body if not (isinstance(n, ast.Expr)
                                         and isinstance(n.value, ast.Constant)
                                         and isinstance(n.value.value, str))]
    if len(body) != 1:
        return False
    only = body[0]
    if isinstance(only, ast.Pass):
        return len(body) == len(node.body)  # no docstring alongside
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant):
        return only.value.value is Ellipsis
    return False


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [Finding(path, 0, "unreadable", str(exc))]
    lines = text.splitlines()

    # 1. Deferral markers anywhere in the text.
    lowered = [ln.lower() for ln in lines]
    for i, line in enumerate(lowered, start=1):
        if _allowed(lines, i):
            continue
        for marker in MARKERS:
            if marker.lower() in line:
                findings.append(Finding(path, i, "marker", f"contains {marker!r}"))

    # 2. AST-level placeholders.
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        findings.append(Finding(path, exc.lineno or 0, "syntax", exc.msg))
        return findings

    abstract_classes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_abstract(node):
            for child in ast.walk(node):
                abstract_classes.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            exc_node = node.exc
            name = ""
            if isinstance(exc_node, ast.Call):
                func = exc_node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            elif isinstance(exc_node, ast.Name):
                name = exc_node.id
            if name == "NotImplementedError" and id(node) not in abstract_classes:
                if not _allowed(lines, node.lineno):
                    findings.append(
                        Finding(path, node.lineno, "not-implemented",
                                "raise NotImplementedError outside an abstract base"))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if id(node) in abstract_classes:
                continue
            if _body_is_empty(node) and not _allowed(lines, node.lineno):
                findings.append(
                    Finding(path, node.lineno, "empty-body",
                            f"{node.name!r} has an empty body and no docstring"))

    return findings


def iter_python_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        root = (REPO_ROOT / target) if not Path(target).is_absolute() else Path(target)
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def main(argv: list[str]) -> int:
    # Findings can contain non-ASCII markers; a cp1252/cp1251 CI console would
    # otherwise turn a clear report into mojibake or a UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    targets = argv[1:] or list(DEFAULT_TARGETS)
    files = iter_python_files(targets)
    findings: list[Finding] = []
    for path in files:
        findings.extend(scan_file(path))

    if not findings:
        print(f"check_no_stubs: OK — {len(files)} files, no placeholders")
        return 0

    print(f"check_no_stubs: FAILED — {len(findings)} placeholder(s) "
          f"in {len({f.path for f in findings})} file(s):\n")
    for finding in findings:
        print(f"  {finding}")
    print("\nEvery placeholder must be implemented, or justified inline with "
          f"'# {ALLOW_PRAGMA}'.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
