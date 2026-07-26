"""Code Self-Modification — read, analyze, modify own source code with safety and rollback."""
import ast
import time
import json
import py_compile
import logging
from pathlib import Path

from aegis._atomic import atomic_write_text
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.code_modifier")

# Substring patterns that must NEVER appear in modified code. This is a cheap
# first pass; the authoritative check is the AST analysis below, which cannot be
# fooled by spacing tricks like "eval ( x )".
FORBIDDEN_PATTERNS = [
    "os.system",
    "os.popen",
    "subprocess",
    "__import__",
    "shutil.rmtree",
    "open('/etc",
    "open('C:\\\\Windows",
]

# Function names that are dangerous to *call* regardless of how they're spelled.
DANGEROUS_CALL_NAMES = {
    "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
}
# Dotted attribute calls that are dangerous (module.attr form).
DANGEROUS_ATTR_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "remove"), ("os", "unlink"),
    ("os", "rmdir"), ("os", "kill"), ("os", "_exit"),
    ("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
    ("subprocess", "check_output"), ("subprocess", "check_call"),
    ("shutil", "rmtree"), ("sys", "exit"),
}
# Imports that modified code may not introduce. importlib/builtins are included
# because they are generic escape hatches (importlib.import_module("subprocess"),
# builtins.__import__) around the direct-import block.
FORBIDDEN_IMPORTS = {"subprocess", "ctypes", "signal", "socket", "marshal",
                     "pickle", "importlib", "builtins"}

# Reflection-escape dunders (audit L6). These are the primitives used to break
# out of a name/import blocklist — e.g. ().__class__.__bases__[0].__subclasses__()
# to reach `os`, or func.__globals__ to reach the module namespace. They have no
# legitimate use in ordinary AEGIS source, so blocking these SPECIFIC dunders
# hardens the blocklist WITHOUT rejecting normal `__init__` / `__file__` /
# `if __name__ == "__main__"` that whole-file rewrites legitimately contain.
DANGEROUS_DUNDERS = {
    "__subclasses__", "__bases__", "__mro__", "__base__",
    "__globals__", "__builtins__", "__code__", "__closure__",
    "__getattribute__", "__reduce__", "__reduce_ex__",
}

# Files that are IMMUTABLE — cannot be modified by the system.
IMMUTABLE_FILES = {
    "layers/ethics_core.py",        # ethical axioms must never be changed by AI
    "layers/self_preservation.py",  # the watchdog must not edit itself away
    "config.py",                    # config changes need human approval
}

# Max diff size (chars) per single modification
MAX_MODIFICATION_SIZE = 5000


def _open_write_mode(call: ast.Call) -> str | None:
    """For an ``open(...)`` Call, return a reason string if the call opens a file
    for WRITING/APPENDING (arbitrary-file-write risk), else None.

    Read-only opens (no mode, or a literal 'r'/'rb'/'rt') are allowed. A mode
    that cannot be resolved to a read-only string literal is treated as unsafe —
    a self-modifying module has no business writing arbitrary files, and a
    dynamic mode (variable/expression) cannot be verified. (Audit A1.)
    """
    mode_node = None
    if len(call.args) >= 2:
        mode_node = call.args[1]
    else:
        for kw in call.keywords:
            if kw.arg == "mode":
                mode_node = kw.value
    if mode_node is None:
        return None  # no mode -> default 'r' (read) -> allowed
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        mode = mode_node.value
        if any(c in mode for c in "wax+"):
            return f"open() in write/append mode ('{mode}')"
        return None  # read-only literal -> allowed
    return "open() with a non-literal mode (cannot verify it is read-only)"


class CodeModifier:
    """Allows AEGIS to read, analyze, and safely modify its own Python source code."""

    def __init__(self, base_dir: Path, backups_dir: Path):
        self.base_dir = base_dir          # aegis/ package directory
        self.backups_dir = backups_dir     # directory for code backups
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        self.modifications: list[dict] = []
        self.rollback_stack: list[dict] = []
        self.total_mods = 0
        self.successful_mods = 0
        self.failed_mods = 0
        self.blocked_mods = 0

        self._stats_path = self.backups_dir / "code_mod_stats.json"
        self._load_stats()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_stats(self):
        if self._stats_path.exists():
            try:
                data = json.loads(self._stats_path.read_text(encoding="utf-8"))
                self.total_mods = data.get("total_mods", 0)
                self.successful_mods = data.get("successful_mods", 0)
                self.failed_mods = data.get("failed_mods", 0)
                self.blocked_mods = data.get("blocked_mods", 0)
                self.modifications = data.get("modifications", [])[-50:]
            except Exception:
                logger.warning("Failed to load code-mod stats from %s", self._stats_path, exc_info=True)

    def _save_stats(self):
        data = {
            "total_mods": self.total_mods,
            "successful_mods": self.successful_mods,
            "failed_mods": self.failed_mods,
            "blocked_mods": self.blocked_mods,
            "modifications": self.modifications[-50:],
            "last_updated": CLOCK.now(),
        }
        try:
            atomic_write_text(self._stats_path, json.dumps(data, indent=1))
        except Exception:
            pass

    # ── Path safety ──────────────────────────────────────────────────

    def _resolve_path(self, relative_path: str) -> tuple[Path, str]:
        """Resolve a caller-supplied path strictly inside ``base_dir``.

        Returns ``(absolute_path, normalized_relative_path)``. Raises
        ``ValueError`` for absolute paths, drive-relative paths, or any path
        that escapes the package directory (e.g. via ``..``) — otherwise the
        IMMUTABLE_FILES check could be bypassed with ``layers/../x`` tricks."""
        candidate = Path(relative_path)
        if candidate.is_absolute() or candidate.drive:
            raise ValueError(f"Absolute paths are not allowed: {relative_path}")
        base = self.base_dir.resolve()
        resolved = (base / candidate).resolve()
        try:
            rel = resolved.relative_to(base)
        except ValueError:
            raise ValueError(f"Path escapes package directory: {relative_path}")
        return resolved, str(rel).replace("\\", "/")

    # ── Read ─────────────────────────────────────────────────────────

    def read_source(self, relative_path: str) -> str:
        """Read a source file from the aegis package."""
        path, _ = self._resolve_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if path.suffix != ".py":
            raise ValueError("Only .py files can be read")
        return path.read_text(encoding="utf-8")

    def list_sources(self) -> list[dict]:
        """List all Python source files with metadata."""
        sources = []
        for p in sorted(self.base_dir.rglob("*.py")):
            if "__pycache__" in str(p):
                continue
            rel = str(p.relative_to(self.base_dir)).replace("\\", "/")
            content = p.read_text(encoding="utf-8")
            sources.append({
                "path": rel,
                "size": len(content),
                "lines": content.count("\n") + 1,
                "immutable": rel in IMMUTABLE_FILES,
            })
        return sources

    # ── Validation ───────────────────────────────────────────────────

    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """Check if code is valid Python by parsing the AST."""
        try:
            ast.parse(code)
            return True, "OK"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

    def validate_safety(self, code: str, relative_path: str) -> tuple[bool, list[str]]:
        """Check code for forbidden patterns and dangerous constructs."""
        warnings = []

        # Immutable file check — on the *resolved* path, so traversal tricks
        # ("layers/../config.py", "./config.py") cannot bypass it.
        try:
            _, norm = self._resolve_path(relative_path)
        except ValueError as e:
            return False, [f"BLOCKED: {e}"]
        if norm in IMMUTABLE_FILES:
            return False, [f"BLOCKED: {norm} is immutable — cannot be modified by AI"]

        # Forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in code:
                warnings.append(f"FORBIDDEN: dangerous pattern '{pattern}' detected")

        # Size check
        try:
            original = self.read_source(relative_path)
            diff_size = abs(len(code) - len(original))
            if diff_size > MAX_MODIFICATION_SIZE:
                warnings.append(f"BLOCKED: modification too large ({diff_size} chars > {MAX_MODIFICATION_SIZE} max)")
        except FileNotFoundError:
            pass  # new file, OK
        except ValueError as e:
            # e.g. non-.py target — surface as a block, don't let it escape.
            warnings.append(f"BLOCKED: {e}")

        # AST analysis — the authoritative check. Detects dangerous imports and
        # dangerous *calls* regardless of whitespace/formatting obfuscation.
        try:
            tree = ast.parse(code)
            # First pass: map local aliases back to their real module, so
            # "import os as o" doesn't let "o.kill(...)" slip past the
            # (module, attr) blocklist.
            alias_to_module: dict[str, str] = {}
            # Local names bound to a dangerous function via `from os import kill`
            # (possibly aliased) — a bare call to them must be treated as the
            # dangerous (module, attr) call it really is (audit: from-import
            # bypass). Maps local_name -> "module.attr".
            dangerous_from_aliases: dict[str, str] = {}
            _danger_modules = {m for m, _ in DANGEROUS_ATTR_CALLS}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".")[0]
                        alias_to_module[alias.asname or root] = root
                elif isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[0]
                    for alias in node.names:
                        if (mod, alias.name) in DANGEROUS_ATTR_CALLS:
                            dangerous_from_aliases[alias.asname or alias.name] = f"{mod}.{alias.name}"
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                            warnings.append(f"BLOCKED: forbidden import '{alias.name}'")
                elif isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".")[0]
                    if node.module and mod in FORBIDDEN_IMPORTS:
                        warnings.append(f"BLOCKED: forbidden import from '{node.module}'")
                    # `from os import *` / `from shutil import *` — a wildcard from
                    # a module that hosts dangerous calls hides them from the
                    # (module, attr) blocklist.
                    if mod in _danger_modules and any(a.name == "*" for a in node.names):
                        warnings.append(f"BLOCKED: wildcard import from '{node.module}'")
                elif isinstance(node, ast.Attribute):
                    # Reflection-escape dunder access (audit L6).
                    if node.attr in DANGEROUS_DUNDERS:
                        warnings.append(f"BLOCKED: reflection dunder '{node.attr}'")
                elif isinstance(node, ast.Name):
                    if node.id in DANGEROUS_DUNDERS or node.id == "__builtins__":
                        warnings.append(f"BLOCKED: reflection name '{node.id}'")
                elif isinstance(node, ast.Call):
                    func = node.func
                    # Bare call: eval(...), exec(...), __import__(...)
                    if isinstance(func, ast.Name) and func.id in DANGEROUS_CALL_NAMES:
                        warnings.append(f"BLOCKED: dangerous call '{func.id}(...)'")
                    # Bare call to a `from os import kill`-style dangerous alias.
                    elif isinstance(func, ast.Name) and func.id in dangerous_from_aliases:
                        warnings.append(
                            f"BLOCKED: dangerous call '{dangerous_from_aliases[func.id]}(...)'")
                    # open(..., 'w') and friends — arbitrary file write (audit A1).
                    elif isinstance(func, ast.Name) and func.id == "open":
                        reason = _open_write_mode(node)
                        if reason:
                            warnings.append(f"BLOCKED: {reason}")
                    # Attribute call: os.system(...), subprocess.run(...),
                    # resolving any import alias to the real module first.
                    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                        module = alias_to_module.get(func.value.id, func.value.id)
                        pair = (module, func.attr)
                        if pair in DANGEROUS_ATTR_CALLS:
                            warnings.append(f"BLOCKED: dangerous call '{pair[0]}.{pair[1]}(...)'")
                        # io.open(..., 'w') / os.open(...) — write via a module
                        # aliasing the builtin open (audit: open() bypass).
                        elif func.attr == "open" and module in ("io", "os"):
                            reason = _open_write_mode(node)
                            if reason or module == "os":
                                warnings.append(
                                    f"BLOCKED: {reason or f'{module}.open() (file write)'}")
        except SyntaxError:
            pass  # already caught by validate_syntax

        blocked = any(w.startswith("FORBIDDEN") or w.startswith("BLOCKED") for w in warnings)
        return not blocked, warnings

    def validate_ethics_preserved(self, code: str, relative_path: str) -> bool:
        """Ensure critical ethics structures are not removed."""
        # Only applies when modifying files that reference ethics
        if "ethics" not in relative_path.lower() and "axiom" not in code.lower():
            return True
        # Check that axiom references are preserved
        required = ["E-001", "E-002", "E-003", "E-004"]
        for req in required:
            if req not in code:
                logger.warning(f"Ethics preservation check failed: {req} missing")
                return False
        return True

    # ── Apply ────────────────────────────────────────────────────────

    def apply_modification(self, relative_path: str, new_code: str,
                           description: str, author: str = "aegis") -> dict:
        """Apply a code modification with backup, validation, and rollback on failure."""
        self.total_mods += 1

        record = {
            "id": f"cmod_{self.total_mods:04d}",
            "timestamp": CLOCK.now(),
            "file": relative_path,
            "description": description,
            "author": author,
            "status": "pending",
        }

        # 0. Path containment — reject anything outside the package directory.
        try:
            path, relative_path = self._resolve_path(relative_path)
            record["file"] = relative_path
        except ValueError as e:
            record["status"] = "path_blocked"
            record["error"] = str(e)
            self.blocked_mods += 1
            self.modifications.append(record)
            self._save_stats()
            logger.warning(f"Code mod blocked (path): {e}")
            return record

        # 1. Syntax validation
        valid, msg = self.validate_syntax(new_code)
        if not valid:
            record["status"] = "syntax_error"
            record["error"] = msg
            self.failed_mods += 1
            self.modifications.append(record)
            self._save_stats()
            logger.warning(f"Code mod rejected (syntax): {msg}")
            return record

        # 2. Safety validation
        safe, warnings = self.validate_safety(new_code, relative_path)
        record["warnings"] = warnings
        if not safe:
            record["status"] = "safety_blocked"
            self.blocked_mods += 1
            self.modifications.append(record)
            self._save_stats()
            logger.warning(f"Code mod blocked (safety): {warnings}")
            return record

        # 3. Ethics preservation
        if not self.validate_ethics_preserved(new_code, relative_path):
            record["status"] = "ethics_blocked"
            record["error"] = "Ethics axioms would be compromised"
            self.blocked_mods += 1
            self.modifications.append(record)
            self._save_stats()
            logger.warning("Code mod blocked: ethics preservation failed")
            return record

        # 4. Backup original. ``None`` means the file did not exist (new file);
        # an empty string means it existed but was empty — these must NOT be
        # conflated, or rollback would delete a legitimately-empty file instead
        # of restoring it.
        original = None
        if path.exists():
            original = path.read_text(encoding="utf-8")
            # time_ns + counter: two mods to the same file in the same second
            # must not overwrite each other's backup.
            backup_name = f"{relative_path.replace('/', '_').replace('.py', '')}_{time.time_ns()}_{self.total_mods}.py"
            backup_path = self.backups_dir / backup_name
            backup_path.write_text(original, encoding="utf-8")
            record["backup"] = str(backup_path)
        else:
            record["is_new_file"] = True

        # 5. Write new code
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_code, encoding="utf-8")
        except Exception as e:
            record["status"] = "write_failed"
            record["error"] = str(e)
            self.failed_mods += 1
            self.modifications.append(record)
            self._save_stats()
            return record

        # 6. Compile-check WITHOUT importing into the live process.
        #
        # We deliberately do NOT importlib.reload here: reloading a module whose
        # class instances are already running (the live Substrate and friends)
        # does not update those objects, gives false confidence, and can corrupt
        # state mid-tick. py_compile validates that the new source compiles
        # (a superset of ast.parse) and writes the .pyc; the change then takes
        # effect cleanly on the next process restart. Backups + rollback remain.
        try:
            py_compile.compile(str(path), doraise=True)

            record["status"] = "applied_pending_restart"
            record["note"] = "Compiled OK; takes effect on next restart (no hot reload)."
            record["lines_before"] = original.count("\n") + 1 if original else 0  # None/"" -> 0
            record["lines_after"] = new_code.count("\n") + 1
            self.successful_mods += 1

            self.rollback_stack.append({
                "file": relative_path,
                "original": original,
                "timestamp": CLOCK.now(),
                "mod_id": record["id"],
            })
            # Keep rollback stack manageable
            if len(self.rollback_stack) > 20:
                self.rollback_stack = self.rollback_stack[-20:]

            logger.info(f"Code mod applied (pending restart): {relative_path} — {description}")

        except Exception as e:
            # Rollback — restore original (None means it was a new file → remove).
            if original is not None:
                path.write_text(original, encoding="utf-8")
            elif path.exists():
                path.unlink()

            record["status"] = "compile_failed_rolled_back"
            record["error"] = str(e)
            self.failed_mods += 1
            logger.warning(f"Code mod rolled back (compile failed): {e}")

        self.modifications.append(record)
        self._save_stats()
        return record

    # ── Rollback ─────────────────────────────────────────────────────

    def rollback_last(self) -> dict:
        """Rollback the most recent successful modification."""
        if not self.rollback_stack:
            return {"success": False, "error": "Nothing to rollback"}

        entry = self.rollback_stack.pop()
        path = self.base_dir / entry["file"]

        try:
            if entry["original"] is not None:
                path.write_text(entry["original"], encoding="utf-8")
            elif path.exists():
                path.unlink()

            self.modifications.append({
                "id": f"cmod_rollback_{int(CLOCK.now())}",
                "timestamp": CLOCK.now(),
                "file": entry["file"],
                "description": f"Rollback of {entry['mod_id']}",
                "status": "rolled_back",
            })
            self._save_stats()
            logger.info(f"Code mod rolled back: {entry['file']}")
            return {"success": True, "file": entry["file"], "mod_id": entry["mod_id"]}
        except Exception as e:
            # Put the entry back so the original content is not lost and the
            # rollback can be retried.
            self.rollback_stack.append(entry)
            return {"success": False, "error": str(e), "file": entry["file"]}

    def rollback_to(self, mod_id: str) -> dict:
        """Roll back modifications from newest down to and including ``mod_id``.

        Returns success=False (without touching anything more) if ``mod_id`` is
        not present in the current rollback stack."""
        if not any(e.get("mod_id") == mod_id for e in self.rollback_stack):
            return {"success": False, "error": f"mod_id {mod_id} not in rollback stack", "rolled_back": []}
        rolled = []
        while self.rollback_stack:
            target_reached = self.rollback_stack[-1].get("mod_id") == mod_id
            result = self.rollback_last()
            if not result.get("success"):
                # Stop and report the partial rollback — the failed entry is
                # back on the stack, so nothing was lost.
                return {"success": False, "error": result.get("error"),
                        "failed_file": result.get("file"), "rolled_back": rolled}
            rolled.append(result["file"])
            if target_reached:
                break
        return {"success": True, "rolled_back": rolled}

    # ── Analysis ─────────────────────────────────────────────────────

    def analyze_file(self, relative_path: str) -> dict:
        """Analyze a source file — extract classes, functions, imports, complexity."""
        code = self.read_source(relative_path)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}

        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({"name": node.name, "line": node.lineno, "methods": methods})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Top-level functions only (not methods)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append({"name": node.name, "line": node.lineno})
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "file": relative_path,
            "lines": code.count("\n") + 1,
            "chars": len(code),
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "immutable": self._resolve_path(relative_path)[1] in IMMUTABLE_FILES,
        }

    # ── Status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "total_modifications": self.total_mods,
            "successful": self.successful_mods,
            "failed": self.failed_mods,
            "blocked": self.blocked_mods,
            "success_rate": round(self.successful_mods / max(1, self.total_mods) * 100, 1),
            "rollback_depth": len(self.rollback_stack),
            "recent_modifications": self.modifications[-10:],
            "immutable_files": list(IMMUTABLE_FILES),
            "source_files": len(self.list_sources()),
        }
