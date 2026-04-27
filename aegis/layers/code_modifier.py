"""Code Self-Modification — read, analyze, modify own source code with safety and rollback."""
import ast
import sys
import time
import json
import importlib
import logging
from pathlib import Path

logger = logging.getLogger("aegis.code_modifier")

# Patterns that must NEVER appear in modified code
FORBIDDEN_PATTERNS = [
    "os.system(",
    "subprocess.run(",
    "subprocess.Popen(",
    "subprocess.call(",
    "__import__('os')",
    '__import__("os")',
    "shutil.rmtree(",
    "eval(",
    "exec(",
    "open('/etc",
    "open('C:\\\\Windows",
    "rmdir(",
    "unlink(",
    # Ethics protection — these strings must not be removed from ethics_core
    # (checked separately in validate_ethics_preserved)
]

# Files that are IMMUTABLE — cannot be modified by the system
IMMUTABLE_FILES = {
    "layers/ethics_core.py",   # ethical axioms must never be changed by AI
    "config.py",               # config changes need human approval
}

# Max diff size (chars) per single modification
MAX_MODIFICATION_SIZE = 5000


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
                pass

    def _save_stats(self):
        data = {
            "total_mods": self.total_mods,
            "successful_mods": self.successful_mods,
            "failed_mods": self.failed_mods,
            "blocked_mods": self.blocked_mods,
            "modifications": self.modifications[-50:],
            "last_updated": time.time(),
        }
        try:
            self._stats_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except Exception:
            pass

    # ── Read ─────────────────────────────────────────────────────────

    def read_source(self, relative_path: str) -> str:
        """Read a source file from the aegis package."""
        path = self.base_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        if not str(path).endswith(".py"):
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

        # Immutable file check
        norm = relative_path.replace("\\", "/")
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

        # AST analysis — check for dangerous node types
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("subprocess", "ctypes", "signal"):
                            warnings.append(f"BLOCKED: forbidden import '{alias.name}'")
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in ("subprocess", "ctypes", "signal"):
                        warnings.append(f"BLOCKED: forbidden import from '{node.module}'")
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
        path = self.base_dir / relative_path

        record = {
            "id": f"cmod_{self.total_mods:04d}",
            "timestamp": time.time(),
            "file": relative_path,
            "description": description,
            "author": author,
            "status": "pending",
        }

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

        # 4. Backup original
        original = ""
        if path.exists():
            original = path.read_text(encoding="utf-8")
            backup_name = f"{relative_path.replace('/', '_').replace('.py', '')}_{int(time.time())}.py"
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

        # 6. Test import — verify the module loads without errors
        module_name = "aegis." + relative_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)

            record["status"] = "applied"
            record["lines_before"] = original.count("\n") + 1 if original else 0
            record["lines_after"] = new_code.count("\n") + 1
            self.successful_mods += 1

            self.rollback_stack.append({
                "file": relative_path,
                "original": original,
                "timestamp": time.time(),
                "mod_id": record["id"],
            })
            # Keep rollback stack manageable
            if len(self.rollback_stack) > 20:
                self.rollback_stack = self.rollback_stack[-20:]

            logger.info(f"Code mod applied: {relative_path} — {description}")

        except Exception as e:
            # Rollback — restore original
            if original:
                path.write_text(original, encoding="utf-8")
            elif path.exists():
                path.unlink()

            record["status"] = "import_failed_rolled_back"
            record["error"] = str(e)
            self.failed_mods += 1
            logger.warning(f"Code mod rolled back (import failed): {e}")

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
            if entry["original"]:
                path.write_text(entry["original"], encoding="utf-8")
            elif path.exists():
                path.unlink()

            self.modifications.append({
                "id": f"cmod_rollback_{int(time.time())}",
                "timestamp": time.time(),
                "file": entry["file"],
                "description": f"Rollback of {entry['mod_id']}",
                "status": "rolled_back",
            })
            self._save_stats()
            logger.info(f"Code mod rolled back: {entry['file']}")
            return {"success": True, "file": entry["file"], "mod_id": entry["mod_id"]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rollback_to(self, mod_id: str) -> dict:
        """Rollback all modifications up to and including the given mod_id."""
        rolled = []
        while self.rollback_stack:
            entry = self.rollback_stack[-1]
            result = self.rollback_last()
            if result.get("success"):
                rolled.append(result["file"])
            if entry.get("mod_id") == mod_id:
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
            "immutable": relative_path.replace("\\", "/") in IMMUTABLE_FILES,
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
