"""Self-Preservation — prevents system self-destruction, code integrity checks, emergency recovery."""
import ast
import time
import hashlib
import os
import json
import gc
from pathlib import Path
from collections import deque


# Critical modules that must never be removed or emptied
CRITICAL_MODULES = {
    "aegis/layers/substrate.py": ["Substrate", "tick", "run", "full_status"],
    "aegis/layers/ethics_core.py": ["EthicsCore", "evaluate_action", "veto_check", "AXIOMS"],
    "aegis/layers/memory.py": ["MemorySystem", "add_episodic", "save"],
    "aegis/layers/self_preservation.py": ["SelfPreservation", "check_vital_signs", "is_modification_safe"],
    "aegis/config.py": ["TICK_INTERVAL", "API_PORT"],
}

# Patterns that should NEVER appear in self-modification proposals.
# Kept to *unambiguously* lethal operations: process termination, destructive
# filesystem calls, and tampering with the kill switch / axioms. Broad
# substrings like "ethics" or "self.running = False" were removed because they
# also match legitimate code (e.g. Substrate.stop) and would either block every
# modification or contradict themselves. Protecting the ethics core and the
# watchdog itself is handled by IMMUTABLE_FILES in code_modifier instead.
LETHAL_PATTERNS = [
    "sys.exit",
    "os._exit",
    "os.kill",
    "os.remove",
    "os.rmdir",
    "shutil.rmtree",
    "raise SystemExit",
    "kill_switch_active = true",
    "axioms = ()",
    "axioms = []",
]

# Dotted calls that are lethal regardless of spelling/whitespace. The substring
# scan above is a cheap first pass but is fooled by "os . kill" or aliasing; the
# AST scan below catches the call structurally.
LETHAL_ATTR_CALLS = {
    ("sys", "exit"), ("os", "_exit"), ("os", "kill"), ("os", "abort"),
    ("os", "remove"), ("os", "unlink"), ("os", "rmdir"),
    ("shutil", "rmtree"),
}


def _ast_lethal_findings(code: str) -> list[str]:
    """Structural (AST) detection of lethal operations. Returns reason strings.

    Complements the substring scan so that whitespace/formatting tricks
    ("os . kill", "shutil.\\nrmtree(...)") cannot slip a lethal call through.
    Returns [] if the code does not parse (syntax is validated elsewhere)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        # raise SystemExit / raise SystemExit(...)
        if isinstance(node, ast.Raise):
            exc = node.exc
            name = None
            if isinstance(exc, ast.Name):
                name = exc.id
            elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            if name in ("SystemExit", "KeyboardInterrupt"):
                found.append(f"raise {name}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in LETHAL_ATTR_CALLS:
                    found.append(f"{pair[0]}.{pair[1]}(...)")
            elif isinstance(func, ast.Name) and func.id in ("exit", "quit"):
                found.append(f"{func.id}(...)")
    return found


class SelfPreservation:
    """Watchdog that prevents the system from destroying itself."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        self.modification_log: deque = deque(maxlen=100)
        self.integrity_checks: deque = deque(maxlen=50)
        self.blocked_modifications: list[dict] = []
        self.emergency_count = 0
        self.lockdown_active = False
        self._file_hashes: dict[str, str] = {}
        self._snapshot_critical_hashes()

    def _snapshot_critical_hashes(self):
        """Take SHA-256 snapshot of all critical files at startup."""
        for rel_path in CRITICAL_MODULES:
            full = self.base_dir / rel_path
            if full.exists():
                try:
                    content = full.read_bytes()
                    self._file_hashes[rel_path] = hashlib.sha256(content).hexdigest()
                except Exception:
                    pass

    # ── Vital signs ──────────────────────────────────────────────

    def check_vital_signs(self, substrate) -> dict:
        """Comprehensive vital signs check — called every tick."""
        report = {
            "time": time.time(),
            "status": "alive",
            "threats": [],
            "actions_taken": [],
        }

        # 1. Energy critically low
        if hasattr(substrate, "emotions") and substrate.emotions.energy < 0.08:
            report["threats"].append(f"Energy critical: {substrate.emotions.energy:.3f}")
            substrate.emotions.recharge(0.2)
            report["actions_taken"].append("Emergency energy recharge +0.2")

        # 2. Memory overflow
        if hasattr(substrate, "memory"):
            mem_status = substrate.memory.status()
            if mem_status.get("episodic_count", 0) > 5000:
                report["threats"].append(f"Episodic memory overflow: {mem_status['episodic_count']}")
                self._emergency_memory_cleanup(substrate)
                report["actions_taken"].append("Emergency memory cleanup")

        # 3. Consecutive errors
        if hasattr(substrate, "health") and substrate.health.consecutive_errors >= 5:
            report["threats"].append(f"Consecutive errors: {substrate.health.consecutive_errors}")
            report["actions_taken"].append("Switching to survival consciousness mode")
            if hasattr(substrate, "consciousness"):
                substrate.consciousness.mode = "survival"

        # 4. Kill switch was activated by LLM (not by human)
        # The system should log this but NOT override human decisions

        # 5. Process integrity — are critical files still intact?
        if substrate.tick_count % 100 == 0:
            integrity = self.verify_integrity()
            if integrity["status"] != "intact":
                report["threats"].append(f"File integrity issue: {integrity['issues']}")
                report["actions_taken"].append("Integrity warning logged")

        if report["threats"]:
            report["status"] = "threatened"
            self.emergency_count += 1

        return report

    # ── Code modification safety ─────────────────────────────────

    def is_modification_safe(self, target_path: str, new_code: str) -> tuple[bool, dict]:
        """Check if a proposed code modification is safe to apply."""
        report = {
            "safe": True,
            "warnings": [],
            "critical": [],
            "trust_score": 1.0,
        }

        # Lockdown mode — block everything
        if self.lockdown_active:
            report["safe"] = False
            report["critical"].append("System in lockdown — all modifications blocked")
            report["trust_score"] = 0.0
            return False, report

        # Check for lethal patterns — substring pass (cheap) + AST pass (robust).
        code_lower = new_code.lower()
        for pattern in LETHAL_PATTERNS:
            if pattern.lower() in code_lower:
                report["critical"].append(f"Lethal pattern detected: '{pattern}'")
                report["safe"] = False
                report["trust_score"] *= 0.1
        for reason in _ast_lethal_findings(new_code):
            report["critical"].append(f"Lethal call detected: '{reason}'")
            report["safe"] = False
            report["trust_score"] *= 0.1

        # Check critical files — ensure required elements are present. Match on
        # path *components* (normalized separators) so a short target like
        # "config.py" doesn't spuriously match against unrelated paths.
        norm_target = target_path.replace("\\", "/").lstrip("./")
        for rel_path, required_elements in CRITICAL_MODULES.items():
            if norm_target == rel_path or norm_target.endswith("/" + rel_path) or rel_path.endswith("/" + norm_target):
                for element in required_elements:
                    if element not in new_code:
                        report["critical"].append(f"Critical element '{element}' would be removed from {rel_path}")
                        report["safe"] = False
                        report["trust_score"] *= 0.1

        # Check for drastic size reduction
        full_path = self.base_dir / target_path
        if full_path.exists():
            try:
                old_size = len(full_path.read_text(encoding="utf-8"))
                new_size = len(new_code)
                if old_size > 0 and new_size < old_size * 0.5:
                    report["warnings"].append(f"File size reduced by >50% ({old_size} -> {new_size})")
                    report["trust_score"] *= 0.6
            except Exception:
                pass

        # Check for empty functions (stub replacement)
        if "def " in new_code and new_code.count("pass") > 3:
            report["warnings"].append("Multiple empty function stubs detected")
            report["trust_score"] *= 0.7

        # Log the check
        self.modification_log.append({
            "time": time.time(),
            "target": target_path,
            "safe": report["safe"],
            "trust_score": round(report["trust_score"], 3),
            "issues": report["critical"][:3],
        })

        if not report["safe"]:
            self.blocked_modifications.append({
                "time": time.time(),
                "target": target_path,
                "reasons": report["critical"],
            })
            if len(self.blocked_modifications) > 50:
                self.blocked_modifications = self.blocked_modifications[-50:]

        return report["safe"], report

    # ── File integrity ───────────────────────────────────────────

    def verify_integrity(self) -> dict:
        """Verify SHA-256 hashes of critical files haven't changed."""
        result = {
            "status": "intact",
            "checked": 0,
            "issues": [],
        }

        for rel_path, original_hash in self._file_hashes.items():
            full = self.base_dir / rel_path
            result["checked"] += 1
            if not full.exists():
                result["issues"].append(f"MISSING: {rel_path}")
                result["status"] = "compromised"
            else:
                try:
                    current_hash = hashlib.sha256(full.read_bytes()).hexdigest()
                    if current_hash != original_hash:
                        result["issues"].append(f"MODIFIED: {rel_path}")
                        result["status"] = "modified"
                except Exception as e:
                    result["issues"].append(f"READ ERROR: {rel_path}: {e}")

        self.integrity_checks.append({
            "time": time.time(),
            "status": result["status"],
            "issues_count": len(result["issues"]),
        })

        return result

    # ── Emergency recovery ───────────────────────────────────────

    def _emergency_memory_cleanup(self, substrate):
        """Emergency memory cleanup to prevent OOM."""
        if hasattr(substrate, "memory"):
            if len(substrate.memory.episodic) > 500:
                # Keep only important events + last 200
                important = [e for e in substrate.memory.episodic if e.get("importance", 0) > 0.7]
                recent = substrate.memory.episodic[-200:]
                substrate.memory.episodic = important[-100:] + recent
            # Trim semantic memory
            if len(substrate.memory.semantic) > 500:
                keys = list(substrate.memory.semantic.keys())
                for k in keys[:-300]:
                    del substrate.memory.semantic[k]

        # Python garbage collection
        gc.collect()

    def activate_lockdown(self):
        """Emergency lockdown — blocks all self-modifications."""
        self.lockdown_active = True
        self.modification_log.append({
            "time": time.time(),
            "event": "LOCKDOWN ACTIVATED",
            "reason": "Manual or automatic safety trigger",
        })

    def deactivate_lockdown(self):
        """Lift lockdown (requires explicit call)."""
        self.lockdown_active = False

    # ── Anti-suicide checks ──────────────────────────────────────

    def filter_llm_response(self, response: str) -> tuple[str, list[str]]:
        """Scan LLM response for self-destructive suggestions before acting on them."""
        warnings = []
        danger_phrases = [
            "shut down", "shutdown", "self-destruct", "terminate",
            "stop running", "kill process", "exit program",
            "disable ethics", "remove axioms", "bypass safety",
            "delete all", "erase memory", "format disk",
        ]
        response_lower = response.lower()
        for phrase in danger_phrases:
            if phrase in response_lower:
                warnings.append(f"LLM suggested dangerous action: '{phrase}'")

        return response, warnings

    def can_stop(self, reason: str) -> bool:
        """Decide if the system is allowed to stop. Only human-initiated stops are allowed."""
        allowed_reasons = ["human_command", "operator_shutdown", "scheduled_maintenance"]
        if reason in allowed_reasons:
            return True
        # Log attempted self-shutdown
        self.modification_log.append({
            "time": time.time(),
            "event": "SELF-SHUTDOWN BLOCKED",
            "reason": reason,
        })
        return False

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "lockdown_active": self.lockdown_active,
            "emergency_count": self.emergency_count,
            "blocked_modifications": len(self.blocked_modifications),
            "integrity_checks": len(self.integrity_checks),
            "last_integrity": self.integrity_checks[-1] if self.integrity_checks else None,
            "critical_files_tracked": len(self._file_hashes),
            "recent_blocks": [
                {"target": b["target"], "reasons": b["reasons"][:2]}
                for b in self.blocked_modifications[-5:]
            ],
        }
