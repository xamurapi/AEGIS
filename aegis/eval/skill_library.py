"""Point 3 — a library of learned SKILLS (replaces self-training on own output).

A skill is a pure Python function ``solve(payload) -> answer`` stored as source,
declaring which task kinds it can attempt. Skills are added only after passing
the sandbox safety check; capability compounds because a useful skill is reused
forever (unlike fine-tuning a model on its own text, which drifts/collapses).
"""
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

from aegis.eval.sandbox import check_safe

logger = logging.getLogger("aegis.skills")


@dataclass
class Skill:
    name: str
    kinds: list[str]
    code: str
    func: str = "solve"
    attempts: int = 0
    successes: int = 0
    created: float = field(default_factory=time.time)
    origin: str = "seed"  # "seed" | "llm" | "synthesized"

    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success_rate"] = round(self.success_rate(), 3)
        return d


# Seeded skills cover four kinds; is_prime and sort_csv are left unsolved on
# purpose so the synthesis loop has a measurable gap to close.
_SEED_SKILLS = [
    Skill("calc_basic", ["calc"], origin="seed", code=(
        "def solve(p):\n"
        "    a, b, op = p['a'], p['b'], p['op']\n"
        "    if op == 'add': return a + b\n"
        "    if op == 'sub': return a - b\n"
        "    if op == 'mul': return a * b\n"
        "    return None\n"
    )),
    Skill("string_reverse", ["reverse"], origin="seed", code=(
        "def solve(p):\n    return p['s'][::-1]\n"
    )),
    Skill("vowel_counter", ["count_vowels"], origin="seed", code=(
        "def solve(p):\n    return sum(1 for ch in p['s'].lower() if ch in 'aeiou')\n"
    )),
    Skill("fibonacci", ["fib"], origin="seed", code=(
        "def solve(p):\n"
        "    n = p['n']; a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a + b\n"
        "    return a\n"
    )),
    Skill("palindrome_check", ["palindrome"], origin="seed", code=(
        "def solve(p):\n    s = p['s']\n    return s == s[::-1]\n"
    )),
    Skill("euclid_gcd", ["gcd"], origin="seed", code=(
        "def solve(p):\n"
        "    a, b = p['a'], p['b']\n"
        "    while b: a, b = b, a % b\n"
        "    return a\n"
    )),
    Skill("factorial", ["factorial"], origin="seed", code=(
        "def solve(p):\n"
        "    n = p['n']; r = 1\n"
        "    for i in range(2, n + 1): r *= i\n"
        "    return r\n"
    )),
    Skill("word_counter", ["word_count"], origin="seed", code=(
        "def solve(p):\n    return len(p['s'].split())\n"
    )),
    Skill("digit_sum", ["sum_digits"], origin="seed", code=(
        "def solve(p):\n    return sum(int(c) for c in str(p['n']))\n"
    )),
    Skill("uppercase", ["upper"], origin="seed", code=(
        "def solve(p):\n    return p['s'].upper()\n"
    )),
]


class SkillLibrary:
    def __init__(self, store_path: Path | None = None, seed: bool = True):
        self.skills: dict[str, Skill] = {}
        self._store_path = store_path
        if seed:
            for s in _SEED_SKILLS:
                self.skills[s.name] = Skill(**asdict(s))
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self):
        if not self._store_path or not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            for d in data.get("skills", []):
                d.pop("success_rate", None)
                self.skills[d["name"]] = Skill(**d)
        except Exception:
            logger.warning("Failed to load skill library from %s", self._store_path, exc_info=True)

    def save(self):
        if not self._store_path:
            return
        try:
            self._store_path.write_text(
                json.dumps({"skills": [s.to_dict() for s in self.skills.values()]}, indent=1),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to save skill library", exc_info=True)

    # ── dispatch ─────────────────────────────────────────────────
    def for_kind(self, kind: str) -> list[Skill]:
        return [s for s in self.skills.values() if kind in s.kinds]

    def add(self, skill: Skill) -> tuple[bool, str]:
        safe, reasons = check_safe(skill.code)
        if not safe:
            return False, f"rejected (unsafe): {reasons}"
        self.skills[skill.name] = skill
        self.save()
        return True, "added"

    def remove(self, name: str):
        self.skills.pop(name, None)
        self.save()

    def record(self, name: str, success: bool):
        s = self.skills.get(name)
        if s:
            s.attempts += 1
            if success:
                s.successes += 1

    def status(self) -> dict:
        return {
            "total_skills": len(self.skills),
            "kinds_covered": sorted({k for s in self.skills.values() for k in s.kinds}),
            "skills": [
                {"name": s.name, "kinds": s.kinds, "origin": s.origin,
                 "attempts": s.attempts, "success_rate": round(s.success_rate(), 3)}
                for s in self.skills.values()
            ],
        }
