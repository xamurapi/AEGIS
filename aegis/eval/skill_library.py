"""Point 3 — a library of learned SKILLS (replaces self-training on own output).

A skill is a pure Python function ``solve(payload) -> answer`` stored as source,
declaring which task kinds it can attempt. Skills are added only after passing
the sandbox safety check; capability compounds because a useful skill is reused
forever (unlike fine-tuning a model on its own text, which drifts/collapses).
"""
import hashlib
import logging
import threading
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

from aegis.clock import CLOCK
from aegis.eval.sandbox import check_safe
from aegis.store.migrations import read_store, write_store

logger = logging.getLogger("aegis.skills")


@dataclass
class Skill:
    name: str
    kinds: list[str]
    code: str
    func: str = "solve"
    attempts: int = 0
    successes: int = 0
    # Injectable clock (spec §3.6) — this was the last direct wall-clock read
    # left in the package, and it made a skill's birth time untestable.
    created: float = field(default_factory=CLOCK.now)
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
    """Thread-safe skill store.

    The capability layer mutates the library from the event-loop thread
    (add/remove during skill synthesis) while the solver reads it from executor
    threads (for_kind during env steps / benchmarks). Without synchronization
    that races: "dictionary changed size during iteration", and concurrent
    save() calls corrupt skills.json (audit H1). A single reentrant lock guards
    every access to self.skills; readers return snapshots so callers can iterate
    outside the lock safely.
    """

    def __init__(self, store_path: Path | None = None, seed: bool = True):
        self.skills: dict[str, Skill] = {}
        self._store_path = store_path
        self._lock = threading.RLock()
        # Records the load could not turn into Skill objects. They are carried
        # verbatim and written back on every save: a record this build cannot
        # parse may still be a learned skill a newer (or repaired) build can,
        # and destroying it because one field looked unfamiliar is exactly the
        # failure mode this library exists to avoid.
        self._quarantine: list = []
        if seed:
            for s in _SEED_SKILLS:
                self.skills[s.name] = Skill(**asdict(s))
        self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self):
        """Load through the versioned-store layer, one record at a time.

        The old path was ``Skill(**d)`` over raw JSON inside one broad except:
        a single unknown per-skill field raised TypeError, the whole load was
        abandoned (library reset to the 10 seeds), and the next save()
        rewrote the file — permanently destroying every learned skill (audit:
        skill-store data loss). Now each record is repaired (unknown fields
        dropped) or quarantined individually, and a file that cannot be read
        at all is preserved on disk before anything may overwrite it.
        """
        if not self._store_path or not self._store_path.exists():
            return
        data = read_store(self._store_path, store="skills")
        rows = data.get("skills")
        if not isinstance(rows, list):
            # The file exists but yielded nothing usable (corrupt JSON, wrong
            # shape, a future schema). Continuing with seeds is survivable;
            # letting the next save() overwrite the only copy of the operator's
            # data is not — park the original bytes beside the store first.
            self._preserve_unreadable()
            return
        known = {f.name for f in fields(Skill)}
        with self._lock:
            for row in rows:
                if not isinstance(row, dict):
                    self._quarantine.append(row)
                    continue
                # Repair: keep only the fields this build's Skill declares.
                # `success_rate` (a derived value to_dict adds) and any field a
                # newer build wrote are dropped from the OBJECT, not the file —
                # unknown-but-parseable records still load.
                repaired = {k: v for k, v in row.items() if k in known}
                try:
                    skill = Skill(**repaired)
                except Exception:
                    logger.warning("Quarantining unreadable skill record %r",
                                   row.get("name", "<unnamed>"), exc_info=True)
                    self._quarantine.append(row)
                    continue
                self.skills[skill.name] = skill

    def _preserve_unreadable(self):
        """Copy a store that failed to load to ``<name>.corrupt`` — once.

        The first preserved copy wins: if the store is corrupt across several
        restarts, the earliest snapshot is the one closest to the good data.
        """
        backup = self._store_path.with_name(self._store_path.name + ".corrupt")
        if backup.exists():
            return
        try:
            backup.write_bytes(self._store_path.read_bytes())
            logger.warning("Skill store %s was unreadable — preserved a copy at %s",
                           self._store_path, backup)
        except Exception:
            logger.warning("Could not preserve unreadable skill store %s",
                           self._store_path, exc_info=True)

    def save(self):
        if not self._store_path:
            return
        try:
            with self._lock:
                payload = {"skills": [s.to_dict() for s in self.skills.values()]
                           + list(self._quarantine)}
            write_store(self._store_path, payload)
        except Exception:
            logger.warning("Failed to save skill library", exc_info=True)

    # ── dispatch ─────────────────────────────────────────────────
    def for_kind(self, kind: str) -> list[Skill]:
        # Snapshot under the lock so a concurrent add/remove cannot mutate the
        # dict mid-iteration.
        with self._lock:
            return [s for s in self.skills.values() if kind in s.kinds]

    def snapshot(self) -> list[dict]:
        """Behaviour-defining view of the library, taken under the lock.

        Name, claimed kinds, a hash of the code and the success counters — that
        is everything about the library that changes what the solver does. The
        code itself is hashed rather than copied so the snapshot stays small
        enough to embed in a state digest or ship to an evaluation worker.
        """
        with self._lock:
            return sorted(
                ({"name": s.name,
                  "kinds": sorted(s.kinds),
                  "func": s.func,
                  "code_hash": hashlib.blake2b(
                      s.code.encode("utf-8"), digest_size=8).hexdigest(),
                  "attempts": s.attempts,
                  "successes": s.successes}
                 for s in self.skills.values()),
                key=lambda row: row["name"],
            )

    def add(self, skill: Skill) -> tuple[bool, str]:
        safe, reasons = check_safe(skill.code)
        if not safe:
            return False, f"rejected (unsafe): {reasons}"
        with self._lock:
            self.skills[skill.name] = skill
        self.save()
        return True, "added"

    def remove(self, name: str):
        with self._lock:
            self.skills.pop(name, None)
        self.save()

    def record(self, name: str, success: bool):
        with self._lock:
            s = self.skills.get(name)
            if s:
                s.attempts += 1
                if success:
                    s.successes += 1

    def status(self) -> dict:
        with self._lock:
            return {
                "total_skills": len(self.skills),
                "kinds_covered": sorted({k for s in self.skills.values() for k in s.kinds}),
                "skills": [
                    {"name": s.name, "kinds": s.kinds, "origin": s.origin,
                     "attempts": s.attempts, "success_rate": round(s.success_rate(), 3)}
                    for s in self.skills.values()
                ],
            }
