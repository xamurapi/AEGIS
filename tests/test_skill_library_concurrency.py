"""Concurrency regression tests for SkillLibrary (audit H1).

Reproduces the "dictionary changed size during iteration" race: many threads
iterate (for_kind/status/save) while others add/remove. With the RLock these
run cleanly; without it they raise RuntimeError intermittently.
"""
import threading

from aegis.eval.skill_library import SkillLibrary, Skill


def _skill(name):
    return Skill(name=name, kinds=["calc"], code="def solve(p):\n    return 1\n", origin="llm")


def test_concurrent_add_remove_and_iterate_is_safe(tmp_path):
    lib = SkillLibrary(store_path=tmp_path / "skills.json", seed=True)
    errors = []
    stop = threading.Event()

    def writer(n):
        try:
            for i in range(200):
                lib.add(_skill(f"w{n}_{i}"))
                lib.remove(f"w{n}_{i}")
        except Exception as e:  # pragma: no cover - only on a real race
            errors.append(repr(e))

    def reader():
        try:
            while not stop.is_set():
                for _ in range(50):
                    lib.for_kind("calc")
                    lib.status()
        except Exception as e:  # pragma: no cover - only on a real race
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads[:4]:
        t.start()
    readers = threads[4:]
    for t in readers:
        t.start()
    for t in threads[:4]:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert errors == [], f"race detected: {errors[:3]}"
    # Seeds survive the churn; transient add/remove pairs cancel out.
    assert lib.for_kind("calc")  # calc_basic seed still present


def test_for_kind_returns_snapshot_list(tmp_path):
    lib = SkillLibrary(store_path=tmp_path / "s.json", seed=True)
    result = lib.for_kind("calc")
    # Mutating the returned list must not affect the library.
    result.clear()
    assert lib.for_kind("calc")
