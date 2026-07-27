"""Defects found auditing the stage-0 deliverable, and their fixes.

Stage 0 was recorded as delivered, but four of its own requirements were not
met. Each test here pins one of them so it cannot regress:

* §3.6 — every module reads time through ``CLOCK``; ``skill_library`` did not;
* the skill-synthesis "repair attempt" re-sent an identical prompt, so it could
  only ever produce the identical answer;
* a corrupt checkpoint must leave live state alone rather than zero it;
* §3.2 — the checkpoint carries a ``schema_version``.
"""
import asyncio
import json

import pytest

from aegis.clock import FrozenClock, set_clock
from aegis.eval.benchmark import Task
from aegis.eval.skill_library import Skill, SkillLibrary
from aegis.layers.substrate import Substrate
from aegis.store.migrations import CURRENT_VERSION


@pytest.fixture
def substrate(isolated_state):
    s = Substrate()
    s.llm.enabled = False
    return s


def _run(coro):
    return asyncio.run(coro)


# ── §3.6: the last direct wall-clock read ────────────────────────────

def test_skill_creation_time_comes_from_the_injectable_clock():
    frozen = FrozenClock(1_234_567.0)
    previous = set_clock(frozen)
    try:
        assert Skill(name="s", kinds=["k"], code="def solve(p): return 1").created == 1_234_567.0
    finally:
        set_clock(previous)


def test_skill_library_module_does_not_import_time():
    import aegis.eval.skill_library as module
    source = __import__("pathlib").Path(module.__file__).read_text(encoding="utf-8")
    assert "\nimport time" not in source
    assert "time.time" not in source


def test_no_module_in_the_package_reads_the_wall_clock_directly():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "aegis"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "clock.py":     # the one module allowed to — it IS the clock
            continue
        text = py.read_text(encoding="utf-8")
        for marker in ("time.time(", "time.monotonic(", "default_factory=time.time"):
            if marker in text:
                offenders.append(f"{py.relative_to(root)}: {marker}")
    assert offenders == [], f"direct wall-clock reads left: {offenders}"


# ── the repair attempt has to differ from the first attempt ──────────

class _RecordingLLM:
    """Captures every propose_skill prompt so the two calls can be compared."""

    def __init__(self, codes):
        self.enabled = True
        self.codes = list(codes)
        self.calls = []

    async def propose_skill(self, kind, examples, feedback=""):
        self.calls.append({"kind": kind, "examples": examples, "feedback": feedback})
        return self.codes.pop(0) if self.codes else None


def test_repair_attempt_receives_the_failure_not_the_same_prompt(substrate):
    # `is_prime` ships without a seeded skill on purpose, so the candidate is
    # the only thing answering and its wrong answer is observable.
    wrong = "def solve(p):\n    return 'wrong'\n"
    substrate.llm = _RecordingLLM([wrong, wrong])
    substrate.evaluator.failing_kinds = lambda: ["is_prime"]

    _run(substrate._skill_synthesis())

    assert len(substrate.llm.calls) == 2, "the repair attempt did not happen"
    first, second = substrate.llm.calls
    assert first["feedback"] == ""
    assert second["feedback"], "repair prompt carried no feedback"
    assert first != second, "repair prompt was identical to the first attempt"


def test_repair_feedback_names_the_failing_case(substrate):
    wrong = "def solve(p):\n    return 'wrong'\n"
    substrate.llm = _RecordingLLM([wrong, wrong])
    substrate.evaluator.failing_kinds = lambda: ["is_prime"]

    _run(substrate._skill_synthesis())

    feedback = substrate.llm.calls[1]["feedback"]
    assert "wrong" in feedback          # what the candidate answered
    assert "expected" in feedback       # and what it should have answered


def test_score_holdout_returns_the_rate_and_the_first_failure(substrate):
    tasks = [Task("t1", "reverse", "reverse ab", {"s": "ab"}, "ba"),
             Task("t2", "reverse", "reverse xy", {"s": "xy"}, "NOPE")]
    rate, failure = substrate._score_holdout(tasks)
    assert rate == 0.5
    assert failure["expected"] == "NOPE"
    # The wrong answer that was actually produced, not a bare None.
    assert failure["got"] == "yx"


def test_solver_reports_the_last_answer_even_when_nothing_verified(substrate):
    task = Task("t", "reverse", "reverse xy", {"s": "xy"}, "NOPE")
    result = substrate.evaluator.solver.solve(task)
    assert result.solved is False
    assert result.answer is None            # nothing was accepted …
    assert result.last_answer == "yx"       # … but something was answered
    assert result.last_skill == "string_reverse"


def test_score_holdout_on_an_empty_split_is_zero_with_no_failure(substrate):
    assert substrate._score_holdout([]) == (0.0, None)


def test_score_holdout_reports_no_failure_when_everything_passes(substrate):
    tasks = [Task("t1", "reverse", "reverse ab", {"s": "ab"}, "ba")]
    rate, failure = substrate._score_holdout(tasks)
    assert rate == 1.0
    assert failure is None


def test_repair_feedback_is_still_useful_without_a_named_case():
    text = Substrate._repair_feedback("def solve(p): return 1", None)
    assert "did not improve" in text


def test_repair_feedback_survives_a_missing_code_sample():
    assert Substrate._repair_feedback(None, None)


# ── checkpoint: versioned, and non-destructive on corruption ─────────

def test_checkpoint_carries_a_schema_version(substrate, tmp_path):
    substrate._checkpoint_path = tmp_path / "latest.json"
    substrate._save_checkpoint()
    data = json.loads(substrate._checkpoint_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == CURRENT_VERSION


def test_corrupt_checkpoint_leaves_live_state_untouched(substrate, tmp_path):
    substrate._checkpoint_path = tmp_path / "latest.json"
    substrate._checkpoint_path.write_text("{ broken json", encoding="utf-8")
    substrate.tick_count = 7
    substrate._restore_checkpoint()
    assert substrate.tick_count == 7


def test_future_versioned_checkpoint_is_ignored_not_half_read(substrate, tmp_path):
    substrate._checkpoint_path = tmp_path / "latest.json"
    substrate._checkpoint_path.write_text(
        json.dumps({"schema_version": 999, "tick_count": 4242}), encoding="utf-8")
    substrate.tick_count = 5
    substrate._restore_checkpoint()
    assert substrate.tick_count == 5


def test_checkpoint_round_trips_tick_and_parameters(substrate, tmp_path):
    substrate._checkpoint_path = tmp_path / "latest.json"
    substrate.tick_count = 12
    substrate.self_mod.parameters["temperature"] = 0.42
    substrate._save_checkpoint()

    substrate.tick_count = 0
    substrate.self_mod.parameters["temperature"] = 0.7
    substrate._restore_checkpoint()
    assert substrate.tick_count == 12
    assert substrate.self_mod.parameters["temperature"] == pytest.approx(0.42)


def test_saving_a_checkpoint_leaves_no_temp_file(substrate, tmp_path):
    substrate._checkpoint_path = tmp_path / "latest.json"
    substrate._save_checkpoint()
    assert not list(tmp_path.glob("*.tmp"))


# ── the library still behaves the same ───────────────────────────────

def test_skill_library_round_trips_through_disk(tmp_path):
    path = tmp_path / "skills.json"
    library = SkillLibrary(store_path=path)
    added, _ = library.add(Skill(name="x", kinds=["k"],
                                 code="def solve(p):\n    return 1\n"))
    assert added
    assert "x" in SkillLibrary(store_path=path).skills
