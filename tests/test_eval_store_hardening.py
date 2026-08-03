"""Persistence hardening for the eval layer (audit: skill-store data loss).

The migration registry declared ("skills", 1) and ("eval_history", 1), but the
loaders read raw JSON — the registrations were decoration. Worse, the skill
loader did ``Skill(**d)`` inside one broad except: a single unknown per-skill
field made the WHOLE load fail, the library silently reset to the 10 seeds,
and the next save() rewrote the file — a learned skill was permanently
destroyed by one unfamiliar key.

The rules now under test:

* records load through ``read_store``/``write_store`` (versioned, atomic);
* a bad RECORD costs that record, never the file — and even then the record
  is quarantined and written back, not destroyed;
* a file that cannot be read at all is preserved on disk (``.corrupt``)
  before any save may overwrite it;
* an empty split reports None, not a fabricated 0.0.
"""
import json

import pytest

from aegis.eval.evaluator import Evaluator
from aegis.eval.skill_library import Skill, SkillLibrary
from aegis.eval.solver import MultiAgentSolver
from aegis.store.migrations import CURRENT_VERSION


def _skill_row(name="learned_x", extra=None):
    row = Skill(name=name, kinds=["calc"], origin="llm",
                code="def solve(p):\n    return p['a'] + p['b']\n").to_dict()
    if extra:
        row.update(extra)
    return row


# ── the skill library ────────────────────────────────────────────────

def test_an_unknown_per_skill_field_does_not_destroy_the_library(tmp_path):
    """The demonstrated data-loss case: one extra field used to abandon the
    whole load, and the next save wiped the learned skill from disk."""
    path = tmp_path / "skills.json"
    path.write_text(json.dumps(
        {"skills": [_skill_row(extra={"flavour": "new-build-field"})]}),
        encoding="utf-8")

    library = SkillLibrary(store_path=path)
    assert "learned_x" in library.skills            # loaded, not reset to seeds

    library.save()
    reloaded = SkillLibrary(store_path=path)
    assert "learned_x" in reloaded.skills           # and it survived the save


def test_one_bad_record_costs_that_record_not_the_file(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({"skills": [
        _skill_row(name="good_one"),
        {"name": "half_a_record"},                  # no kinds, no code
        _skill_row(name="good_two"),
    ]}), encoding="utf-8")

    library = SkillLibrary(store_path=path)
    assert "good_one" in library.skills
    assert "good_two" in library.skills
    assert "half_a_record" not in library.skills


def test_a_quarantined_record_is_written_back_not_destroyed(tmp_path):
    """A record this build cannot parse may still be a learned skill a newer
    build can — it rides along on every save instead of vanishing."""
    path = tmp_path / "skills.json"
    odd = {"name": "from_the_future", "payload": {"opaque": True}}
    path.write_text(json.dumps({"skills": [odd]}), encoding="utf-8")

    library = SkillLibrary(store_path=path)
    library.save()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert odd in on_disk["skills"]


def test_an_unreadable_file_is_preserved_before_any_save(tmp_path):
    original = "{ this is not json"
    path = tmp_path / "skills.json"
    path.write_text(original, encoding="utf-8")

    library = SkillLibrary(store_path=path)
    library.save()                                   # overwrites the store...

    backup = tmp_path / "skills.json.corrupt"
    assert backup.read_text(encoding="utf-8") == original   # ...but not the copy
    assert json.loads(path.read_text(encoding="utf-8"))["skills"]


def test_the_first_preserved_copy_wins(tmp_path):
    """Corrupt across several restarts: the earliest snapshot is the one
    closest to the good data and must not be clobbered by newer garbage."""
    path = tmp_path / "skills.json"
    path.write_text("{ garbage one", encoding="utf-8")
    SkillLibrary(store_path=path)
    path.write_text("{ garbage two", encoding="utf-8")
    SkillLibrary(store_path=path)
    backup = tmp_path / "skills.json.corrupt"
    assert backup.read_text(encoding="utf-8") == "{ garbage one"


def test_the_skill_store_is_versioned_now(tmp_path):
    """save() routes through write_store, so the registered ("skills", 1)
    migration finally has a loader that can fire it."""
    path = tmp_path / "skills.json"
    library = SkillLibrary(store_path=path)
    library.save()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == CURRENT_VERSION


def test_a_legacy_unversioned_skill_file_still_loads(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(json.dumps({"skills": [_skill_row(name="old_timer")]}),
                    encoding="utf-8")
    assert "old_timer" in SkillLibrary(store_path=path).skills


# ── the evaluator's history store ────────────────────────────────────

@pytest.fixture()
def solver():
    return MultiAgentSolver(SkillLibrary(store_path=None), timeout=5.0)


def test_eval_history_round_trips_versioned(tmp_path, solver):
    path = tmp_path / "eval_history.json"
    ev = Evaluator(solver, store_path=path)
    ev.last_score = 0.5
    ev.total_runs = 3
    ev.history.append({"t": 1.0, "score": 0.5})
    ev._save()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == CURRENT_VERSION

    again = Evaluator(solver, store_path=path)
    assert again.last_score == 0.5
    assert again.total_runs == 3
    assert list(again.history) == [{"t": 1.0, "score": 0.5}]


def test_a_malformed_history_row_is_skipped_not_fatal(tmp_path, solver):
    path = tmp_path / "eval_history.json"
    path.write_text(json.dumps({
        "last_score": 0.25, "total_runs": 2,
        "history": ["not a row", {"t": 1.0, "score": 0.25}],
    }), encoding="utf-8")
    ev = Evaluator(solver, store_path=path)
    assert ev.last_score == 0.25
    assert list(ev.history) == [{"t": 1.0, "score": 0.25}]


def test_an_unreadable_history_is_preserved_before_any_save(tmp_path, solver):
    original = "{ torn mid-write"
    path = tmp_path / "eval_history.json"
    path.write_text(original, encoding="utf-8")

    ev = Evaluator(solver, store_path=path)
    ev._save()

    backup = tmp_path / "eval_history.json.corrupt"
    assert backup.read_text(encoding="utf-8") == original


# ── an empty split is "not measurable", never a score ────────────────

def test_an_empty_split_scores_none_not_zero(solver):
    """The default benchmark has at most three tasks per kind, so its `test`
    split is empty by construction. Scoring the void 0.0 made valid_test_gap
    report the ENTIRE valid score as an overfitting gap."""
    ev = Evaluator(solver, coding_tasks=[], composite_tasks=[], autocompose_tasks=[])
    assert ev.split_sizes()["test"] == 0             # the premise of the finding
    assert ev.score_on_split("test") is None
    assert ev.valid_test_gap() is None


def test_a_populated_split_still_scores_a_float(solver):
    from aegis.eval.generators import variations

    # Eight generated calc tasks give every split members (4/2/2) while
    # keeping the sandboxed solve count small.
    ev = Evaluator(solver, tasks=variations("calc", 8), coding_tasks=[],
                   composite_tasks=[], autocompose_tasks=[])
    assert ev.split_sizes()["test"] > 0
    gap = ev.valid_test_gap()
    assert isinstance(gap, float)
    assert -1.0 <= gap <= 1.0
