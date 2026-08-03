"""The strategy library and the reasoning engine (spec M6.4, M6.5, M6.9).

Admission is the boundary: a strategy enters only through ``admit``, so if that
refuses everything the interpreter cannot run, nothing downstream needs to
re-check. The rest is bookkeeping that has to be per-family, because a strategy
is rarely good or bad in general.
"""
import pytest

from aegis.eval import reasoning_bench
from aegis.eval import reasoning_bench as bench  # the same module, under both names
from aegis.layers.reasoning import (
    MIN_ATTEMPTS_PER_FAMILY, MIN_RESULTS_FOR_WEAKNESS, ReasoningEngine,
)
from aegis.layers.reasoning.dsl import DSLError, validate
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES, Library, Strategy


@pytest.fixture
def library(tmp_path):
    return Library(store_path=tmp_path / "strategies.json")


@pytest.fixture
def engine(tmp_path):
    return ReasoningEngine(store_path=tmp_path / "strategies.json")


# ── the built-ins ────────────────────────────────────────────────────

def test_the_eight_built_in_strategies_of_the_spec_are_present(library):
    assert set(library.strategies) == {
        "direct", "verify_then_answer", "decompose_solve_combine",
        "program_of_thought", "self_consistency_k", "analogy_from_graph",
        "predictive_check", "abstain_on_low_confidence"}


def test_every_built_in_is_admissible():
    for name, steps in BUILTIN_STRATEGIES.items():
        assert validate(steps) == [], name


def test_no_built_in_solves_the_benchmark_on_its_own(engine):
    """The room the synthesiser needs (M6.7).

    A library that already contained the best strategy would make M6
    unfalsifiable: every measured gain would be the selector rediscovering what
    someone wrote by hand. Decomposition and abstention therefore live in
    different built-ins, and neither alone is enough.
    """
    tasks = [bench.build(index) for index in range(160)]
    for strategy in engine.library.active():
        solved = sum(1 for task in tasks
                     if engine.interpreter.run(strategy, task).solved)
        assert solved / len(tasks) < 0.9, strategy.name


def test_combining_decomposition_and_abstention_beats_every_built_in(engine):
    """The gain stage 9 is asked to find, shown to exist here."""
    combined = engine.library.admit("both", [
        {"op": "DECOMPOSE", "max_parts": 8},
        {"op": "SOLVE"},
        {"op": "VERIFY", "checker": "confidence"},
        {"op": "BRANCH", "cond": "insufficient", "then": [{"op": "ABSTAIN"}]},
    ])
    tasks = [bench.build(index) for index in range(160)]

    def score(strategy):
        return sum(1 for task in tasks
                   if engine.interpreter.run(strategy, task).solved) / len(tasks)

    best_builtin = max(score(strategy) for strategy in engine.library.builtins())
    assert score(combined) - best_builtin >= 0.10


def test_a_built_in_is_repaired_if_a_stored_copy_drifted(tmp_path):
    """The baseline has to be the same baseline across runs, or a comparison
    against it means nothing."""
    path = tmp_path / "strategies.json"
    first = Library(store_path=path)
    first.strategies["direct"].steps = [{"op": "REFLECT"}]
    first.save()
    assert Library(store_path=path).get("direct").steps == BUILTIN_STRATEGIES["direct"]


def test_a_built_in_cannot_be_retired(library):
    """A measuring stick that can be discarded when it reads badly is not one."""
    assert library.retire("direct") is False
    assert not library.get("direct").retired


def test_a_stored_built_in_marked_retired_comes_back_active(tmp_path):
    """Otherwise one bad write to the store quietly removes the baseline, and
    every later comparison is against a smaller set than it claims."""
    path = tmp_path / "strategies.json"
    first = Library(store_path=path)
    first.strategies["direct"].status = "retired"
    first.save()
    assert Library(store_path=path).get("direct").retired is False


def test_retiring_a_synthesised_strategy_takes_it_out_of_use(library):
    library.admit("mine", [{"op": "REFLECT"}])
    assert library.retire("mine", reason="never won anything") is True
    assert library.get("mine").retired is True
    assert "mine" not in [strategy.name for strategy in library.active()]


def test_retiring_something_that_does_not_exist_reports_so(library):
    assert library.retire("imaginary") is False


def test_a_strategy_is_in_use_until_it_is_retired(library):
    assert Strategy(name="fresh").retired is False
    assert Strategy.from_dict({"name": "fresh"}).retired is False


# ── standing ─────────────────────────────────────────────────────────

def test_the_three_standings_are_distinguished(library):
    ordinary = library.get("direct")
    trial = library.admit("t", [{"op": "REFLECT"}], status="trial")
    gone = library.admit("g", [{"op": "REFLECT"}, {"op": "REFLECT"}])
    library.retire("g")
    assert (ordinary.status, ordinary.on_trial, ordinary.retired) == \
        ("active", False, False)
    assert (trial.status, trial.on_trial, trial.retired) == ("trial", True, False)
    assert (gone.status, gone.on_trial, gone.retired) == ("retired", False, True)


def test_a_trial_is_in_use_but_not_in_service(library):
    """It runs — that is what a trial is — but it is not what the system would
    answer with, and the two lists are what say so."""
    trial = library.admit("t", [{"op": "REFLECT"}], status="trial")
    assert trial in library.in_use() and trial not in library.active()
    assert trial in library.trials()


def test_a_retired_strategy_is_in_neither_list(library):
    gone = library.admit("g", [{"op": "REFLECT"}], status="trial")
    library.retire("g")
    assert gone not in library.in_use() and gone not in library.trials()


def test_an_unknown_standing_falls_back_to_ordinary_use(library):
    assert library.admit("t", [{"op": "REFLECT"}], status="probation").status \
        == "active"
    assert Strategy.from_dict({"name": "x", "status": "probation"}).status \
        == "active"


def test_a_trial_is_promoted_only_from_trial(library):
    library.admit("t", [{"op": "REFLECT"}], status="trial")
    assert library.promote("t") is True
    assert library.get("t").status == "active"
    assert library.promote("t") is False
    assert library.promote("direct") is False
    assert library.promote("imaginary") is False


def test_a_trial_declares_which_class_it_is_for(library):
    library.admit("mine", [{"op": "REFLECT"}], status="trial",
                  family="grid_planning", weakness="family=grid_planning",
                  incumbent="direct")
    trial = library.get("mine")
    assert trial.family == "grid_planning" and trial.incumbent == "direct"
    assert library.trials("grid_planning") == [trial]
    assert library.trials("magnitude") == []


def test_a_trial_with_no_class_is_for_all_of_them(library):
    """Measured: filtering these out meant a trial accepted for a weakness that
    spanned classes got no traffic at all for a whole thirty-cycle run."""
    library.admit("mine", [{"op": "REFLECT"}], status="trial")
    assert library.trials("magnitude") and library.trials("grid_planning")


def test_a_trials_standing_survives_a_restart(tmp_path):
    path = tmp_path / "strategies.json"
    first = Library(store_path=path)
    first.admit("mine", [{"op": "REFLECT"}], status="trial",
                family="grid_planning", incumbent="direct",
                weakness="family=grid_planning")
    first.save()

    trial = Library(store_path=path).get("mine")
    assert trial.on_trial and trial.family == "grid_planning"
    assert trial.incumbent == "direct" and trial.weakness == "family=grid_planning"


def test_a_file_written_before_trials_existed_is_still_read(tmp_path):
    """Dropping the old boolean would put every retired strategy back into
    rotation on the next restart."""
    import json

    path = tmp_path / "strategies.json"
    Library(store_path=path).save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["strategies"].append({"name": "old", "steps": [{"op": "REFLECT"}],
                              "retired": True})
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert Library(store_path=path).get("old").retired is True


# ── admission ────────────────────────────────────────────────────────

def test_admission_refuses_what_the_interpreter_cannot_run(library):
    with pytest.raises(DSLError):
        library.admit("bad", [{"op": "EXEC", "code": "rm -rf /"}])
    assert "bad" not in library.strategies
    assert library.refused == 1


def test_admission_refuses_a_duplicate_shape(library):
    """Two names for one strategy split its record in half and make both halves
    look inconclusive."""
    with pytest.raises(DSLError, match="identical to 'direct'"):
        library.admit("also_direct", [{"op": "SOLVE"}])


def test_admission_refuses_a_name_already_taken(library):
    with pytest.raises(DSLError, match="already exists"):
        library.admit("direct", [{"op": "REFLECT"}])


def test_admission_refuses_a_nameless_strategy(library):
    with pytest.raises(DSLError):
        library.admit("   ", [{"op": "REFLECT"}])


def test_an_admitted_strategy_is_stored_normalised(library):
    strategy = library.admit("mine", [{"k": 2, "source": "memory",
                                       "op": "RETRIEVE"}])
    assert list(strategy.steps[0]) == ["op", "k", "source"]


def test_an_unknown_origin_falls_back_to_synth(library):
    assert library.admit("mine", [{"op": "REFLECT"}],
                         origin="somewhere").origin == "synth"


# ── the record ───────────────────────────────────────────────────────

def test_the_record_is_kept_per_family(library):
    library.note_result("direct", "grid_planning", solved=True)
    library.note_result("direct", "missing_data", solved=False)
    direct = library.get("direct")
    assert direct.accuracy("grid_planning") == 1.0
    assert direct.accuracy("missing_data") == 0.0
    assert direct.used() == 2


def test_one_lucky_attempt_does_not_beat_a_long_record(library):
    """The whole reason to keep the interval is to stop this."""
    lucky = library.admit("lucky", [{"op": "REFLECT"}])
    lucky.note("grid_planning", solved=True)
    steady = library.get("direct")
    for index in range(50):
        steady.note("grid_planning", solved=index < 40)
    assert steady.lower("grid_planning") > lucky.lower("grid_planning")
    assert library.best_for("grid_planning", min_used=1) is steady


def test_accuracy_is_the_share_solved(library):
    direct = library.get("direct")
    for index in range(4):
        direct.note("grid_planning", solved=index == 0)
    assert direct.accuracy("grid_planning") == 0.25
    assert direct.accuracy() == 0.25


def test_the_cost_reported_is_the_cost_per_use(library):
    """A total masquerading as an average makes every well-used strategy look
    ruinous, and the selector prefers cost when accuracy ties."""
    direct = library.get("direct")
    direct.note("grid_planning", solved=True, cost_ms=10.0)
    direct.note("grid_planning", solved=True, cost_ms=30.0)
    assert direct.mean_cost_ms("grid_planning") == 20.0
    assert direct.mean_cost_ms() == 20.0


def test_an_attempt_is_not_an_abstention_unless_it_says_so(library):
    """Both ways in. An attempt recorded as an abstention it never made would
    read as caution the strategy never showed."""
    library.note_result("direct", "grid_planning", solved=True)
    library.get("direct").note("grid_planning", solved=True)
    assert library.get("direct").record["grid_planning"]["abstained"] == 0


def test_a_family_nobody_has_tried_has_no_best(library):
    assert library.best_for("grid_planning") is None


def test_an_unknown_strategy_name_is_ignored_rather_than_raising(library):
    library.note_result("nonexistent", "grid_planning", solved=True)
    assert library.status()["used"] == 0


# ── persistence ──────────────────────────────────────────────────────

def test_the_library_defaults_to_the_configured_reasoning_directory(monkeypatch,
                                                                    tmp_path):
    """Where it writes is not incidental: an A/B harness redirects this
    directory to keep its arms out of the repository's live state."""
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "REASONING_DIR", tmp_path / "elsewhere")
    (tmp_path / "elsewhere").mkdir()
    library = Library()
    library.save()
    assert (tmp_path / "elsewhere" / "strategies.json").exists()


def test_the_record_survives_a_restart(tmp_path):
    path = tmp_path / "strategies.json"
    first = Library(store_path=path)
    first.admit("mine", [{"op": "REFLECT"}])
    first.note_result("mine", "grid_planning", solved=True, cost_ms=12.0)
    first.save()

    second = Library(store_path=path)
    assert second.get("mine").solved("grid_planning") == 1
    assert second.get("mine").mean_cost_ms("grid_planning") == 12.0


def test_a_stored_strategy_the_grammar_has_moved_past_is_dropped(tmp_path):
    """Keeping it would mean the interpreter meets an operation it does not
    have — at run time, inside a strategy."""
    path = tmp_path / "strategies.json"
    library = Library(store_path=path)
    library.strategies["stale"] = Strategy(name="stale", steps=[{"op": "EXEC"}])
    library.save()
    assert "stale" not in Library(store_path=path).strategies


def test_a_malformed_stored_row_does_not_stop_the_rest_loading(tmp_path):
    import json

    path = tmp_path / "strategies.json"
    Library(store_path=path).save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["strategies"].append("not a strategy")
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert len(Library(store_path=path).strategies) == len(BUILTIN_STRATEGIES)


def test_the_library_stays_within_its_cap(tmp_path):
    library = Library(store_path=tmp_path / "s.json", max_strategies=10)
    for index in range(20):
        library.admit(f"s{index}", [{"op": "REFLECT"}] * (index + 1))
    assert len(library.strategies) <= 10
    assert len(library.builtins()) == len(BUILTIN_STRATEGIES)


# ── the engine ───────────────────────────────────────────────────────

def test_the_queue_refills_rather_than_reporting_nothing_to_do(engine):
    """The supply of problems is unbounded; an empty queue is a fact about the
    queue, not about whether there is anything to think about."""
    engine.queue.clear()
    assert engine.has_queued_task()
    assert engine.queue


def test_problems_are_met_in_the_same_order_on_every_run(tmp_path):
    first = ReasoningEngine(store_path=tmp_path / "a.json")
    second = ReasoningEngine(store_path=tmp_path / "b.json")
    first.refill(8)
    second.refill(8)
    assert [task.id for task in first.queue] == [task.id for task in second.queue]


def test_solving_records_one_row_per_attempt(engine):
    summary = engine.solve(5)
    assert summary["worked"] == 5
    assert len(engine.results) == 5 and len(engine.traces) == 5


def test_every_active_strategy_is_tried_before_any_is_preferred(engine):
    """Otherwise one lucky first attempt takes a family permanently, because
    nothing else ever runs there again."""
    tasks = bench.build_family("grid_planning", 40)
    chosen = []
    for task in tasks[:len(engine.library.active()) * MIN_ATTEMPTS_PER_FAMILY]:
        chosen.append(engine.attempt(task)["strategy"])
    assert set(chosen) == {s.name for s in engine.library.active()}


def test_a_family_with_too_little_evidence_is_not_called_weak(engine):
    for task in bench.build_family("missing_data", MIN_RESULTS_FOR_WEAKNESS - 1):
        engine.attempt(task)
    assert engine.top_weakness() is None


def test_the_worst_measured_family_is_the_weakness(engine):
    for task in bench.build_family("missing_data", 24):
        engine.attempt(task)
    for task in bench.build_family("grid_planning", 24):
        engine.attempt(task)
    weakness = engine.top_weakness()
    assert weakness["family"] == "missing_data"
    assert weakness["used"] == 24


def test_a_family_that_never_fails_is_not_a_weakness(engine):
    for task in bench.build_family("grid_planning", 24):
        engine.attempt(task)
    assert not [entry for entry in engine.weaknesses()
                if entry["family"] == "grid_planning"]


def test_a_confident_wrong_answer_is_counted_separately(engine):
    """The number that matters most: wrong while sure is worse than silent."""
    for task in bench.build_family("missing_data", 8):
        engine.attempt(task, engine.library.get("direct"))
    assert engine.confident_errors == 8
    assert engine.abstentions == 0


def test_abstaining_is_not_a_confident_error(engine):
    for task in bench.build_family("missing_data", 8):
        engine.attempt(task, engine.library.get("abstain_on_low_confidence"))
    assert engine.confident_errors == 0
    assert engine.abstentions == 8 and engine.solved_count == 8


def test_the_evidence_is_bounded(engine, monkeypatch):
    import aegis.config as cfg

    monkeypatch.setattr(cfg, "REASON_MAX_TRACES", 100)
    engine.solve(140)
    assert len(engine.results) == 100 and len(engine.traces) == 100


def test_the_dataset_pairs_a_trace_with_its_outcome(engine):
    engine.solve(4)
    rows = engine.dataset()
    assert len(rows) == 4
    assert rows[0]["trace"]["strategy"] == rows[0]["strategy"]


def test_the_genome_reaches_the_interpreter(engine):
    engine.set_genome({"reason_decompose_parts": 7, "reason_budget": 3})
    assert engine.interpreter.genome["reason_decompose_parts"] == 7
    assert engine._budget() == 3


def test_a_nonsense_budget_gene_falls_back_to_the_configured_maximum(engine):
    import aegis.config as cfg

    engine.set_genome({"reason_budget": "plenty"})
    assert engine._budget() == cfg.REASON_MAX_STEPS


def test_the_held_out_score_never_meets_the_queue(engine):
    """Held out by construction: the queue walks forward from zero and the
    holdout walks back from a far index."""
    engine.refill(64)
    queued = {task.id for task in engine.queue}
    holdout = {bench.build(10_000_000 - offset).id for offset in range(64)}
    assert not (queued & holdout)
    assert 0.0 <= engine.holdout_score(8) <= 1.0


def test_the_holdout_metric_is_the_holdout_score_not_the_live_accuracy(engine,
                                                                       monkeypatch):
    """`aegis.reason.pass_holdout` used to carry `accuracy()` — the in-sample,
    live-queue number — while the genuine `holdout_score()` was called by
    nobody. The discovery engine mines laws over exactly this metric, so every
    "law" about holdout reasoning performance was actually about the data the
    engine trains on. The metric must carry the holdout, and between cadence
    points it stays absent rather than substituted.
    """
    from aegis.telemetry import metrics as M

    class _Telemetry:
        def __init__(self):
            self.rows = []

        def record(self, metric, value, tick, tags=None):
            self.rows.append((metric, value))

    engine.telemetry = _Telemetry()
    engine.attempts, engine.solved_count = 10, 10       # in-sample accuracy 1.0
    monkeypatch.setattr(engine, "holdout_score", lambda count=32: 0.25)

    engine.publish_metrics(tick=1)
    recorded = dict(engine.telemetry.rows)
    assert recorded[M.REASON_PASS_HOLDOUT] == 0.25      # holdout, not accuracy

    # Inside the cadence window: absent, not approximated with something else.
    engine.telemetry.rows.clear()
    engine.publish_metrics(tick=2)
    assert all(metric != M.REASON_PASS_HOLDOUT
               for metric, _ in engine.telemetry.rows)


def test_the_held_out_set_walks_backwards_from_its_far_index(engine, monkeypatch):
    """Walking forward from ten million would still be held out today and would
    quietly stop being so the moment the working cursor was given a head start.
    """
    seen = []
    original = reasoning_bench.build
    monkeypatch.setattr(reasoning_bench, "build",
                        lambda index: seen.append(index) or original(index))
    engine.holdout_score(3)
    assert seen == [10_000_000, 9_999_999, 9_999_998]


def test_selection_always_returns_something_that_can_run(engine):
    """Falling through to nothing would make the engine unable to attempt the
    task at all, which is worse than any choice it could have made."""
    for strategy in engine.library.active():
        for _ in range(MIN_ATTEMPTS_PER_FAMILY):
            strategy.note("grid_planning", solved=True)
    chosen = engine.select("grid_planning")
    assert chosen in engine.library.active()


def test_a_library_emptied_at_run_time_is_reseeded(engine):
    engine.library.strategies.clear()
    assert engine.select("grid_planning") is not None
    assert len(engine.library.active()) == len(BUILTIN_STRATEGIES)


def test_accuracy_is_the_share_of_attempts_that_were_solved(engine):
    for task in bench.build_family("missing_data", 3):
        engine.attempt(task, engine.library.get("abstain_on_low_confidence"))
    for task in bench.build_family("missing_data", 1):
        engine.attempt(task, engine.library.get("direct"))
    assert engine.attempts == 4 and engine.solved_count == 3
    assert engine.accuracy() == 0.75


def test_the_per_family_table_reports_a_share_not_a_count(engine):
    for index, task in enumerate(bench.build_family("missing_data", 4)):
        engine.attempt(task, engine.library.get(
            "abstain_on_low_confidence" if index else "direct"))
    entry = engine.per_family()["missing_data"]
    assert entry["used"] == 4 and entry["solved"] == 3
    assert entry["accuracy"] == 0.75


def test_the_published_rates_are_shares_of_the_attempts(engine):
    from aegis.telemetry import metrics as M

    recorded = {}

    class Telemetry:
        def record(self, metric, value, tick, tags=None):
            recorded.setdefault(metric, value)

    engine.telemetry = Telemetry()
    for index, task in enumerate(bench.build_family("missing_data", 4)):
        engine.attempt(task, engine.library.get(
            "abstain_on_low_confidence" if index else "direct"))
    engine.publish_metrics(tick=1)
    assert recorded[M.REASON_ABSTAIN_RATE] == 0.75
    assert recorded[M.REASON_CONFIDENT_ERROR] == 0.25


def test_an_answer_that_is_merely_absent_is_not_a_confident_error(engine):
    """Silence and a wrong number are different failures, and only one of them
    is the one abstention exists to prevent."""
    class Bare:
        id = "bare"
        family = "bare"
        prompt = "Consider the matter."

        def verify(self, answer):
            return False

    row = engine.attempt(Bare(), engine.library.get("direct"))
    assert row["solved"] is False and row["confident_error"] is False


def test_status_reports_what_an_operator_needs(engine):
    engine.solve(3)
    status = engine.status()
    assert status["attempts"] == 3
    assert status["library"]["builtin"] == len(BUILTIN_STRATEGIES)


def test_the_table_has_one_row_per_strategy(engine):
    assert len(engine.library.table()) == len(BUILTIN_STRATEGIES)
