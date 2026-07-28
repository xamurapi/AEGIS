"""Every operation, every branch, and what each writes into the trace.

The trace is not a log. It is what the arena scores, what the dataset builder
reads and what an operator is shown, so "which branch ran and why" has to be
recorded correctly — a step that reported the wrong reason is a step that
misleads all three. These tests therefore assert the ``(result, ok, note)`` of
each branch rather than only the final answer.
"""
import pytest

from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning import interpreter as interp
from aegis.layers.reasoning.interpreter import (
    UNAVAILABLE, Interpreter, Step, Trace, _clauses, _renderable,
)


# ── doubles ──────────────────────────────────────────────────────────

class FakeMemory:
    def __init__(self, hits=None):
        self.hits = hits
        self.calls = []

    def retrieve(self, query, limit=5):
        self.calls.append((query, limit))
        return self.hits


class FakeGraph:
    def __init__(self, hits=None):
        self.hits = hits

    def related(self, query):
        return self.hits


class _Skill:
    def __init__(self, name):
        self.name = name


class _SkillLibrary:
    def __init__(self, skills):
        self.skills = skills

    def for_kind(self, kind):
        self.kind = kind
        return self.skills


class FakeSolver:
    def __init__(self, names):
        self.library = _SkillLibrary([_Skill(name) for name in names])


class _Outcome:
    def __init__(self, p_success):
        self.p_success = p_success


class FakeWorldModel:
    def __init__(self, p_success):
        self.p_success = p_success

    def predict_outcome(self, state, family):
        return _Outcome(self.p_success)


class FakeCortex:
    def __init__(self, available=True):
        self.available = available

    def role_available(self, role):
        return self.available


@pytest.fixture
def task():
    return bench.build_family("grid_planning", 1)[0]


def _step(trace, index=0):
    return trace.steps[index]


# ── RETRIEVE ─────────────────────────────────────────────────────────

def test_retrieving_from_memory_reports_what_it_found(task):
    memory = FakeMemory(["a", "b", "c"])
    trace = Interpreter(memory=memory).run(
        [{"op": "RETRIEVE", "source": "memory", "k": 2}], task)
    assert _step(trace).result == ["a", "b"]
    assert _step(trace).note == "3 from memory"
    assert memory.calls == [(task.prompt[:120], 2)]


def test_a_memory_that_returns_nothing_is_not_an_error(task):
    trace = Interpreter(memory=FakeMemory(None)).run(
        [{"op": "RETRIEVE", "source": "memory"}], task)
    assert _step(trace).result == [] and _step(trace).ok


def test_retrieving_from_the_graph_says_so(task):
    trace = Interpreter(graph=FakeGraph(["x"])).run(
        [{"op": "RETRIEVE", "source": "graph"}], task)
    assert _step(trace).note == "1 from the graph"


def test_the_named_source_is_the_one_that_is_asked(task):
    """With both attached, "graph" must not quietly read memory."""
    memory = FakeMemory(["from memory"])
    trace = Interpreter(memory=memory, graph=FakeGraph(["from the graph"])).run(
        [{"op": "RETRIEVE", "source": "graph"}], task)
    assert _step(trace).result == ["from the graph"] and memory.calls == []


def test_a_graph_that_returns_nothing_is_not_an_error(task):
    trace = Interpreter(graph=FakeGraph(None)).run(
        [{"op": "RETRIEVE", "source": "graph"}], task)
    assert _step(trace).result == []


def test_retrieving_skills_asks_for_the_tasks_own_family(task):
    solver = FakeSolver(["roman", "primes"])
    trace = Interpreter(solver=solver).run(
        [{"op": "RETRIEVE", "source": "skills"}], task)
    assert _step(trace).result == ["roman", "primes"]
    assert _step(trace).note == "2 skill(s)"
    assert solver.library.kind == task.family


def test_retrieving_from_a_source_that_is_not_attached_says_which(task):
    trace = Interpreter().run([{"op": "RETRIEVE", "source": "skills"}], task)
    assert _step(trace).note == "no skills attached"


def test_what_was_retrieved_is_available_to_a_later_branch(task):
    strategy = [{"op": "RETRIEVE", "source": "graph"},
                {"op": "BRANCH", "cond": "nothing_retrieved",
                 "then": [{"op": "REFLECT"}]}]
    empty = Interpreter(graph=FakeGraph([])).run(strategy, task)
    full = Interpreter(graph=FakeGraph(["x"])).run(strategy, task)
    assert _step(empty, 1).result is True
    assert _step(full, 1).result is False


# ── PREDICT ──────────────────────────────────────────────────────────

def test_predicting_without_a_world_model_says_so(task):
    trace = Interpreter().run([{"op": "PREDICT"}], task)
    assert _step(trace).result == UNAVAILABLE
    assert _step(trace).note == "no world model attached"


def test_a_prediction_carries_its_horizon(task):
    trace = Interpreter(world_model=FakeWorldModel(0.4)).run(
        [{"op": "PREDICT", "horizon": 3}], task)
    assert _step(trace).result == {"p_success": 0.4, "horizon": 3}
    assert _step(trace).ok and _step(trace).note == "predicted"


def test_a_low_prediction_takes_the_abstaining_branch(task):
    strategy = [{"op": "PREDICT"},
                {"op": "BRANCH", "cond": "p_success_below:0.25",
                 "then": [{"op": "ABSTAIN"}], "else": [{"op": "SOLVE"}]}]
    poor = Interpreter(world_model=FakeWorldModel(0.1)).run(strategy, task)
    good = Interpreter(world_model=FakeWorldModel(0.9)).run(strategy, task)
    assert poor.abstained and not good.abstained


def test_a_prediction_exactly_at_the_threshold_does_not_trigger(task):
    """Strictly below. A threshold that fired at equality would abstain on the
    boundary case, which is the one the threshold was chosen to keep."""
    trace = Interpreter(world_model=FakeWorldModel(0.25)).run(
        [{"op": "PREDICT"},
         {"op": "BRANCH", "cond": "p_success_below:0.25",
          "then": [{"op": "ABSTAIN"}]}], task)
    assert not trace.abstained


def test_a_threshold_that_is_not_a_number_never_fires(task):
    trace = Interpreter(world_model=FakeWorldModel(0.0)).run(
        [{"op": "PREDICT"},
         {"op": "BRANCH", "cond": "p_success_below:soon",
          "then": [{"op": "ABSTAIN"}]}], task)
    assert not trace.abstained


def test_without_a_prediction_the_threshold_condition_is_false(task):
    trace = Interpreter().run(
        [{"op": "BRANCH", "cond": "p_success_below:0.9",
          "then": [{"op": "ABSTAIN"}]}], task)
    assert _step(trace, 0).result is False


# ── SOLVE ────────────────────────────────────────────────────────────

def test_solving_names_the_parser_that_answered(task):
    trace = Interpreter().run([{"op": "SOLVE"}], task)
    assert _step(trace).note == "grid" and _step(trace).ok


def test_solving_a_prompt_with_nothing_in_it_fails_the_step():
    class Bare:
        id = "bare"
        prompt = "Consider the matter."

    trace = Interpreter().run([{"op": "SOLVE"}], Bare())
    assert not _step(trace).ok and _step(trace).note == "no answer (none)"


def test_the_parts_from_decomposition_reach_the_solver():
    """Without this ``DECOMPOSE`` costs a step and changes nothing."""
    task = next(item for item in bench.build_family("arithmetic_chain", 40)
                if not item.features["incomplete"])
    trace = Interpreter().run(
        [{"op": "DECOMPOSE", "max_parts": 8}, {"op": "SOLVE"}], task)
    assert _step(trace, 1).note == "chain" and trace.solved


# ── COMPUTE ──────────────────────────────────────────────────────────

def test_computing_with_no_expression_fails_the_step(task):
    trace = Interpreter(sandbox=lambda *a: {"ok": True}).run(
        [{"op": "COMPUTE", "expr": "   "}], task)
    assert not _step(trace).ok and _step(trace).note == "nothing to compute"


def test_computing_reports_the_sandboxes_error(task):
    trace = Interpreter(sandbox=lambda *a: {"ok": False, "error": "NameError"}).run(
        [{"op": "COMPUTE", "expr": "x"}], task)
    assert not _step(trace).ok and _step(trace).note == "NameError"


def test_computing_takes_the_previous_step_as_its_expression(task):
    """``$last`` is for the case where a model wrote the expression. When the
    previous step left a number rather than an expression there is nothing to
    compute, and saying so beats stringifying it and running it."""
    seen = []

    def sandbox(source, entry, payload):
        seen.append(source)
        return {"ok": True, "result": 1}

    numeric = Interpreter(sandbox=sandbox).run(
        [{"op": "SOLVE"}, {"op": "COMPUTE", "expr": "$last"}], task)
    assert _step(numeric, 1).note == "nothing to compute" and not seen

    written = Interpreter(sandbox=sandbox, cortex=FakeCortex()).run(
        [{"op": "LLM_STEP", "template": "x"}, {"op": "COMPUTE", "expr": "$last"}],
        task)
    assert _step(written, 1).result == 1 and seen


# ── LLM_STEP ─────────────────────────────────────────────────────────

def test_a_model_step_without_a_cortex_says_so(task):
    trace = Interpreter().run([{"op": "LLM_STEP", "template": "x"}], task)
    assert _step(trace).note == "no cortex attached"


def test_a_model_step_whose_role_is_unavailable_says_which(task):
    trace = Interpreter(cortex=FakeCortex(available=False)).run(
        [{"op": "LLM_STEP", "template": "x", "role": "deep"}], task)
    assert _step(trace).note == "role deep unavailable"
    # The step ran and found the door shut. That is not a failed step: an
    # unavailable role is a fact about the deployment, not about the strategy.
    assert _step(trace).ok is True


def test_a_model_step_with_a_live_role_defers_to_the_engine(task):
    trace = Interpreter(cortex=FakeCortex(available=True)).run(
        [{"op": "LLM_STEP", "template": "x"}], task)
    assert _step(trace).note == "cortex step deferred to the engine"


# ── VERIFY ───────────────────────────────────────────────────────────

def test_verifying_against_a_solver_facing_check_uses_it(task):
    class Checked:
        id = "checked"
        prompt = task.prompt
        family = task.family

        def self_check(self, answer):
            return answer == task.expected

    trace = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "task"}], Checked())
    assert _step(trace, 1).result is True and _step(trace, 1).note == "checked"
    # The check ran and agreed. "The check ran" and "the check agreed" are two
    # facts and the trace carries both.
    assert _step(trace, 1).ok is True


def test_a_solver_facing_check_that_fails_says_so(task):
    class Checked:
        id = "checked"
        prompt = task.prompt
        family = task.family

        def self_check(self, answer):
            return False

    trace = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "task"}], Checked())
    assert _step(trace, 1).result is False and _step(trace, 1).note == "check failed"


def test_verifying_by_type_reports_whether_there_is_a_value(task):
    with_value = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "type"}], task)
    without = Interpreter().run([{"op": "VERIFY", "checker": "type"}], task)
    assert _step(with_value, 1).note == "has a value"
    assert _step(without, 0).note == "no value"


def test_verifying_by_confidence_names_which_it_was(task):
    reasoned = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "confidence"}], task)
    guessed = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "confidence"}],
        bench.build_family("missing_data", 1)[0])
    assert _step(reasoned, 1).note == "reasoned"
    assert _step(guessed, 1).note == "guessed"


def test_verifying_by_consistency_is_stable_against_a_deterministic_reasoner(task):
    """Worth nothing here, and a strategy using it should be scored as
    discovering that rather than credited for the attempt."""
    trace = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "consistency"}], task)
    assert _step(trace, 1).result is True and _step(trace, 1).note == "stable"


def test_consistency_without_an_answer_to_compare_is_unstable(task):
    trace = Interpreter().run([{"op": "VERIFY", "checker": "consistency"}], task)
    assert _step(trace, 0).result is False and _step(trace, 0).note == "unstable"


def test_two_absences_agreeing_is_not_consistency():
    """Both runs returning nothing is not "the same answer twice". Without a
    value there is nothing for consistency to be a property of, and a strategy
    that treated it as one would stand behind an answer it never produced.
    """
    class Bare:
        id = "bare"
        prompt = "Consider the matter."

    trace = Interpreter().run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "consistency"}], Bare())
    assert _step(trace, 1).result is False


def test_a_failed_verification_is_what_verify_failed_reads(task):
    strategy = [{"op": "VERIFY", "checker": "type"},
                {"op": "BRANCH", "cond": "verify_failed",
                 "then": [{"op": "REFLECT"}]}]
    trace = Interpreter().run(strategy, task)
    assert _step(trace, 1).result is True


def test_verify_failed_is_false_when_nothing_was_verified(task):
    trace = Interpreter().run(
        [{"op": "BRANCH", "cond": "verify_failed", "then": [{"op": "REFLECT"}]}],
        task)
    assert _step(trace, 0).result is False


# ── VOTE ─────────────────────────────────────────────────────────────

def test_a_vote_of_zero_rounds_still_runs_one(task):
    trace = Interpreter().run(
        [{"op": "VOTE", "n": 0, "body": [{"op": "SOLVE"}]}], task)
    assert trace.answer == task.expected


def test_taking_the_first_vote_says_so(task):
    trace = Interpreter().run(
        [{"op": "VOTE", "n": 2, "agg": "first", "body": [{"op": "SOLVE"}]}], task)
    assert _step(trace).note == "first"


def test_a_majority_reports_how_large_it_was(task):
    trace = Interpreter().run(
        [{"op": "VOTE", "n": 3, "agg": "majority", "body": [{"op": "SOLVE"}]}],
        task)
    assert _step(trace).note == "majority 3/3"


def test_a_unanimous_vote_says_it_was_unanimous(task):
    trace = Interpreter().run(
        [{"op": "VOTE", "n": 2, "agg": "unanimous", "body": [{"op": "SOLVE"}]}],
        task)
    assert _step(trace).note == "unanimous" and _step(trace).ok


def test_a_vote_over_an_empty_body_has_nothing_to_aggregate(task):
    trace = Interpreter().run([{"op": "VOTE", "n": 2, "body": []}], task)
    assert _step(trace).result is None and _step(trace).note == "no votes"


# ── REFLECT and ABSTAIN ──────────────────────────────────────────────

def test_reflecting_reports_the_state_it_can_see(task):
    trace = Interpreter().run(
        [{"op": "DECOMPOSE", "max_parts": 3}, {"op": "SOLVE"},
         {"op": "VERIFY", "checker": "type"}, {"op": "REFLECT"}], task)
    assert _step(trace, 3).result == {"verified": True,
                                      "parts": len(_clauses(task.prompt))}


def test_an_abstention_records_the_reason_given(task):
    trace = Interpreter().run(
        [{"op": "ABSTAIN", "reason": "the units do not match"}], task)
    assert _step(trace).note == "the units do not match"


def test_an_abstention_with_no_reason_still_gives_one(task):
    trace = Interpreter().run([{"op": "ABSTAIN"}], task)
    assert _step(trace).note == "insufficient data"


# ── failure and timing ───────────────────────────────────────────────

def test_an_unknown_operation_is_recorded_as_a_failed_step(task):
    trace = Interpreter().run([{"op": "TELEPORT"}], task)
    assert _step(trace).ok is False


def test_a_strategy_that_cannot_even_be_walked_is_recorded_not_raised(task):
    """The tick must survive a malformed strategy. A raise here would take
    down the cognitive cycle for a bad row in a store."""
    class Hostile(list):
        def __iter__(self):
            raise RuntimeError("boom")

    trace = Interpreter().run(Hostile([{"op": "SOLVE"}]), task)
    assert _step(trace).op == "ERROR" and not _step(trace).ok


def test_a_step_records_how_long_it_took(task, monkeypatch):
    """Milliseconds, measured as end minus start. Both halves of that have been
    got wrong before, and neither shows up against a frozen clock."""
    ticks = iter([2.0, 3.0, 3.25, 5.0])

    class Stub:
        def monotonic(self):
            return next(ticks)

    monkeypatch.setattr(interp, "CLOCK", Stub())
    trace = Interpreter().run([{"op": "REFLECT"}], task)
    assert _step(trace).elapsed_ms == pytest.approx(250.0)
    assert trace.elapsed_ms == pytest.approx(3000.0)


# ── "the step ran" is not "the step said yes" ────────────────────────

def test_a_step_that_did_its_job_is_ok_even_when_its_answer_is_no(task):
    """``ok`` records whether the operation could be performed; ``result``
    records what it concluded. Collapsing the two would make "the checker said
    the answer is wrong" indistinguishable from "there was no checker", and the
    arena scores those differently.
    """
    missing = bench.build_family("missing_data", 1)[0]
    cases = [
        # (strategy, task, step index, expected ok, expected result)
        ([{"op": "PREDICT"}], task, 0, True, UNAVAILABLE),
        ([{"op": "LLM_STEP", "template": "x"}], task, 0, True, UNAVAILABLE),
        ([{"op": "SOLVE"}, {"op": "VERIFY", "checker": "task"}],
         task, 1, True, None),
        ([{"op": "SOLVE"}, {"op": "VERIFY", "checker": "confidence"}],
         missing, 1, True, False),
        ([{"op": "VERIFY", "checker": "type"}], task, 0, True, False),
        ([{"op": "VERIFY", "checker": "consistency"}], task, 0, True, False),
        ([{"op": "VERIFY", "checker": "vibes"}], task, 0, True, None),
        ([{"op": "BRANCH", "cond": "insufficient"}], task, 0, True, True),
        ([{"op": "LOOP", "max_iter": 2, "body": [{"op": "REFLECT"}]}],
         task, 0, True, 2),
        ([{"op": "ABSTAIN"}], task, 0, True, None),
        ([{"op": "VOTE", "n": 2, "agg": "first", "body": [{"op": "SOLVE"}]}],
         task, 0, True, task.expected),
        ([{"op": "VOTE", "n": 2, "body": [{"op": "SOLVE"}]}],
         task, 0, True, task.expected),
    ]
    for strategy, subject, index, expected_ok, expected_result in cases:
        trace = Interpreter().run(strategy, subject)
        step = _step(trace, index)
        assert step.ok is expected_ok, (strategy, step)
        assert step.result == expected_result, (strategy, step)


def test_a_vote_that_could_not_run_is_not_ok(task):
    empty = Interpreter().run([{"op": "VOTE", "n": 2, "body": []}], task)
    starved = Interpreter(max_steps=1).run(
        [{"op": "VOTE", "n": 2, "body": [{"op": "SOLVE"}]}], task)
    assert _step(empty).ok is False and _step(empty).note == "no votes"
    assert _step(starved).ok is False and _step(starved).note == "no votes"


def test_consistency_re_solves_with_the_same_parts_it_was_given():
    """Re-solving from the undivided prompt would call a decomposed answer
    unstable every time, and every strategy that checked consistency after
    decomposing would throw away a correct answer."""
    task = next(item for item in bench.build_family("arithmetic_chain", 40)
                if not item.features["incomplete"])
    trace = Interpreter().run(
        [{"op": "DECOMPOSE", "max_parts": 8}, {"op": "SOLVE"},
         {"op": "VERIFY", "checker": "consistency"}], task)
    assert _step(trace, 2).note == "stable"


# ── clause splitting ─────────────────────────────────────────────────

def test_a_clause_keeps_its_question_mark_and_loses_its_full_stop():
    assert _clauses("One. Two? Three.") == ["One", "Two?", "Three"]


def test_an_empty_prompt_has_no_clauses():
    assert _clauses("") == [] and _clauses(None) == []


# ── the trace's own defaults ─────────────────────────────────────────

def test_a_step_is_recorded_as_having_worked_until_told_otherwise():
    """The default matters: a step is created before its handler runs, and a
    handler that returns nothing at all should not silently mark it failed."""
    assert Step(index=0, op="REFLECT").ok is True


def test_a_fresh_trace_claims_neither_abstention_nor_exhaustion():
    trace = Trace()
    assert trace.abstained is False and trace.budget_exhausted is False


def test_an_absent_value_renders_as_absent_not_as_the_word_none():
    """A trace is read by people and by the dataset builder. ``"None"`` and
    ``None`` are different facts to both."""
    assert _renderable(None) is None
    assert _renderable("None") == "None"
