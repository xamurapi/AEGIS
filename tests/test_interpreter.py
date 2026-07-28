"""The strategy interpreter (spec M6.3, M6.10, M6.11).

Three claims are load-bearing and each is tested against the failure it exists
to prevent: the step budget cannot be exceeded (including by a loop whose
condition is always true), a strategy reaches nothing it was not handed, and
two runs of one strategy on one task produce the same trace.
"""
import pytest

from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning.dsl import MAX_LOOP_ITERATIONS
from aegis.layers.reasoning.interpreter import (
    ANSWERING_OPS, MAX_DECOMPOSE_PARTS, UNAVAILABLE, Interpreter, Trace,
    _clauses,
)


@pytest.fixture
def interpreter():
    return Interpreter()


@pytest.fixture
def task():
    return bench.build_family("arithmetic_chain", 1)[0]


# ── the budget cannot be exceeded ────────────────────────────────────

def test_a_loop_whose_condition_is_always_true_still_terminates(interpreter, task):
    """The named negative test of M6.10.

    ``insufficient`` is true whenever the working answer is a guess, and the
    body here never produces one. Without the ceiling this loops forever.
    """
    strategy = [{"op": "LOOP", "while": "insufficient",
                 "max_iter": MAX_LOOP_ITERATIONS, "body": [{"op": "REFLECT"}]}]
    trace = interpreter.run(strategy, task, budget=12)
    assert trace.step_count <= 12 + 1        # +1 for the budget marker


def test_the_budget_is_the_total_across_nesting(interpreter, task):
    strategy = [{"op": "LOOP", "max_iter": 8, "body": [{"op": "REFLECT"}] * 8}]
    trace = interpreter.run(strategy, task, budget=5)
    assert trace.budget_exhausted
    assert len([step for step in trace.steps if step.op == "REFLECT"]) <= 5


def test_a_gene_cannot_buy_more_budget_than_the_configured_maximum(task):
    """A gene may spend within a limit, never past it."""
    interpreter = Interpreter(max_steps=4)
    trace = interpreter.run([{"op": "REFLECT"}] * 20, task, budget=999)
    assert len([step for step in trace.steps if step.op == "REFLECT"]) == 4


def test_a_budget_of_zero_still_runs_one_step(task):
    """Clamped to at least one: a strategy that cannot take a single step is
    indistinguishable from no strategy, and hides the fault."""
    interpreter = Interpreter()
    trace = interpreter.run([{"op": "REFLECT"}], task, budget=0)
    assert trace.steps and trace.steps[0].op == "REFLECT"


def test_the_budget_marker_says_why_the_strategy_stopped(interpreter, task):
    trace = interpreter.run([{"op": "REFLECT"}] * 6, task, budget=2)
    assert any(step.op == "BUDGET" and not step.ok for step in trace.steps)


# ── it reaches only what it was handed ───────────────────────────────

def test_retrieving_from_nothing_is_unavailable_not_a_crash(interpreter, task):
    trace = interpreter.run([{"op": "RETRIEVE", "source": "graph"}], task)
    assert trace.steps[0].ok and trace.steps[0].result == []


def test_compute_without_a_sandbox_never_evaluates_anything(interpreter, task):
    """A synthesised expression evaluated in this process is arbitrary code
    execution. With no sandbox attached it must not be evaluated at all."""
    trace = interpreter.run([{"op": "COMPUTE", "expr": "__import__('os').getcwd()"}],
                            task)
    assert trace.answer == UNAVAILABLE


def test_compute_goes_through_the_injected_sandbox():
    calls = []

    def sandbox(source, entry, payload):
        calls.append(source)
        return {"ok": True, "result": 42}

    interpreter = Interpreter(sandbox=sandbox)
    trace = interpreter.run([{"op": "COMPUTE", "expr": "6 * 7"}],
                            bench.build(0))
    assert trace.answer == 42 and "6 * 7" in calls[0]


def test_a_step_that_raises_costs_the_step_not_the_run(task):
    def sandbox(source, entry, payload):
        raise RuntimeError("sandbox down")

    interpreter = Interpreter(sandbox=sandbox)
    trace = interpreter.run([{"op": "COMPUTE", "expr": "1"}, {"op": "REFLECT"}],
                            task)
    assert not trace.steps[0].ok and trace.steps[1].ok


def test_an_unknown_operation_is_noted_and_skipped(interpreter, task):
    trace = interpreter.run([{"op": "EXEC"}, {"op": "REFLECT"}], task)
    assert trace.steps[0].note == "unknown operation"
    assert trace.steps[1].op == "REFLECT"


# ── the answer comes only from operations that produce answers ───────

def test_verifying_does_not_become_the_answer(interpreter):
    """The obvious version of this turned every verified answer into ``True``."""
    task = bench.build_family("grid_planning", 1)[0]
    trace = interpreter.run([{"op": "SOLVE"}, {"op": "VERIFY", "checker": "type"}],
                            task)
    assert trace.answer == task.expected


def test_only_answering_operations_set_the_answer():
    assert ANSWERING_OPS == {"SOLVE", "COMPUTE", "LLM_STEP", "VOTE"}


def test_verify_never_consults_the_benchmarks_grader(interpreter):
    """Measured: a strategy that abstained whenever the grader disagreed scored
    100% on a benchmark it had not solved at all."""
    task = bench.build_family("missing_data", 1)[0]
    trace = interpreter.run([{"op": "SOLVE"}, {"op": "VERIFY", "checker": "task"}],
                            task)
    assert trace.steps[1].result is None
    assert "no solver-facing check" in trace.steps[1].note


def test_verify_by_confidence_separates_reasoning_from_guessing(interpreter):
    reasoned = interpreter.run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "confidence"}],
        bench.build_family("grid_planning", 1)[0])
    guessed = interpreter.run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": "confidence"}],
        bench.build_family("missing_data", 1)[0])
    assert reasoned.steps[1].result is True
    assert guessed.steps[1].result is False


# ── decomposition ────────────────────────────────────────────────────

def test_decomposition_splits_a_chain_written_as_one_sentence():
    """Splitting only on the full stop leaves a chain of instructions
    undivided — which is the case decomposition exists for."""
    parts = _clauses("start with 5, then add 3, then multiply by 2. "
                     "What is the result?")
    assert parts == ["start with 5", "add 3", "multiply by 2",
                     "What is the result?"]


def test_the_part_cap_is_a_real_limit(interpreter):
    task = bench.build_family("arithmetic_chain", 1)[0]
    trace = interpreter.run([{"op": "DECOMPOSE", "max_parts": 2}], task)
    assert len(trace.steps[0].result) == 2


def test_the_part_cap_cannot_be_raised_past_the_ceiling(task):
    interpreter = Interpreter(genome={"reason_decompose_parts": 999})
    trace = interpreter.run([{"op": "DECOMPOSE"}], task)
    assert len(trace.steps[0].result) <= MAX_DECOMPOSE_PARTS


def test_a_strategy_that_names_no_cap_gets_the_genes(task):
    narrow = Interpreter(genome={"reason_decompose_parts": 2})
    wide = Interpreter(genome={"reason_decompose_parts": 8})
    assert (len(narrow.run([{"op": "DECOMPOSE"}], task).steps[0].result)
            < len(wide.run([{"op": "DECOMPOSE"}], task).steps[0].result))


def test_a_truncated_chain_is_answered_without_confidence(interpreter):
    """The reasoner did the arithmetic it was shown and never saw the end of
    the chain. Reporting that as a confident answer is the mistake."""
    task = bench.build_family("arithmetic_chain", 40)[0]
    trace = interpreter.run(
        [{"op": "DECOMPOSE", "max_parts": 2}, {"op": "SOLVE"},
         {"op": "VERIFY", "checker": "confidence"}], task)
    assert trace.steps[2].result is False


# ── branching and voting ─────────────────────────────────────────────

def test_a_branch_takes_exactly_one_side(interpreter, task):
    trace = interpreter.run([{"op": "BRANCH", "cond": "insufficient",
                              "then": [{"op": "REFLECT"}],
                              "else": [{"op": "ABSTAIN"}]}], task)
    taken = [step.op for step in trace.steps]
    assert ("REFLECT" in taken) != ("ABSTAIN" in taken)


def test_an_unknown_condition_is_false_not_an_error(interpreter, task):
    trace = interpreter.run([{"op": "BRANCH", "cond": "the vibes are off",
                              "then": [{"op": "ABSTAIN"}]}], task)
    assert trace.steps[0].result is False and not trace.abstained


def test_a_unanimous_vote_over_a_deterministic_step_agrees(interpreter):
    task = bench.build_family("grid_planning", 1)[0]
    trace = interpreter.run([{"op": "VOTE", "n": 3, "agg": "unanimous",
                              "body": [{"op": "SOLVE"}]}], task)
    assert trace.answer == task.expected


def test_a_vote_with_no_budget_left_for_its_body_fails_cleanly(task):
    interpreter = Interpreter(max_steps=1)
    trace = interpreter.run([{"op": "VOTE", "n": 3, "body": [{"op": "SOLVE"}]}],
                            task)
    assert trace.step_count >= 1


# ── abstention ───────────────────────────────────────────────────────

def test_abstaining_ends_the_strategy(interpreter, task):
    trace = interpreter.run([{"op": "ABSTAIN", "reason": "no"},
                             {"op": "SOLVE"}], task)
    assert trace.abstained and trace.answer is None
    assert [step.op for step in trace.steps] == ["ABSTAIN"]


def test_abstaining_is_the_right_answer_when_the_data_is_missing(interpreter):
    task = bench.build_family("missing_data", 1)[0]
    trace = interpreter.run([{"op": "ABSTAIN"}], task)
    assert trace.solved is True


def test_abstaining_is_the_wrong_answer_when_the_data_is_there(interpreter):
    task = bench.build_family("grid_planning", 1)[0]
    trace = interpreter.run([{"op": "ABSTAIN"}], task)
    assert trace.solved is False


# ── the trace ────────────────────────────────────────────────────────

def test_two_runs_of_one_strategy_produce_the_same_trace(interpreter, task):
    """Without this nothing in the reasoning contour can be compared across
    runs, and every measured gain is unfalsifiable (§3.1)."""
    strategy = [{"op": "DECOMPOSE", "max_parts": 6}, {"op": "SOLVE"},
                {"op": "VERIFY", "checker": "confidence"},
                {"op": "BRANCH", "cond": "insufficient",
                 "then": [{"op": "ABSTAIN"}]}]
    first = interpreter.run(strategy, task).as_dict()
    second = interpreter.run(strategy, task).as_dict()
    for record in (first, second):
        record.pop("elapsed_ms")
        for step in record["steps"]:
            step.pop("elapsed_ms")
    assert first == second


def test_a_trace_serialises_to_data(interpreter, task):
    """Traces are persisted and shipped. An object in one is a trace that
    cannot be written down."""
    import json

    trace = interpreter.run([{"op": "DECOMPOSE"}, {"op": "SOLVE"}], task)
    assert json.loads(json.dumps(trace.as_dict()))["strategy"] == "anonymous"


def test_a_trace_records_the_price_of_what_it_ran(interpreter, task):
    trace = interpreter.run([{"op": "LLM_STEP", "template": "x"}], task)
    assert trace.cost.llm_tokens > 0


def test_a_trace_names_the_strategy_and_the_task():
    class Named:
        name = "mine"
        steps = [{"op": "REFLECT"}]

    task = bench.build(3)
    trace = Interpreter().run(Named(), task)
    assert trace.strategy == "mine" and trace.task_id == task.id


def test_an_empty_trace_is_still_a_trace():
    assert Trace().as_dict()["steps"] == []
