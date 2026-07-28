"""The strategy grammar (spec M6.3, M6.11).

The DSL is the security boundary of the reasoning contour: strategies are
synthesised, and one of the synthesisers is a language model. Everything here
is about admission — what gets in, what is refused, and whether the refusal
happens before anything runs rather than halfway through a strategy.
"""
import pytest

from aegis.layers.reasoning.dsl import (
    AGGREGATORS, CHECKERS, DSLError, MAX_LOOP_ITERATIONS, OPS, RETRIEVE_SOURCES,
    cost_of, count_steps, digest, normalise, validate,
)


# ── the vocabulary is closed ─────────────────────────────────────────

def test_the_vocabulary_is_exactly_the_twelve_operations():
    """M6.3 names twelve. A thirteenth would be one the interpreter cannot run."""
    assert set(OPS) == {
        "DECOMPOSE", "RETRIEVE", "PREDICT", "SOLVE", "COMPUTE", "LLM_STEP",
        "VERIFY", "VOTE", "BRANCH", "LOOP", "REFLECT", "ABSTAIN"}


def test_an_unknown_operation_is_refused():
    problems = validate([{"op": "EXEC", "code": "import os"}])
    assert problems and "unknown operation" in problems[0]


def test_every_operation_has_an_interpreter_handler():
    """Admission and execution must agree.

    An operation the grammar accepts but the interpreter cannot perform would
    be refused at run time — inside a strategy, where nobody is watching —
    instead of at the door.
    """
    from aegis.layers.reasoning.interpreter import Interpreter

    for name in OPS:
        assert hasattr(Interpreter, f"_op_{name.lower()}"), name


def test_a_field_the_operation_does_not_have_is_refused():
    """Otherwise a typo silently does nothing and the strategy looks fine."""
    problems = validate([{"op": "SOLVE", "kinds": "python"}])
    assert any("has no field" in problem for problem in problems)


def test_a_missing_required_field_is_refused():
    problems = validate([{"op": "RETRIEVE"}])
    assert any("needs 'source'" in problem for problem in problems)


def test_retrieve_sources_are_a_closed_set():
    assert validate([{"op": "RETRIEVE", "source": "graph"}]) == []
    assert validate([{"op": "RETRIEVE", "source": "the internet"}])
    assert set(RETRIEVE_SOURCES) == {"memory", "graph", "skills"}


def test_vote_aggregators_are_a_closed_set():
    for aggregator in AGGREGATORS:
        assert validate([{"op": "VOTE", "agg": aggregator,
                          "body": [{"op": "SOLVE"}]}]) == []
    assert validate([{"op": "VOTE", "agg": "whichever", "body": [{"op": "SOLVE"}]}])


def test_verify_cannot_name_a_checker_that_does_not_exist():
    """The grader is not in the set, and that is the point.

    A strategy able to ask "is this the right answer?" could answer anything by
    guessing until the grader agreed.
    """
    assert "grader" not in CHECKERS
    assert validate([{"op": "VERIFY", "checker": "oracle"}])
    for checker in CHECKERS:
        assert validate([{"op": "VERIFY", "checker": checker}]) == []


# ── the limits are structural ────────────────────────────────────────

def test_a_strategy_over_the_step_budget_is_refused():
    problems = validate([{"op": "SOLVE"}] * 4, max_steps=3)
    assert any("exceeds the budget" in problem for problem in problems)


def test_nested_steps_count_toward_the_budget():
    """A three-step strategy whose loop body is twenty is not a three-step
    strategy, and budgeting only the top level would budget nothing."""
    steps = [{"op": "LOOP", "max_iter": 4, "body": [{"op": "SOLVE"}] * 3}]
    assert count_steps(steps) == 1 + 4 * 3


def test_vote_multiplies_its_body_too():
    steps = [{"op": "VOTE", "n": 3, "body": [{"op": "SOLVE"}, {"op": "VERIFY"}]}]
    assert count_steps(steps) == 1 + 3 * 2


def test_a_gene_reference_counts_at_its_ceiling():
    """A budget computed from an optimistic guess would admit a strategy that
    then overran it."""
    steps = [{"op": "LOOP", "max_iter": "$gene:reason_budget",
              "body": [{"op": "SOLVE"}]}]
    assert count_steps(steps) == 1 + MAX_LOOP_ITERATIONS


def test_a_loop_cannot_ask_for_more_iterations_than_the_ceiling():
    assert validate([{"op": "LOOP", "max_iter": MAX_LOOP_ITERATIONS,
                      "body": [{"op": "SOLVE"}]}]) == []
    assert validate([{"op": "LOOP", "max_iter": MAX_LOOP_ITERATIONS + 1,
                      "body": [{"op": "SOLVE"}]}])


def test_a_loop_iteration_count_that_is_not_a_number_is_refused():
    assert validate([{"op": "LOOP", "max_iter": "lots",
                      "body": [{"op": "SOLVE"}]}])


# ── shape ────────────────────────────────────────────────────────────

def test_a_strategy_is_a_list_of_objects():
    assert validate("SOLVE") == ["a strategy is a list of steps"]
    assert validate([]) == ["a strategy with no steps does nothing"]
    assert any("a step is an object" in problem
               for problem in validate([["SOLVE"]]))


def test_every_problem_is_reported_not_just_the_first():
    """A synthesiser handed one error at a time fixes one error at a time."""
    problems = validate([{"op": "NOPE"}, {"op": "RETRIEVE"},
                         {"op": "VOTE", "agg": "?", "body": []}])
    assert len(problems) >= 3


def test_a_problem_inside_a_body_is_reported_with_its_path():
    problems = validate([{"op": "BRANCH", "cond": "insufficient",
                          "then": [{"op": "MAGIC"}]}])
    assert any("[0].then[0]" in problem for problem in problems)


# ── cost ─────────────────────────────────────────────────────────────

def test_cost_is_known_before_the_strategy_runs():
    cost = cost_of([{"op": "LLM_STEP", "template": "x"}])
    assert cost.llm_tokens > 0 and cost.llm_calls == 1


def test_a_loop_is_priced_for_every_iteration_it_may_take():
    once = cost_of([{"op": "LLM_STEP", "template": "x"}])
    looped = cost_of([{"op": "LOOP", "max_iter": 4,
                       "body": [{"op": "LLM_STEP", "template": "x"}]}])
    assert looped.llm_tokens >= 4 * once.llm_tokens


def test_a_vote_is_priced_for_every_round_it_will_run():
    once = cost_of([{"op": "LLM_STEP", "template": "x"}])
    voted = cost_of([{"op": "VOTE", "n": 3,
                      "body": [{"op": "LLM_STEP", "template": "x"}]}])
    assert voted.llm_tokens >= 3 * once.llm_tokens


def test_a_free_strategy_costs_nothing():
    assert cost_of([{"op": "REFLECT"}]).llm_tokens == 0


def test_a_gene_reference_is_priced_at_the_ceiling_it_could_reach():
    """A cost computed from an optimistic guess would let a strategy that
    cannot be paid for start anyway."""
    fixed = cost_of([{"op": "LOOP", "max_iter": MAX_LOOP_ITERATIONS,
                      "body": [{"op": "LLM_STEP", "template": "x"}]}])
    gene = cost_of([{"op": "LOOP", "max_iter": "$gene:reason_budget",
                     "body": [{"op": "LLM_STEP", "template": "x"}]}])
    assert gene.llm_tokens == fixed.llm_tokens


# ── the spec table itself ────────────────────────────────────────────

def test_an_operation_specification_cannot_be_edited_at_run_time():
    """The vocabulary is a constant. A contour able to widen an operation's
    allowed fields could widen them for a synthesised strategy too."""
    with pytest.raises(Exception):
        OPS["SOLVE"].optional = ("anything",)


def test_a_gene_reference_satisfies_a_bounded_integer_field():
    """It is resolved at run time and clamped there; refusing it at admission
    would make every gene-driven strategy inadmissible."""
    assert validate([{"op": "LOOP", "max_iter": "$gene:reason_budget",
                      "body": [{"op": "SOLVE"}]}]) == []


def test_a_nested_body_that_is_not_a_list_is_reported_at_its_own_path():
    problems = validate([{"op": "VOTE", "body": "SOLVE"}])
    assert problems == ["[0].body: expected a list of steps"]


# ── identity ─────────────────────────────────────────────────────────

def test_two_spellings_of_one_strategy_have_one_digest():
    """Without this a generation proposes the same strategy four times with the
    fields in different orders and evaluates all four."""
    first = [{"op": "RETRIEVE", "source": "memory", "k": 3}]
    second = [{"k": 3, "source": "memory", "op": "RETRIEVE"}]
    assert digest(first) == digest(second)


def test_different_strategies_have_different_digests():
    assert digest([{"op": "SOLVE"}]) != digest([{"op": "REFLECT"}])


def test_normalising_drops_unknown_steps_and_keeps_order():
    steps = normalise([{"op": "SOLVE"}, {"op": "NOPE"}, {"op": "REFLECT"}])
    assert [step["op"] for step in steps] == ["SOLVE", "REFLECT"]


def test_normalising_recurses_into_bodies():
    steps = normalise([{"op": "BRANCH", "cond": "insufficient",
                        "then": [{"k": 1, "source": "memory", "op": "RETRIEVE"}]}])
    assert list(steps[0]["then"][0]) == ["op", "k", "source"]


def test_dsl_error_is_a_value_error():
    """Callers that catch ValueError should catch this, because a refused
    strategy is a bad value and nothing more dramatic."""
    assert issubclass(DSLError, ValueError)
    with pytest.raises(ValueError):
        raise DSLError("refused")
