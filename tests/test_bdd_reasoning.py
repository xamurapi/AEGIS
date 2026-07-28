"""pytest-bdd step definitions for tests/features/reasoning.feature.

Executable Gherkin: every scenario drives the real engine, interpreter and
library, so the feature file is both the description of how the reasoning
contour behaves and the test that it behaves that way.
"""
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from aegis.eval import reasoning_bench as bench
from aegis.layers.reasoning import ReasoningEngine
from aegis.layers.reasoning.dsl import DSLError, validate
from aegis.layers.reasoning.interpreter import UNAVAILABLE, Interpreter
from aegis.layers.reasoning.library import BUILTIN_STRATEGIES

scenarios("features/reasoning.feature")


@given("a reasoning engine with the built-in strategies", target_fixture="ctx")
def _engine(tmp_path):
    return {"engine": ReasoningEngine(store_path=tmp_path / "strategies.json")}


# ── admission ────────────────────────────────────────────────────────

@when(parsers.parse('a strategy using the operation "{operation}" is offered'))
def _offer_unknown(ctx, operation):
    try:
        ctx["engine"].library.admit("intruder", [{"op": operation}])
        ctx["refused"] = False
    except DSLError:
        ctx["refused"] = True
    ctx["name"] = "intruder"


@when(parsers.parse('a strategy identical in shape to "{existing}" is offered'))
def _offer_duplicate(ctx, existing):
    steps = ctx["engine"].library.get(existing).steps
    try:
        ctx["engine"].library.admit("copy", list(steps))
        ctx["refused"] = False
    except DSLError:
        ctx["refused"] = True
    ctx["name"] = "copy"


@then("it should be refused")
def _was_refused(ctx):
    assert ctx["refused"] is True


@then("the library should not contain it")
def _not_present(ctx):
    assert ctx["name"] not in ctx["engine"].library.strategies


@then("every built-in strategy should pass validation")
def _builtins_validate(ctx):
    for name, steps in BUILTIN_STRATEGIES.items():
        assert validate(steps) == [], name


# ── the interpreter's limits ─────────────────────────────────────────

@when(parsers.parse("a strategy loops forever on {budget:d} steps of budget"))
def _endless_loop(ctx, budget):
    task = bench.build_family("arithmetic_chain", 1)[0]
    ctx["trace"] = ctx["engine"].interpreter.run(
        [{"op": "LOOP", "while": "insufficient", "max_iter": 8,
          "body": [{"op": "REFLECT"}]}], task, budget=budget)


@then(parsers.parse("it should stop within {limit:d} steps"))
def _stopped(ctx, limit):
    assert ctx["trace"].step_count <= limit


@when(parsers.parse("the step budget is asked to be {asked:d} against a maximum "
                    "of {maximum:d}"))
def _oversized_budget(ctx, asked, maximum):
    interpreter = Interpreter(max_steps=maximum)
    ctx["trace"] = interpreter.run([{"op": "REFLECT"}] * 30,
                                   bench.build(0), budget=asked)


@then(parsers.parse("no more than {limit:d} steps should run"))
def _within_maximum(ctx, limit):
    assert len([step for step in ctx["trace"].steps
                if step.op == "REFLECT"]) <= limit


@when(parsers.parse('a strategy computes "{expression}" with no sandbox'))
def _compute_without_sandbox(ctx, expression):
    ctx["trace"] = Interpreter().run([{"op": "COMPUTE", "expr": expression}],
                                     bench.build(0))


@then("nothing should have been evaluated")
def _nothing_evaluated(ctx):
    assert ctx["trace"].answer == UNAVAILABLE


# ── verification ─────────────────────────────────────────────────────

@when(parsers.parse('a strategy verifies its answer with the checker "{checker}"'))
def _verify_with(ctx, checker):
    ctx["trace"] = ctx["engine"].interpreter.run(
        [{"op": "SOLVE"}, {"op": "VERIFY", "checker": checker}],
        bench.build_family("missing_data", 1)[0])


@then("the verification should report that the task carries no check")
def _no_check(ctx):
    step = ctx["trace"].steps[1]
    assert step.result is None and "no solver-facing check" in step.note


@when(parsers.parse('a task with missing data is worked by "{strategy}"'))
def _work_missing_data(ctx, strategy):
    engine = ctx["engine"]
    ctx["row"] = engine.attempt(bench.build_family("missing_data", 1)[0],
                                engine.library.get(strategy))


@then("the engine should abstain")
def _abstained(ctx):
    assert ctx["row"]["abstained"] is True


@then("the answer should count as correct")
def _correct(ctx):
    assert ctx["row"]["solved"] is True


@then("the engine should record a confident error")
def _confident_error(ctx):
    assert ctx["row"]["confident_error"] is True
    assert ctx["engine"].confident_errors == 1


# ── selection and weakness ───────────────────────────────────────────

@when(parsers.parse("{count:d} problems from one family are worked"))
def _work_family(ctx, count):
    engine = ctx["engine"]
    ctx["chosen"] = [engine.attempt(task)["strategy"]
                     for task in bench.build_family("grid_planning", count)]


@then("more than one strategy should have been tried on it")
def _spread(ctx):
    assert len(set(ctx["chosen"])) > 1


@when(parsers.parse('{count:d} problems with missing data are worked by "{strategy}"'))
def _work_missing(ctx, count, strategy):
    engine = ctx["engine"]
    chosen = engine.library.get(strategy)
    for task in bench.build_family("missing_data", count):
        engine.attempt(task, chosen)


@then(parsers.parse('the top weakness should be "{family}"'))
def _weakness_is(ctx, family):
    weakness = ctx["engine"].top_weakness()
    assert weakness is not None and weakness["family"] == family


@then("there should be no weakness")
def _no_weakness(ctx):
    assert ctx["engine"].top_weakness() is None


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    caplog.set_level("ERROR", logger="aegis.reasoning")
