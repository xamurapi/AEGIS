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


# ── weakness, synthesis and the arena (stage 9) ──────────────────────

@when(parsers.parse("{count:d} attempts fail at random"))
def _random_failures(ctx, count):
    from aegis.util.quasirandom import hash_index

    ctx["rows"] = [
        {"task": f"t{index}", "family": ("alpha", "beta", "gamma")[index % 3],
         "solved": hash_index(3, "noise", index) != 0,
         "features": {"numeric": bool(index % 2), "steps": index % 4}}
        for index in range(count)]


@when(parsers.parse("{failing:d} attempts of one kind fail among {count:d} that "
                    "mostly do not"))
def _one_weak_kind(ctx, failing, count):
    _random_failures(ctx, count)
    ctx["rows"] += [{"task": f"w{index}", "family": "delta", "solved": False,
                     "features": {"brittle": True}} for index in range(failing)]


@then("no weakness should be reported")
def _no_weakness_reported(ctx):
    assert ctx["engine"].detector.scan(ctx["rows"]) == []


@then("that kind should be reported as a weakness")
def _weakness_reported(ctx):
    found = ctx["engine"].detector.scan(ctx["rows"])
    assert any("brittle" in weakness.combo for weakness in found)


@when(parsers.parse("{rounds:d} rounds of work and improvement are run"))
def _improvement_rounds(ctx, rounds):
    engine = ctx["engine"]
    engine.set_genome({"reason_decompose_parts": 10})
    for cycle in range(1, rounds + 1):
        engine.solve(64)
        engine.scan_weakness()
        engine.propose_strategy(tick=cycle)
        while engine.pending_candidates():
            engine.evaluate_candidate(tick=cycle)
        engine.review_trials(tick=cycle)


@then("a strategy that is not built in should exist")
def _synthesised_exists(ctx):
    assert [s for s in ctx["engine"].library.strategies.values() if not s.builtin]


@then("every synthesised strategy should be on trial or promoted")
def _on_trial_or_promoted(ctx):
    synthesised = [s for s in ctx["engine"].library.strategies.values()
                   if not s.builtin]
    assert synthesised
    assert all(s.status in ("trial", "active", "retired") for s in synthesised)


@then("no synthesised strategy should be in service without having run")
def _no_untested_promotion(ctx):
    for strategy in ctx["engine"].library.active():
        if not strategy.builtin:
            assert strategy.used() > 0, strategy.name


@when(parsers.parse('a strategy that always abstains is judged for "{family}"'))
def _judge_always_abstain(ctx, family):
    from aegis.layers.reasoning.weakness import Weakness

    engine = ctx["engine"]
    weakness = Weakness(combo=(f"family={family}",), fail_rate=0.9,
                        base_rate=0.2, support=40, fails=36, lower=0.7,
                        excess=0.7, p_value=1e-6, rank=28.0, family=family)

    class Bare:
        steps = [{"op": "ABSTAIN", "reason": "no"}]
        name = "always_abstain"

    ctx["verdict"] = engine.arena.evaluate(Bare(), weakness,
                                           engine.library.get("direct"))


@then("it should be refused for regressing the general benchmark")
def _refused_for_regression(ctx):
    verdict = ctx["verdict"]
    assert not verdict.accepted
    assert any("general benchmark" in reason for reason in verdict.reasons)


# ── the external anchor ──────────────────────────────────────────────

def _golden_tasks():
    from aegis.eval import reasoning_bench as bench
    from tests.test_reasoning_bench import GOLDEN_REASONING
    for task_id, prompt, answer in GOLDEN_REASONING:
        family = ("arithmetic_chain" if "arith" in task_id
                  else "constraint_puzzle")
        index = int(task_id.rsplit("_", 1)[1])
        task = bench.build_family(family, index + 1, start=index)[0]
        assert task.id == task_id and task.prompt == prompt
        yield task, answer


@then("every hand-solved golden task should match its generated answer")
def _golden_matches(ctx):
    count = 0
    for task, answer in _golden_tasks():
        assert task.expected == answer, task.id
        count += 1
    assert count >= 5


@then("every hand-solved golden task should reject a wrong answer")
def _golden_rejects(ctx):
    from aegis.eval.reasoning_bench import ABSTAIN
    for task, answer in _golden_tasks():
        wrong = 48 if answer == ABSTAIN else \
            answer - 1 if isinstance(answer, int) else "crate"
        assert not task.verify(wrong), task.id
