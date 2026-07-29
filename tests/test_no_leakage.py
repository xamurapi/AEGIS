"""Held-out tasks must never reach a synthesiser (spec M9.3).

A benchmark measures generalisation only for as long as the thing being measured
has not seen the answers. The leak that matters here is quiet: skill synthesis
builds a prompt out of example tasks, and if the examples come from the whole
set rather than from `train`, the model is shown the very cases its output will
be graded on. The score then goes up, the capability does not, and nothing looks
wrong.

So every outgoing prompt is intercepted and checked against the held-out tasks —
their ids, their payloads and their expected answers. This is the only test in
the suite that reads what the system *says* rather than what it does, and that
is the point: the leak lives in the text.
"""
import asyncio

import pytest

from aegis.eval.benchmark import (
    DEFAULT_BENCHMARK, split_tasks, tasks_for_kind, three_way_split,
)
from aegis.eval.generators import generated_benchmark
from aegis.layers.substrate import Substrate


class _Recorder:
    """A cortex stand-in that remembers every prompt and proposes nothing.

    Returning nothing keeps the test to the one question it is about: what was
    the model shown. A recorder that answered would drag the acceptance gate,
    the sandbox and the benchmark in with it.
    """

    def __init__(self):
        #: subject -> everything said while proposing for it. Bucketed rather
        #: than pooled because prompts are per kind, and so is the claim. The
        #: benchmark reuses payloads across kinds — "hello" is a *train* case of
        #: `reverse` and a *held-out* case of `palindrome` — so a pooled check
        #: would be permanently red for something that is not a leak: writing a
        #: reverse skill tells you nothing about a palindrome answer.
        self.seen: dict[str, list[str]] = {}
        self._subject = ""

    def _remember(self, *parts):
        bucket = self.seen.setdefault(self._subject, [])
        for part in parts:
            if isinstance(part, str):
                bucket.append(part)
            elif isinstance(part, dict):
                self._remember(*[str(key) for key in part])
                self._remember(*part.values())
            elif isinstance(part, (list, tuple)):
                self._remember(*part)
            else:
                bucket.append(str(part))

    async def propose_skill(self, kind, examples, feedback="", lease=None):
        self._subject = str(kind)
        self._remember(kind, examples, feedback)
        return None

    async def propose_coding_solution(self, func_name, spec, visible_tests,
                                      lease=None):
        self._subject = str(func_name)
        self._remember(func_name, spec, visible_tests)
        return None

    def blob_for(self, subject: str) -> str:
        return "\n".join(self.seen.get(str(subject), []))

    @property
    def blob(self) -> str:
        return "\n".join(line for lines in self.seen.values() for line in lines)


@pytest.fixture
def substrate(isolated_state):
    system = Substrate()
    system.llm.enabled = True
    # Nothing here may run the real benchmark: it is seconds of sandboxed
    # subprocess work per call, and this test is about prompts.
    system.evaluator.pass_rate_on = lambda tasks: 0.0
    system._score_holdout = lambda holdout: (0.0, None)
    return system


def _forbidden_strings(tasks) -> list[str]:
    """Every distinctive string a held-out task could be recognised by.

    Short values are skipped: a payload of ``5`` appears in half the prompts in
    the system for reasons that have nothing to do with leakage, and a test that
    flagged it would be turned off within a week.
    """
    out = []
    for task in tasks:
        out.append(task.id)
        for value in task.payload.values():
            text = str(value)
            if len(text) >= 4:
                out.append(text)
        expected = str(task.expected)
        if len(expected) >= 4:
            out.append(expected)
    return sorted(set(out))


def _synthesise(substrate, recorder, kinds):
    substrate.llm.propose_skill = recorder.propose_skill
    substrate.llm.propose_coding_solution = recorder.propose_coding_solution
    substrate.evaluator.failing_kinds = lambda: list(kinds)

    async def run():
        for _ in kinds:
            await substrate._skill_synthesis()

    asyncio.run(run())


# ── the split itself ─────────────────────────────────────────────────

def test_train_and_holdout_do_not_overlap():
    for kind in sorted({task.kind for task in DEFAULT_BENCHMARK}):
        if len(tasks_for_kind(kind)) <= 1:
            continue                       # degenerate by construction
        train, holdout = split_tasks(kind)
        assert {task.id for task in train} & {task.id for task in holdout} == set()


def test_the_three_way_split_is_a_partition():
    tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=6)
    split = three_way_split(tasks)
    ids = [task.id for group in split.values() for task in group]
    assert sorted(ids) == sorted(task.id for task in tasks)
    assert len(ids) == len(set(ids))


def test_no_task_appears_in_two_splits():
    tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=6)
    split = three_way_split(tasks)
    for left in split:
        for right in split:
            if left >= right:
                continue
            assert not ({task.id for task in split[left]}
                        & {task.id for task in split[right]})


# ── the prompts ──────────────────────────────────────────────────────

def test_skill_synthesis_is_shown_only_training_examples(substrate):
    """The whole point: what the synthesiser sees decides what the score means."""
    kinds = sorted({task.kind for task in DEFAULT_BENCHMARK
                    if len(tasks_for_kind(task.kind)) > 1})
    recorder = _Recorder()
    _synthesise(substrate, recorder, kinds)
    assert recorder.seen, "no prompt was captured — the test proves nothing"

    for kind in kinds:
        blob = recorder.blob_for(kind)
        assert blob, f"nothing was proposed for {kind!r}"
        _, holdout = split_tasks(kind)
        for needle in _forbidden_strings(holdout):
            assert needle not in blob, (
                f"held-out material for {kind!r} reached its own synthesiser: "
                f"{needle!r}")


def test_the_training_examples_do_reach_the_synthesiser(substrate):
    """The other half of the claim. A synthesiser shown nothing at all would
    pass the leakage test trivially and be useless."""
    recorder = _Recorder()
    _synthesise(substrate, recorder, ["roman"])
    train, _ = split_tasks("roman")
    blob = recorder.blob_for("roman")
    assert any(str(task.expected) in blob for task in train)


def test_the_held_out_ids_are_never_named_in_a_prompt(substrate):
    recorder = _Recorder()
    _synthesise(substrate, recorder, ["calc", "is_prime"])
    for kind in ("calc", "is_prime"):
        _, holdout = split_tasks(kind)
        for task in holdout:
            assert task.id not in recorder.blob_for(kind)


def test_a_generated_holdout_is_not_leaked_either(substrate):
    """Generators make the held-out set unbounded, which only helps if the
    synthesiser still never sees it."""
    substrate.evaluator.tasks = list(DEFAULT_BENCHMARK) + generated_benchmark(per_kind=4)
    recorder = _Recorder()
    _synthesise(substrate, recorder, ["roman"])

    held_out = [task for task in three_way_split(substrate.evaluator.tasks)["test"]
                if task.kind == "roman"]
    assert held_out, "the generated split produced no held-out roman tasks"
    blob = recorder.blob_for("roman")
    for needle in _forbidden_strings(held_out):
        assert needle not in blob


def test_a_coding_prompt_carries_only_the_visible_tests(substrate):
    """Coding tasks keep their hidden tests hidden, by the same rule."""
    from aegis.eval.coding import CODING_BENCHMARK

    recorder = _Recorder()
    substrate.llm.propose_coding_solution = recorder.propose_coding_solution

    async def run():
        await substrate._coding_synthesis(list(CODING_BENCHMARK))

    asyncio.run(run())
    assert recorder.seen

    blob = recorder.blob
    for task in CODING_BENCHMARK:
        visible = {repr(list(case)) for case in task.visible_tests}
        for case in task.hidden_tests:
            rendered = repr(list(case))
            if rendered in visible:
                continue                   # the same case is legitimately both
            assert rendered not in blob, (
                f"a hidden test for {task.id!r} reached the synthesiser")


# ── the reasoning strategy synthesiser (M6.7, M9.3) ──────────────────
#
# The same leak, one contour along. A weakness carries failing *examples*, and
# those examples are the text of problems the system attempted — pasted straight
# into the prompt asking a model for a better strategy. If the attempts a
# weakness is built from ever came from a set the arena later grades on, the
# gain the arena measures is the model having been shown the answers.
#
# The separation is by construction: the working queue walks up from zero, the
# arena's sets start at one, two and three million, and the acceptance harness
# counts down from ten million. Construction is exactly the kind of thing that
# survives until someone changes an index, so it is asserted rather than trusted.

def _reasoning_engine():
    from aegis.layers.reasoning import ReasoningEngine

    return ReasoningEngine()


def test_the_arena_never_judges_on_problems_the_queue_has_worked():
    """The queue and the three arena sets must not intersect at all."""
    from aegis.layers.reasoning import arena as arena_module

    engine = _reasoning_engine()
    engine.refill(200)
    worked = {task.id for task in engine.queue}

    from aegis.eval import reasoning_bench as bench

    for base in (arena_module.TRAIN_BASE, arena_module.HOLDOUT_BASE,
                 arena_module.REGRESSION_BASE):
        judged = {bench.build(base + offset).id for offset in range(200)}
        assert not (worked & judged), (
            f"the queue and the arena set at {base} share problems")


def test_the_synthesiser_prompt_carries_no_held_out_problem():
    """What the model is actually shown, checked against what it is graded on.

    Built the long way — work problems, scan for a weakness, render the prompt —
    because the leak would live in whatever the *engine* put in the examples,
    not in whatever a hand-made weakness carries.
    """
    from aegis.eval import reasoning_bench as bench
    from aegis.layers.reasoning import arena as arena_module
    from aegis.layers.reasoning.synthesis import Synthesiser

    engine = _reasoning_engine()
    engine.solve(240)
    weaknesses = engine.detector.scan(engine.results)
    assert weaknesses, "no weakness to build a prompt from"

    parent = engine.synthesiser.parent_for(weaknesses[0], engine.library)
    prompt = Synthesiser._prompt(weaknesses[0], parent)
    assert prompt, "the synthesiser prompt is empty"

    graded = [bench.build(arena_module.HOLDOUT_BASE + offset)
              for offset in range(240)]
    graded += [bench.build(10_000_000 - offset) for offset in range(240)]
    for task in graded:
        # An attempt record carries the task's id, so this is the check that
        # bites today: point the queue at the arena's held-out range and these
        # ids appear in the prompt verbatim.
        assert task.id not in prompt, (
            f"held-out {task.id!r} was named to the strategy synthesiser")
        # A forward guard rather than a live one — the day a record carries the
        # problem text instead of its id, the leak becomes far worse and this is
        # what catches it.
        assert str(task.prompt) not in prompt, (
            f"held-out problem {task.id!r} reached the strategy synthesiser")


def test_the_examples_that_do_reach_it_are_the_ones_it_failed():
    """The complement of the leak test: a prompt with no examples at all would
    pass every check above and tell the model nothing."""
    engine = _reasoning_engine()
    engine.solve(240)
    weaknesses = engine.detector.scan(engine.results)
    assert weaknesses and weaknesses[0].examples, "no failing examples carried"

    worked = {str(row.get("task", "")) for row in engine.results}
    assert set(weaknesses[0].examples) <= worked
