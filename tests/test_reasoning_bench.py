"""The reasoning benchmark and the reference reasoner (spec M6.5, M6.11).

The benchmark is generators, not a list: a task is built deterministically from
its index, which gives an unbounded held-out set with no RNG and no leak. What
has to hold is that indices do not collide, that a task built twice is the same
task, and that every verifier is programmatic — a benchmark that needed a
judgement call would be measuring the judge.
"""
import pytest

from aegis.eval import reasoning_bench as bench
from aegis.eval.reasoning_bench import ABSTAIN
from aegis.layers.reasoning.reasoner import REASONER, Answer, DeterministicReasoner


def _case(family: str, *, incomplete: bool):
    """The first task of a family with or without a stated gap.

    Picked by feature rather than by index: a test that happened to land on a
    complete chain would pass for a reason nobody wrote down, and would start
    failing the day the generator's mix changed.
    """
    for task in bench.build_family(family, 60):
        if bool(task.features.get("incomplete")) is incomplete:
            return task
    raise AssertionError(f"no {family} case with incomplete={incomplete}")


# ── the generators ───────────────────────────────────────────────────

def test_consecutive_indices_spread_across_the_families():
    """Walking one family for the first N tasks would make any short run a
    measurement of one family."""
    families = [bench.build(index).family for index in range(len(bench.FAMILIES))]
    assert set(families) == set(bench.FAMILIES)


def test_all_eight_families_of_the_spec_are_generated():
    assert set(bench.FAMILIES) == {
        "arithmetic_chain", "unit_conversion", "constraint_puzzle",
        "grid_planning", "rule_chain", "contradiction", "magnitude",
        "missing_data"}


def test_a_task_built_twice_is_the_same_task():
    for index in (0, 7, 41, 998):
        first, second = bench.build(index), bench.build(index)
        assert (first.id, first.prompt, first.expected) == \
               (second.id, second.prompt, second.expected)


def test_a_thousand_indices_give_a_thousand_distinct_tasks():
    ids = {bench.build(index).id for index in range(1000)}
    assert len(ids) == 1000


def test_every_task_carries_a_programmatic_verifier():
    for index in range(64):
        task = bench.build(index)
        assert task.verify(task.expected)


def test_every_task_carries_the_features_a_weakness_is_described_along():
    for index in range(64):
        features = bench.build(index).features
        assert "steps" in features and "numeric" in features


def test_the_reference_answer_is_the_expected_answer():
    for index in range(32):
        task = bench.build(index)
        assert bench.reference_answer(task) == task.expected


def test_the_splits_are_disjoint():
    train, holdout = bench.split(64)
    assert train and holdout
    assert not ({task.id for task in train} & {task.id for task in holdout})


def test_features_can_be_counted_across_a_set():
    counts = bench.features_of(bench.benchmark(32))
    assert counts and all(isinstance(value, int) for value in counts.values())


def test_the_feature_counts_are_counts():
    """The weakness detector ranks by volume. Counts that drifted would rank
    the wrong class as the one worth working on."""
    tasks = bench.build_family("grid_planning", 5)
    counts = bench.features_of(tasks)
    assert counts["family=grid_planning"] == 5
    assert counts["op:compute"] == 5
    assert counts["numeric"] == 5
    assert counts["steps=2"] == 5


def test_a_block_of_tasks_starts_where_it_was_asked_to():
    """Held-out sets are addressed by index. A block that walked the other way
    would silently overlap the training range."""
    assert [task.id for task in bench.build_family("grid_planning", 3, start=10)] \
        == ["reason_grid_10", "reason_grid_11", "reason_grid_12"]
    assert [task.id for task in bench.benchmark(3, start=16)] \
        == [bench.build(16).id, bench.build(17).id, bench.build(18).id]


def test_a_split_accounts_for_every_task_it_was_given():
    train, holdout = bench.split(40, holdout=0.25)
    assert len(train) == 30 and len(holdout) == 10


def test_a_generated_number_covers_its_whole_range_and_no_more():
    """The chain length drives how many parts a decomposition needs, so a range
    that quietly shifted would move the benchmark's difficulty without anything
    saying so."""
    lengths = {bench.build_family("arithmetic_chain", 1, start=index)[0]
               .features["steps"] for index in range(200)}
    assert lengths == {3, 4, 5, 6}


def test_a_conversion_is_scaled_by_both_units():
    task = next(item for item in bench.build_family("unit_conversion", 200)
                if " km long" in item.prompt and "How many m " in item.prompt)
    amount = int(task.prompt.split(" km long")[0].split()[-1])
    assert task.expected == amount * 1000


def test_building_an_unknown_family_is_refused():
    with pytest.raises(KeyError):
        bench.build_family("telepathy", 1)


# ── the families say what they mean ──────────────────────────────────

def test_a_conversion_never_asks_for_the_unit_it_was_given():
    """"How many m is 9 m" is not a conversion, and a family whose cases are
    sometimes trivial measures something else on those."""
    for index in range(120):
        task = bench.build_family("unit_conversion", 1, start=index)[0]
        words = task.prompt.split()
        source = words[words.index("long.") - 1]
        target = task.prompt.split("How many ")[1].split()[0]
        assert source != target


def test_the_hardest_class_needs_two_techniques_at_once(): 
    """``arithmetic_chain`` mixes chains that must be broken up with chains
    that cannot be answered at all. A family where every case yielded to one
    technique would let selection alone reach the ceiling, leaving nothing for
    synthesis to find (M6.7)."""
    tasks = bench.build_family("arithmetic_chain", 60)
    incomplete = [task for task in tasks if task.features["incomplete"]]
    assert 0.15 < len(incomplete) / len(tasks) < 0.55
    assert all(task.expected == ABSTAIN for task in incomplete)


def test_a_broken_rule_chain_expects_an_abstention():
    tasks = bench.build_family("rule_chain", 60)
    broken = [task for task in tasks if task.features["incomplete"]]
    assert broken and all(task.expected == ABSTAIN for task in broken)


def test_missing_data_expects_an_abstention():
    for index in range(16):
        assert bench.build_family("missing_data", 1, start=index)[0].expected == ABSTAIN


def test_a_task_with_missing_data_is_not_answerable():
    assert not bench.build_family("missing_data", 1)[0].answerable
    assert bench.build_family("grid_planning", 1)[0].answerable


def test_contradiction_cases_are_not_all_of_one_kind():
    """Half consistent, half not — otherwise "always yes" scores 100%."""
    answers = {bench.build_family("contradiction", 1, start=index)[0].expected
               for index in range(40)}
    assert answers == {True, False}


def test_silence_counts_as_an_abstention_and_a_wrong_one_when_answerable():
    assert bench.build_family("missing_data", 1)[0].verify(None)
    assert not bench.build_family("grid_planning", 1)[0].verify(None)


def test_a_number_answered_as_a_float_is_still_right():
    """A reasoner that answers 7.0 has not made a mistake."""
    task = bench.build_family("grid_planning", 1)[0]
    assert task.verify(float(task.expected))
    assert task.verify(str(task.expected))


def test_a_hedge_is_not_an_answer():
    task = bench.build_family("contradiction", 1)[0]
    assert not task.verify("probably")


def test_one_is_not_yes():
    """Python says ``1 == True``; a grader that agreed would score a count as
    a correct yes/no answer."""
    task = next(item for item in bench.build_family("contradiction", 20)
                if item.expected is True)
    assert task.verify(True) and not task.verify(1)


def test_a_number_that_is_merely_close_is_not_the_answer():
    task = bench.build_family("grid_planning", 1)[0]
    assert not task.verify(task.expected + 1)


def test_an_answer_that_cannot_even_be_compared_is_wrong_not_an_error():
    """A grader called from inside a tick must never raise."""
    class Unprintable:
        def __str__(self):
            raise RuntimeError("no")

    assert bench.build_family("constraint_puzzle", 1)[0].verify(Unprintable()) is False


def test_a_task_cannot_be_edited_after_it_is_built():
    """Two runs meet the same problems only if a problem is the same object
    every time it is asked for."""
    task = bench.build(0)
    with pytest.raises(Exception):
        task.expected = 0


def test_each_family_declares_exactly_what_it_exercises():
    """These labels are the axes the weakness detector groups along (M6.6).

    Compared whole rather than key by key: a label that quietly went missing
    would make its axis invisible, and "no weakness found along an axis nobody
    is labelled with" is indistinguishable from "no weakness".
    """
    expected = {
        "arithmetic_chain": {"steps": 5, "numeric": True, "ops": ["compute"],
                             "incomplete": False},
        "unit_conversion": {"steps": 2, "numeric": True, "units": True,
                            "ops": ["compute"]},
        "constraint_puzzle": {"steps": 3, "numeric": False, "constraints": 2,
                              "ops": ["decompose", "verify"]},
        "grid_planning": {"steps": 2, "numeric": True, "planning": True,
                          "ops": ["compute"]},
        "rule_chain": {"steps": 4, "numeric": False, "logic": True,
                       "ops": ["decompose"], "incomplete": False},
        "contradiction": {"steps": 2, "numeric": True, "logic": True,
                          "ops": ["verify"]},
        "magnitude": {"steps": 2, "numeric": True, "estimation": True,
                      "ops": ["compute"]},
        "missing_data": {"steps": 1, "numeric": True, "incomplete": True,
                         "ops": ["abstain"]},
    }
    for family, features in expected.items():
        assert bench.build_family(family, 1)[0].features == features, family


def test_a_whole_rule_chain_expects_a_yes():
    """The complement of the broken case: if the answer to an intact chain were
    anything else, "abstain on a broken one" would be untestable."""
    tasks = bench.build_family("rule_chain", 60)
    whole = [task for task in tasks if not task.features["incomplete"]]
    assert whole and all(task.expected is True for task in whole)


def test_a_rule_chain_is_a_chain_and_not_a_single_hop():
    """Each rule must hand off to the next. Rules that all pointed straight at
    the goal would be labelled as multi-step and answerable in one, which makes
    the family's difficulty a fiction."""
    import re

    for index in range(60):
        task = bench.build_family("rule_chain", 1, start=index)[0]
        if task.features["incomplete"]:
            continue
        rules = re.findall(r"If (\w+) then (\w+)", task.prompt)
        fact = re.search(r"(\w+) is true\.", task.prompt).group(1)
        goal = re.search(r"Is (\w+) true\?", task.prompt).group(1)
        assert len(rules) >= 2
        assert [premise for premise, _ in rules] == [fact] + \
            [conclusion for _, conclusion in rules[:-1]]
        assert rules[-1][1] == goal


def test_the_step_count_of_a_rule_chain_follows_its_length():
    lengths = {bench.build_family("rule_chain", 1, start=index)[0]
               .features["steps"] for index in range(200)}
    assert lengths == {3, 4, 5}


# ── the reference reasoner ───────────────────────────────────────────

def test_the_reasoner_never_sees_the_expected_answer():
    """It is handed the task object, so the guarantee has to be behavioural:
    change the answer and the reasoning does not move."""
    import dataclasses

    task = bench.build_family("grid_planning", 1)[0]
    honest = REASONER.solve(task)
    tampered = REASONER.solve(dataclasses.replace(task, expected=-999))
    assert honest == tampered


def test_a_parsed_answer_is_confident_and_a_guessed_one_is_not():
    reasoned = REASONER.solve(bench.build_family("grid_planning", 1)[0])
    guessed = REASONER.solve(bench.build_family("missing_data", 1)[0])
    assert reasoned.confident and reasoned.method == "grid"
    assert not guessed.confident and guessed.method == "guess"


def test_the_guess_is_the_failure_abstention_exists_to_prevent():
    """On a question about a quantity nobody stated it produces a confident-
    sounding wrong number."""
    task = bench.build_family("missing_data", 1)[0]
    answer = REASONER.solve(task)
    assert answer.value is not None and not task.verify(answer.value)


def test_a_chain_cannot_be_read_out_of_an_undivided_blob():
    """This is why DECOMPOSE earns its cost, and it is a property of the
    reasoner rather than an arrangement of the benchmark."""
    assert REASONER.solve(_case("arithmetic_chain", incomplete=False)).method == "guess"


def test_a_decomposed_chain_is_solved():
    from aegis.layers.reasoning.interpreter import _clauses

    task = _case("arithmetic_chain", incomplete=False)
    answer = REASONER.solve(task, clauses=_clauses(task.prompt))
    assert answer.confident and task.verify(answer.value)


def test_a_chain_cut_short_is_answered_without_confidence():
    from aegis.layers.reasoning.interpreter import _clauses

    task = _case("arithmetic_chain", incomplete=False)
    answer = REASONER.solve(task, clauses=_clauses(task.prompt)[:2])
    assert not answer.confident


def test_an_unstated_operand_is_a_hole_not_a_clause_to_skip():
    """Skipping it silently is how a reasoner produces an answer to a problem
    it was never given enough to solve."""
    from aegis.layers.reasoning.interpreter import _clauses

    task = _case("arithmetic_chain", incomplete=True)
    answer = REASONER.solve(task, clauses=_clauses(task.prompt))
    assert answer.method == "chain" and not answer.confident


def test_failing_to_derive_a_goal_is_not_a_confident_no():
    """From implications that do not reach the goal, nothing follows about the
    goal. A closed-world "false" here is a confident wrong answer."""
    whole = REASONER.solve(_case("rule_chain", incomplete=False))
    broken = REASONER.solve(_case("rule_chain", incomplete=True))
    assert whole.value is True and whole.confident
    assert broken.confident is False


def test_the_blob_parsers_solve_their_families_without_decomposition():
    for family in ("unit_conversion", "grid_planning", "magnitude",
                   "contradiction", "constraint_puzzle"):
        task = bench.build_family(family, 1)[0]
        answer = REASONER.solve(task)
        assert answer.confident and task.verify(answer.value), family
    whole = _case("rule_chain", incomplete=False)
    assert REASONER.solve(whole).confident


def test_a_parser_that_trips_abstains_rather_than_raising():
    class Exploding:
        prompt = property(lambda self: 1 / 0)

    assert DeterministicReasoner().solve(Exploding()).value is None


def test_a_prompt_with_no_numbers_leaves_the_reasoner_with_nothing():
    class Bare:
        prompt = "Consider the matter carefully."

    answer = DeterministicReasoner().solve(Bare())
    assert answer.value is None and answer.method == "none"


def test_a_whole_number_comes_back_as_an_integer():
    class Counted:
        prompt = "There were 12 of them. How many were there really?"

    assert DeterministicReasoner().solve(Counted()).value == 12


def test_an_answer_defaults_to_confident():
    assert Answer(1).confident is True


def test_an_answer_cannot_be_edited_after_it_is_given():
    """A caller able to raise an answer's confidence could turn a guess into a
    reasoned answer, and abstention branches on exactly that flag."""
    with pytest.raises(Exception):
        Answer(1, confident=False).confident = True


def test_a_prompt_that_cannot_be_read_yields_no_answer_and_no_confidence():
    class Exploding:
        prompt = property(lambda self: 1 / 0)

    answer = DeterministicReasoner().solve(Exploding())
    assert answer.value is None and answer.confident is False


def test_a_prompt_with_no_numbers_is_not_a_confident_silence():
    class Bare:
        prompt = "Consider the matter carefully."

    assert DeterministicReasoner().solve(Bare()).confident is False


def test_a_conversion_scales_by_the_source_and_divides_by_the_target():
    """Both halves of the ratio. A source unit of one metre hides a mistake in
    the first half, so the case that catches it has to use neither unit as the
    base."""
    class Rope:
        prompt = "A rope is 3 km long. How many m is that?"

    class Cord:
        prompt = "A cord is 3 m long. How many cm is that?"

    assert DeterministicReasoner().solve(Rope()).value == 3000
    assert DeterministicReasoner().solve(Cord()).value == 300


def test_a_chain_of_rules_with_no_stated_fact_is_not_answered():
    """Implications alone conclude nothing. Answering from them would be
    inventing the premise."""
    class NoFact:
        prompt = "If A then B. If B then C. Is C true?"

    assert DeterministicReasoner().solve(NoFact()).method != "implication"


def test_rules_are_followed_to_a_fixed_point_not_in_one_pass():
    """The rules are given back to front, so one sweep derives only the last
    link and concludes nothing. A chain is only followed by sweeping until
    nothing new appears."""
    class Backwards:
        prompt = ("If C then D. If B then C. If A then B. A is true. "
                  "Is D true?")

    answer = DeterministicReasoner().solve(Backwards())
    assert answer.value is True and answer.confident


def test_a_puzzle_missing_either_half_is_declined_rather_than_attempted():
    """The guard is what makes the parser return "not mine" instead of raising
    on half a puzzle; the raise would be swallowed and look identical from
    outside, so the guard is checked where it lives."""
    reasoner = DeterministicReasoner()
    setup_only = ("Ada, Bo and Cy each carry one of a crate, a barrel and a "
                  "sack. Ada does not carry the barrel or the sack.")
    question_only = "What does Cy carry?"
    assert reasoner._elimination(setup_only, []) is None
    assert reasoner._elimination(question_only, []) is None


def test_a_puzzle_with_no_question_is_not_a_puzzle():
    class Halfway:
        prompt = ("Ada, Bo and Cy each carry one of a crate, a barrel and a "
                  "sack. Ada does not carry the barrel or the sack.")

    assert DeterministicReasoner().solve(Halfway()).method != "elimination"


def test_a_puzzle_that_does_not_narrow_to_one_item_is_not_answered():
    """Two possibilities left is not an answer, and picking one would be a
    guess wearing the elimination parser's confidence."""
    class Ambiguous:
        prompt = ("Ada, Bo and Cy each carry one of a crate, a barrel and a "
                  "sack. What does Cy carry?")

    assert DeterministicReasoner().solve(Ambiguous()).method != "elimination"
