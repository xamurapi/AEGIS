"""Writing a new strategy (spec M6.7, M6.11).

Six pure transformations over the DSL, and the property that matters for all of
them is the same: whatever they produce must be something the interpreter can
run. They are the path that works with no model attached, so a deployment
without a cortex still improves its own reasoning — and that means they cannot
be allowed to depend on one.
"""
import pytest

from aegis.layers.reasoning.dsl import MAX_LOOP_ITERATIONS, digest, validate
from aegis.layers.reasoning.library import Library
from aegis.layers.reasoning.synthesis import (
    TRANSFORMS, Candidate, Synthesiser, add_abstain, add_decompose,
    add_predict, add_verify, compute_instead_of_llm, raise_vote, traffic_share,
)
from aegis.layers.reasoning.weakness import Weakness


def _weakness(label="family=arithmetic_chain", family="arithmetic_chain"):
    return Weakness(combo=tuple(label.split(" AND ")), fail_rate=0.8,
                    base_rate=0.2, support=40, fails=32, lower=0.6,
                    excess=0.6, p_value=1e-6, rank=24.0, family=family,
                    examples=("t1", "t2"))


@pytest.fixture
def library(tmp_path):
    return Library(store_path=tmp_path / "strategies.json")


@pytest.fixture
def synthesiser():
    return Synthesiser()


# ── every transformation produces something runnable ─────────────────

@pytest.mark.parametrize("name,transform", TRANSFORMS)
def test_a_transformation_never_produces_an_inadmissible_strategy(name, transform):
    """The security position of the whole contour: a synthesised strategy is
    admitted only if the interpreter can run it, so a transformation that could
    produce something else would be relying on the refusal."""
    starts = [
        [{"op": "SOLVE"}],
        [{"op": "DECOMPOSE"}, {"op": "SOLVE"}, {"op": "VERIFY"}],
        [{"op": "LLM_STEP", "template": "x"}, {"op": "COMPUTE", "expr": "$last"}],
        [{"op": "VOTE", "n": 3, "body": [{"op": "SOLVE"}]}],
        [{"op": "PREDICT"}, {"op": "BRANCH", "cond": "insufficient",
                             "then": [{"op": "ABSTAIN"}]}],
    ]
    for steps in starts:
        result = transform(list(steps))
        if result is None:
            continue
        assert validate(result) == [], (name, result)


@pytest.mark.parametrize("name,transform", TRANSFORMS)
def test_a_transformation_that_does_not_apply_says_so(name, transform):
    """Returning the input unchanged would make the deduplicator responsible
    for noticing that nothing happened."""
    already = [{"op": "DECOMPOSE"}, {"op": "PREDICT"},
               {"op": "VOTE", "n": 5, "body": [{"op": "SOLVE"}]},
               {"op": "VERIFY", "checker": "confidence"},
               {"op": "ABSTAIN"}]
    assert transform(list(already)) is None


# ── each transformation does what it says ────────────────────────────

def test_adding_a_check_adds_a_check():
    steps = add_verify([{"op": "SOLVE"}])
    assert steps[-1] == {"op": "VERIFY", "checker": "confidence"}


def test_decomposition_goes_first():
    """Breaking a problem up after the answer is produced changes nothing, and
    a transformation that reliably produced a no-op would spend an arena run
    per round proving it."""
    steps = add_decompose([{"op": "SOLVE"}])
    assert steps[0]["op"] == "DECOMPOSE"
    assert steps[0]["max_parts"] == "$gene:reason_decompose_parts"


def test_raising_a_vote_raises_it():
    steps = raise_vote([{"op": "VOTE", "n": 1, "body": [{"op": "SOLVE"}]}])
    assert steps[0]["n"] == 3


def test_a_strategy_with_no_vote_gets_one_around_what_was_there():
    steps = raise_vote([{"op": "SOLVE"}, {"op": "VERIFY"}])
    assert steps[0]["op"] == "VOTE"
    assert [step["op"] for step in steps[0]["body"]] == ["SOLVE", "VERIFY"]


def test_a_vote_at_the_ceiling_is_left_alone():
    assert raise_vote([{"op": "VOTE", "n": 5, "body": [{"op": "SOLVE"}]}]) is None


def test_a_model_step_becomes_a_computation():
    steps = compute_instead_of_llm(
        [{"op": "LLM_STEP", "template": "x"}, {"op": "SOLVE"}])
    assert [step["op"] for step in steps] == ["COMPUTE", "SOLVE"]


def test_prediction_goes_first():
    assert add_predict([{"op": "SOLVE"}])[0]["op"] == "PREDICT"


def test_abstention_needs_something_to_branch_on():
    """A branch on confidence with nothing having measured it would never
    fire, and the transformation would be decoration."""
    steps = add_abstain([{"op": "SOLVE"}])
    assert [step["op"] for step in steps] == ["SOLVE", "VERIFY", "BRANCH"]
    assert steps[-1]["then"][0]["op"] == "ABSTAIN"


def test_abstention_reuses_a_check_that_is_already_there():
    steps = add_abstain([{"op": "SOLVE"}, {"op": "VERIFY", "checker": "type"}])
    assert len([step for step in steps if step["op"] == "VERIFY"]) == 1


def test_a_transformation_looks_inside_bodies():
    """A ``VERIFY`` buried in a branch is still a verify, and adding a second
    one would be adding a step for nothing."""
    assert add_verify([{"op": "BRANCH", "cond": "insufficient",
                        "then": [{"op": "VERIFY"}]}]) is None


def test_a_transformation_does_not_edit_what_it_was_given():
    original = [{"op": "SOLVE"}]
    add_decompose(original)
    add_abstain(original)
    assert original == [{"op": "SOLVE"}]


# ── proposing ────────────────────────────────────────────────────────

def test_proposals_are_admissible_and_distinct(synthesiser, library):
    candidates = synthesiser.propose(_weakness(), library, tick=3)
    assert candidates
    shapes = set()
    for candidate in candidates:
        assert validate(candidate.steps) == []
        assert candidate.digest not in shapes
        shapes.add(candidate.digest)
        assert candidate.created_tick == 3


def test_a_proposal_that_already_exists_is_not_proposed_again(synthesiser, library):
    """``add_abstain`` applied to ``direct`` is exactly a built-in, and
    proposing it would cost an arena run to rediscover the library."""
    candidates = synthesiser.propose(_weakness(), library)
    existing = {strategy.digest for strategy in library.strategies.values()}
    assert not ({candidate.digest for candidate in candidates} & existing)
    assert synthesiser.duplicates > 0


def test_the_parent_is_the_best_strategy_for_the_weak_class(synthesiser, library):
    """For the class, not overall. ``direct`` here is the better strategy
    everywhere else and the worse one exactly where the work is needed."""
    for _ in range(20):
        library.note_result("decompose_solve_combine", "arithmetic_chain",
                            solved=True)
        library.note_result("direct", "arithmetic_chain", solved=False)
    for _ in range(200):
        library.note_result("direct", "grid_planning", solved=True)
    assert library.best_for("").name == "direct"
    assert synthesiser.parent_for(_weakness(), library).name == \
        "decompose_solve_combine"


def test_a_weakness_spanning_classes_starts_from_the_best_overall(synthesiser,
                                                                  library):
    """Getting this wrong meant transforming ``direct`` and judging the result
    against a far better incumbent, so every candidate lost by a wide margin
    for a reason that had nothing to do with the candidate."""
    for _ in range(5):
        library.note_result("abstain_on_low_confidence", "alpha", solved=True)
        library.note_result("direct", "beta", solved=False)
    parent = synthesiser.parent_for(_weakness("incomplete", family=""), library)
    assert parent.name == "abstain_on_low_confidence"


def test_a_weakness_with_no_family_attribute_at_all_still_works(synthesiser,
                                                                library):
    """A weakness arrives from several sources (M6.6) and not all of them carry
    a class. Treating "no attribute" as an error would make the synthesiser
    refuse to work on anything the reasoning benchmark did not produce."""
    class Bare:
        label = "something is wrong"
        examples = ()

    assert synthesiser.propose(Bare(), library)


def test_a_candidate_names_what_was_done_to_what(synthesiser, library):
    candidate = synthesiser.propose(_weakness(), library)[0]
    assert candidate.parent in candidate.name
    assert candidate.transform in candidate.name


def test_the_number_of_candidates_is_bounded(library):
    synthesiser = Synthesiser(max_candidates=2)
    assert len(synthesiser.propose(_weakness(), library)) <= 2


def test_an_empty_library_proposes_nothing(synthesiser, tmp_path):
    class Empty:
        strategies = {}

        def best_for(self, family, **kwargs):
            return None

        def get(self, name):
            return None

    assert synthesiser.propose(_weakness(), Empty()) == []


def test_a_candidate_renders_as_data(synthesiser, library):
    import json

    candidate = synthesiser.propose(_weakness(), library)[0]
    assert json.loads(json.dumps(candidate.as_dict()))["name"] == candidate.name


def test_status_reports_the_transformations_it_has(synthesiser, library):
    synthesiser.propose(_weakness(), library)
    status = synthesiser.status()
    assert status["proposed"] > 0
    assert set(status["transforms"]) == {name for name, _ in TRANSFORMS}


# ── the cortex path ──────────────────────────────────────────────────

class _Cortex:
    def __init__(self, reply, available=True):
        self.reply = reply
        self.available = available
        self.prompts = []

    def role_available(self, role):
        return self.available

    async def structured(self, role, messages, schema_name):
        self.prompts.append(messages[0]["content"])
        return self.reply


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_the_cortex_path_is_optional(library, caplog):
    """Having no model is a deployment, not a fault. Reaching for one that is
    not there and catching the resulting error would put a stack trace in the
    log on every synthesis round of every offline run."""
    with caplog.at_level("ERROR", logger="aegis.reasoning"):
        assert _run(Synthesiser().propose_with_cortex(_weakness(), library)) == []
    assert caplog.records == []


def test_something_that_is_not_a_cortex_is_not_called(library, caplog):
    """Anything can be handed in here. Whether it is *a cortex* is decided by
    whether it can do the thing, not by whether it is not None."""
    class NotACortex:
        def role_available(self, role):
            return True

    with caplog.at_level("ERROR", logger="aegis.reasoning"):
        assert _run(Synthesiser(cortex=NotACortex()).propose_with_cortex(
            _weakness(), library)) == []
    assert caplog.records == []


def test_a_model_strategy_is_admitted_like_any_other(library):
    cortex = _Cortex({"steps": [{"op": "REFLECT"}, {"op": "SOLVE"},
                                {"op": "VERIFY", "checker": "consistency"}]})
    candidates = _run(Synthesiser(cortex=cortex).propose_with_cortex(
        _weakness(), library))
    assert len(candidates) == 1 and candidates[0].origin == "cortex"


def test_a_model_strategy_the_interpreter_cannot_run_is_refused(library):
    """The model is shown the grammar and can still answer outside it. The
    refusal is the boundary, not the prompt."""
    synthesiser = Synthesiser(cortex=_Cortex({"steps": [{"op": "EXEC"}]}))
    assert _run(synthesiser.propose_with_cortex(_weakness(), library)) == []
    assert synthesiser.refused == 1


def test_a_model_that_answers_with_nonsense_costs_nothing(library):
    for reply in (None, {}, {"steps": "SOLVE"}, {"steps": None}):
        synthesiser = Synthesiser(cortex=_Cortex(reply))
        assert _run(synthesiser.propose_with_cortex(_weakness(), library)) == []


def test_a_model_that_raises_does_not_take_the_call_down(library):
    class Broken:
        def role_available(self, role):
            raise RuntimeError("no")

        async def structured(self, *args, **kwargs):
            return None

    assert _run(Synthesiser(cortex=Broken()).propose_with_cortex(
        _weakness(), library)) == []


def test_an_unavailable_role_is_not_called(library):
    cortex = _Cortex({"steps": [{"op": "REFLECT"}]}, available=False)
    assert _run(Synthesiser(cortex=cortex).propose_with_cortex(
        _weakness(), library)) == []
    assert cortex.prompts == []


def test_the_prompt_carries_the_weakness_the_examples_and_the_grammar(library):
    cortex = _Cortex({"steps": [{"op": "REFLECT"}]})
    _run(Synthesiser(cortex=cortex).propose_with_cortex(_weakness(), library))
    prompt = cortex.prompts[0]
    assert "family=arithmetic_chain" in prompt and "t1" in prompt
    assert "SOLVE" in prompt and str(MAX_LOOP_ITERATIONS) in prompt
    # It must be told that no checker reveals the answer, or it will ask for one.
    assert "no checker can tell you the right answer" in prompt


def test_a_model_strategy_the_templates_already_produced_is_dropped(library):
    steps = add_abstain([{"op": "SOLVE"}])
    synthesiser = Synthesiser(cortex=_Cortex({"steps": steps}))
    twin = Candidate(name="twin", steps=steps)
    assert _run(synthesiser.propose_with_cortex(
        _weakness(), library, existing=[twin])) == []


# ── trial traffic ────────────────────────────────────────────────────

def test_a_trial_gets_roughly_its_share():
    keys = [f"task{index}" for index in range(400)]
    share = sum(1 for key in keys if traffic_share("t", key, 4)) / len(keys)
    assert 0.15 < share < 0.35


def test_the_share_is_the_same_on_every_run():
    """A counter would make the split depend on how many other strategies
    happened to run first, and two runs of one experiment would divide the
    traffic differently."""
    first = [traffic_share("t", f"k{i}", 4) for i in range(50)]
    second = [traffic_share("t", f"k{i}", 4) for i in range(50)]
    assert first == second


def test_two_trials_do_not_get_the_same_requests():
    keys = [f"k{index}" for index in range(200)]
    mine = {key for key in keys if traffic_share("a", key, 4)}
    theirs = {key for key in keys if traffic_share("b", key, 4)}
    assert mine != theirs


def test_a_share_of_one_takes_everything():
    assert all(traffic_share("t", f"k{i}", 1) for i in range(20))


def test_a_nonsense_share_does_not_divide_by_zero():
    assert traffic_share("t", "k", 0) is True


def test_a_transformation_of_a_transformation_is_still_admissible(library):
    """Candidates compound over generations, so the result of one round is the
    input of the next."""
    steps = [{"op": "SOLVE"}]
    for _, transform in TRANSFORMS:
        result = transform(list(steps))
        if result is not None:
            steps = result
    assert validate(steps) == []
    assert digest(steps)
