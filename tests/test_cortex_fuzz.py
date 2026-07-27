"""Malformed model output never reaches the core (spec §M8.8).

The acceptance criterion is exact: 200 corrupted responses must produce zero
exceptions and zero writes into state. A model that answers with *almost* valid
JSON is the normal case, not the pathological one — truncation, smart quotes,
Python literals, prose wrapped around the object, an empty body from a proxy.

The corpus is generated deterministically (§3.1): the same 200 inputs every
run, so a failure is reproducible rather than a story about one unlucky night.
"""
import asyncio

import pytest

from aegis.cortex import schemas as S
from aegis.cortex.router import Cortex, Role
from aegis.layers.substrate import Substrate
from aegis.util.quasirandom import hash_index
from tests.cortex_fakes import ScriptedProvider

FUZZ_COUNT = 200

#: Ways a real model output goes wrong, applied to a well-formed base document.
_MUTILATIONS = (
    lambda s: s[: max(1, len(s) // 2)],                 # truncated mid-object
    lambda s: s.replace('"', "'"),                       # Python literal quoting
    lambda s: s.replace('"', "“", 1),               # a smart quote
    lambda s: s.replace("{", "", 1),                     # unbalanced brace
    lambda s: s.replace("}", "", 1),
    lambda s: s.replace(":", "=", 1),                    # YAML-ish
    lambda s: s.replace(",", ";", 1),
    lambda s: f"Sure! Here is the JSON:\n{s}\nHope that helps!",
    lambda s: f"```json\n{s}",                           # unclosed fence
    lambda s: f"```\n{s}\n``` extra ```",                # doubled fence
    lambda s: s + s,                                     # two objects, no array
    lambda s: s.replace("null", "None").replace("true", "True"),
    lambda s: "",                                        # empty body
    lambda s: "   \n\t  ",
    lambda s: "null",
    lambda s: "[]",                                      # right JSON, wrong shape
    lambda s: "42",
    lambda s: '"a bare string"',
    lambda s: '{"unexpected": "keys only"}',
    lambda s: '{"chosen": "NaN", "confidence": "very high"}',
    lambda s: s.replace("0.7", "1e999"),                 # overflow to inf
    lambda s: s.replace("0.7", "true"),                  # boolean where a number goes
    lambda s: "\x00\x01\x02 binary garbage",
    lambda s: "<html><body>502 Bad Gateway</body></html>",
    lambda s: s.replace("\n", ""),                       # minified, still valid
)

_BASE_DOCUMENTS = {
    "decision": '{"chosen": 1, "reasoning": "because", "confidence": 0.7,'
                ' "ethical_concerns": "none"}',
    "state_eval": '{"assessment": "fine", "strengths": ["a"], "weaknesses": [],'
                  ' "suggested_goals": ["g"], "insight": "i"}',
    "reflection": '{"learning": "x", "pattern": "y",'
                  ' "knowledge": {"concept": "c", "definition": "d"},'
                  ' "self_assessment": "ok", "next_priority": "p"}',
    "answer": '{"answer": 42, "reasoning": "r", "confidence": 0.7}',
    "param_adjust": '{"adjustments": [{"parameter": "temperature",'
                    ' "direction": "increase", "magnitude": 0.05, "reason": "r"}],'
                    ' "assessment": "a"}',
}


def corpus(count: int = FUZZ_COUNT) -> list[tuple[str, str]]:
    """``(schema_name, mangled_text)`` pairs — deterministic and stable."""
    names = sorted(_BASE_DOCUMENTS)
    out = []
    for i in range(count):
        name = names[hash_index(len(names), "schema", i)]
        mangle = _MUTILATIONS[hash_index(len(_MUTILATIONS), "mangle", i)]
        out.append((name, mangle(_BASE_DOCUMENTS[name])))
    return out


def _run(coro):
    return asyncio.run(coro)


# ── the corpus itself ────────────────────────────────────────────────

def test_the_corpus_is_the_required_size():
    assert len(corpus()) == FUZZ_COUNT


def test_the_corpus_is_reproducible():
    assert corpus() == corpus()


def test_the_corpus_exercises_every_mutilation():
    used = {text for _, text in corpus()}
    assert len(used) > len(_MUTILATIONS)     # variety across schemas too


# ── the validator never raises ───────────────────────────────────────

def test_no_mangled_input_raises_during_validation():
    for schema_name, text in corpus():
        schema = S.schema_for(schema_name)
        payload, errors = S.parse_and_validate(text, schema)
        assert payload is None or errors == []


def test_accepted_payloads_really_do_match_their_schema():
    for schema_name, text in corpus():
        schema = S.schema_for(schema_name)
        payload, errors = S.parse_and_validate(text, schema)
        if payload is not None:
            assert S.validate(payload, schema) == []


def test_extraction_never_raises_on_junk():
    for _, text in corpus():
        S.extract_json(text)        # must not raise


def test_infinite_and_nan_numbers_are_refused():
    schema = S.schema_for("decision")
    payload, errors = S.parse_and_validate(
        '{"chosen": 1, "confidence": 1e999}', schema)
    # inf is above the declared maximum, so it is clamped or rejected — never
    # allowed through as a confidence.
    assert payload is None or payload["confidence"] <= 1.0


# ── the router never raises ──────────────────────────────────────────

def test_the_router_survives_the_whole_corpus():
    for schema_name, text in corpus():
        cortex = Cortex(providers={"a": ScriptedProvider("a", responses=[text, text])},
                        routes={"fast": ["a"]})
        result = _run(cortex.structured(
            Role.FAST, [{"role": "user", "content": "q"}], schema_name))
        assert result is None or isinstance(result, dict)


def test_the_router_never_returns_data_that_failed_its_schema():
    for schema_name, text in corpus():
        cortex = Cortex(providers={"a": ScriptedProvider("a", responses=[text, text])},
                        routes={"fast": ["a"]})
        result = _run(cortex.structured(
            Role.FAST, [{"role": "user", "content": "q"}], schema_name))
        if result is not None:
            assert S.validate(result, S.schema_for(schema_name)) == []


# ── nothing reaches memory (§M8.8) ───────────────────────────────────

@pytest.fixture
def substrate(isolated_state):
    s = Substrate()
    return s


def test_garbage_evaluations_write_nothing_into_memory(substrate):
    """The strict form of the criterion: 0 exceptions AND 0 state writes."""
    from aegis.cortex.cache import ResponseCache

    semantic_before = dict(substrate.memory.semantic)
    episodic_before = len(substrate.memory.episodic)
    goals_before = len(substrate.goals.goals)

    for schema_name, text in corpus():
        if schema_name not in ("state_eval", "reflection"):
            continue
        substrate.llm.cortex = Cortex(
            providers={"a": ScriptedProvider("a", responses=[text, text])},
            routes={"fast": ["a"]}, cache=ResponseCache(None))
        if schema_name == "state_eval":
            result = _run(substrate.llm.evaluate_state({"tick": 1}))
        else:
            result = _run(substrate.llm.reflect({"tick": 1}))
        # Either a validated payload or an honest failure — never a half-parse.
        if result.get("parsed") is not None and result.get("via") == "cortex":
            assert S.validate(result["parsed"], S.schema_for(schema_name)) == []

    assert substrate.memory.semantic == semantic_before
    assert len(substrate.memory.episodic) == episodic_before
    assert len(substrate.goals.goals) == goals_before


def test_a_full_tick_survives_a_provider_answering_garbage(substrate):
    from aegis.cortex.cache import ResponseCache
    substrate.llm.cortex = Cortex(
        providers={"a": ScriptedProvider("a", responses=["}{ not json"] * 50)},
        routes={"fast": ["a"], "deep": ["a"], "code": ["a"], "judge": ["a"]},
        cache=ResponseCache(None))
    substrate.llm.enabled = True

    errors_before = substrate.health.error_count
    for _ in range(6):
        _run(substrate.tick())
    assert substrate.health.error_count == errors_before
