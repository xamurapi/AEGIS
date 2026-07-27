"""Structured output: extraction, validation, coercion, repair (spec M8.5).

The rule under test is absolute: nothing that failed its schema reaches the
core. "Almost right" data is refused, because the deterministic fallback is
correct while half-parsed data merely looks correct.
"""
import asyncio

import pytest

from aegis.cortex import schemas as S
from aegis.cortex.router import Cortex, Role
from tests.cortex_fakes import ScriptedProvider


def _run(coro):
    return asyncio.run(coro)


def _cortex(responses):
    provider = ScriptedProvider("a", responses=responses)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]},
                    cache=_NoCache())
    return cortex, provider


class _NoCache:
    def get(self, key):
        return None

    def put(self, key, entry):
        return None

    def hit_rate(self):
        return 0.0

    def save(self):
        return None

    def status(self):
        return {}


# ── extraction ───────────────────────────────────────────────────────

def test_plain_json_is_extracted():
    assert S.extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_is_extracted():
    assert S.extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_unlabelled_fence_is_extracted():
    assert S.extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_buried_in_prose_is_recovered():
    text = 'Sure! Here you go: {"a": 1} — let me know if you need more.'
    assert S.extract_json(text) == {"a": 1}


def test_a_bare_array_is_extracted():
    assert S.extract_json("[1, 2, 3]") == [1, 2, 3]


def test_empty_and_junk_input_yields_none():
    assert S.extract_json("") is None
    assert S.extract_json("   ") is None
    assert S.extract_json("no json here at all") is None
    assert S.extract_json(None) is None


def test_the_longest_brace_span_wins_over_a_fragment():
    assert S.extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}


def test_an_incidental_list_does_not_beat_the_real_object():
    # Both spans parse; picking the shorter one would hand back a perfectly
    # valid list that means nothing.
    text = 'The options were [1, 2] and my answer is {"answer": 42}'
    assert S.extract_json(text) == {"answer": 42}


def test_a_non_string_input_is_not_parsed_as_its_repr():
    # str(0) is "0", which is valid JSON — reading a stray integer argument as
    # a model response would be worse than refusing it.
    assert S.extract_json(0) is None
    assert S.extract_json(False) is None


# ── validation ───────────────────────────────────────────────────────

def test_a_matching_payload_has_no_errors():
    schema = {"type": "object", "required": ["a"],
              "properties": {"a": {"type": "number"}}}
    assert S.validate({"a": 1.5}, schema) == []


def test_a_missing_required_key_is_reported():
    schema = {"type": "object", "required": ["a"]}
    assert S.validate({}, schema) == ["$: missing required key 'a'"]


def test_a_wrong_type_is_reported():
    errors = S.validate("text", {"type": "object"})
    assert errors and "expected object" in errors[0]


def test_all_errors_are_collected_not_just_the_first():
    # The list is fed straight back to the model as the repair instruction, so
    # reporting one problem per round-trip would cost one round-trip each.
    schema = {"type": "object", "required": ["a", "b", "c"]}
    assert len(S.validate({}, schema)) == 3


def test_a_boolean_does_not_satisfy_a_number():
    assert S.validate(True, {"type": "number"}) != []


def test_range_bounds_do_not_apply_to_booleans():
    # Python evaluates `True < 5` happily; without an explicit exclusion a
    # boolean would pass a numeric range check and be treated as 1.
    assert S.validate(True, {"minimum": 5}) == []
    assert S.validate(False, {"maximum": -1}) == []


def test_range_bounds_are_enforced():
    schema = {"type": "number", "minimum": 0, "maximum": 1}
    assert S.validate(1.5, schema) != []
    assert S.validate(-0.5, schema) != []
    assert S.validate(0.5, schema) == []


def test_enum_membership_is_enforced():
    assert S.validate("sideways", {"enum": ["up", "down"]}) != []


def test_nested_properties_are_validated():
    schema = {"type": "object",
              "properties": {"inner": {"type": "object",
                                       "required": ["x"]}}}
    errors = S.validate({"inner": {}}, schema)
    assert errors == ["$.inner: missing required key 'x'"]


def test_array_items_are_validated_with_their_index():
    schema = {"type": "array", "items": {"type": "number"}}
    errors = S.validate([1, "two"], schema)
    assert errors and "[1]" in errors[0]


def test_array_length_bounds_are_enforced():
    assert S.validate([], {"type": "array", "minItems": 1}) != []
    assert S.validate([1, 2], {"type": "array", "maxItems": 1}) != []


def test_string_length_bounds_are_enforced():
    assert S.validate("", {"type": "string", "minLength": 1}) != []
    assert S.validate("abcd", {"type": "string", "maxLength": 2}) != []


def test_additional_properties_can_be_forbidden():
    schema = {"type": "object", "properties": {"a": {}},
              "additionalProperties": False}
    assert S.validate({"a": 1, "b": 2}, schema) != []


def test_a_union_type_accepts_either():
    schema = {"type": ["string", "number"]}
    assert S.validate("x", schema) == []
    assert S.validate(3, schema) == []
    assert S.validate([], schema) != []


def test_a_schemaless_field_accepts_anything():
    # `answer` is deliberately untyped: tasks return strings, numbers and bools.
    schema = {"type": "object", "properties": {"answer": {}}}
    for value in ("text", 1, True, [1], {"k": "v"}, None):
        assert S.validate({"answer": value}, schema) == []


# ── coercion ─────────────────────────────────────────────────────────

def test_a_numeric_string_is_coerced_to_a_number():
    schema = {"type": "object", "properties": {"c": {"type": "number"}}}
    assert S.coerce_to_schema({"c": "0.5"}, schema) == {"c": 0.5}


def test_a_number_is_coerced_to_a_string_when_asked():
    schema = {"type": "object", "properties": {"s": {"type": "string"}}}
    assert S.coerce_to_schema({"s": 7}, schema) == {"s": "7"}


def test_a_lone_value_is_coerced_into_a_one_element_array():
    schema = {"type": "object",
              "properties": {"xs": {"type": "array", "items": {"type": "string"}}}}
    assert S.coerce_to_schema({"xs": "solo"}, schema) == {"xs": ["solo"]}


def test_an_out_of_range_number_is_clamped_not_rejected():
    schema = {"type": "object",
              "properties": {"c": {"type": "number", "minimum": 0, "maximum": 1}}}
    assert S.coerce_to_schema({"c": 4.0}, schema) == {"c": 1.0}
    assert S.coerce_to_schema({"c": -4.0}, schema) == {"c": 0.0}


def test_an_integer_field_is_coerced_and_clamped():
    schema = {"type": "object",
              "properties": {"n": {"type": "integer", "minimum": 1}}}
    assert S.coerce_to_schema({"n": "0"}, schema) == {"n": 1}


def test_a_boolean_is_never_coerced_into_a_number():
    # A confidence of `true` scoring as 1.0 is exactly the silent nonsense this
    # whole layer exists to prevent.
    schema = {"type": "object", "properties": {"c": {"type": "number"}}}
    coerced = S.coerce_to_schema({"c": True}, schema)
    assert S.validate(coerced, schema) != []


def test_unrelated_keys_survive_coercion():
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    assert S.coerce_to_schema({"a": "1", "extra": "kept"}, schema) \
        == {"a": 1.0, "extra": "kept"}


def test_coerce_number_and_int_helpers():
    assert S.coerce_number("2.5") == 2.5
    assert S.coerce_number("nope", 0.0) == 0.0
    assert S.coerce_number(True, None) is None
    assert S.coerce_int("3") == 3
    assert S.coerce_int("x", 7) == 7


def test_coerce_to_schema_passes_through_a_non_schema():
    assert S.coerce_to_schema({"a": 1}, "not a schema") == {"a": 1}


# ── parse_and_validate ───────────────────────────────────────────────

def test_parse_and_validate_returns_the_payload_on_success():
    payload, errors = S.parse_and_validate('{"answer": 42}', S.schema_for("answer"))
    assert payload == {"answer": 42} and errors == []


def test_parse_and_validate_never_returns_partial_data():
    payload, errors = S.parse_and_validate('{"nope": 1}', S.schema_for("answer"))
    assert payload is None and errors


def test_parse_and_validate_reports_missing_json():
    payload, errors = S.parse_and_validate("just prose", S.schema_for("answer"))
    assert payload is None
    assert errors == ["response contained no JSON value"]


# ── the declared catalogue ───────────────────────────────────────────

def test_every_schema_named_by_the_spec_is_declared():
    required = {"state_eval", "decision", "reflection", "curiosity", "skill_code",
                "code_change", "param_adjust", "plan_rerank", "reasoning_strategy",
                "hypothesis", "genome_proposal", "chain_refine"}
    assert required <= set(S.schema_names())


def test_an_unknown_schema_name_is_a_loud_error():
    with pytest.raises(S.SchemaError):
        S.schema_for("no_such_schema")


def test_every_declared_schema_is_a_well_formed_object():
    for name in S.schema_names():
        schema = S.schema_for(name)
        assert schema.get("type") == "object", name


def test_repair_instruction_names_the_schema_and_the_problems():
    text = S.repair_instruction(["$: missing required key 'a'"], "decision")
    assert "decision" in text
    assert "missing required key" in text


# ── repair round-trip through the router ─────────────────────────────

def test_a_valid_answer_needs_no_repair():
    cortex, _ = _cortex(['{"answer": 5}'])
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) == {"answer": 5}
    assert cortex.repairs == 0


def test_a_malformed_answer_triggers_exactly_one_repair():
    cortex, provider = _cortex(["not json at all", '{"answer": 5}'])
    result = _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                    "answer"))
    assert result == {"answer": 5}
    assert cortex.repairs == 1
    assert cortex.repairs_succeeded == 1
    assert len(provider.invocations) == 2


def test_the_repair_prompt_carries_the_errors_back():
    cortex, provider = _cortex(['{"wrong": 1}', '{"answer": 5}'])
    _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}], "answer"))
    repair_prompt = provider.invocations[1][-1]["content"]
    assert "missing required key" in repair_prompt


def test_two_failures_in_a_row_yield_none_not_bad_data():
    cortex, provider = _cortex(["garbage", "still garbage"])
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) is None
    assert len(provider.invocations) == 2      # exactly one repair, then stop


def test_schema_failures_are_counted():
    cortex, _ = _cortex(["garbage", "garbage"])
    _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}], "answer"))
    assert cortex.schema_failures == 2


def test_structured_returns_none_when_the_role_is_unavailable():
    cortex = Cortex(providers={}, routes={})
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) is None


def test_structured_returns_none_when_the_provider_fails():
    provider = ScriptedProvider("a", fail=True)
    cortex = Cortex(providers={"a": provider}, routes={"fast": ["a"]}, cache=_NoCache())
    assert _run(cortex.structured(Role.FAST, [{"role": "user", "content": "q"}],
                                  "answer")) is None
