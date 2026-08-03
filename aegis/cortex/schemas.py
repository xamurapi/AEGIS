"""Structured output: schemas, extraction, validation and repair (spec M8.5).

The old path parsed model output with ``json.loads`` and hoped. A response that
was *almost* right — a number sent as a string, a list where an object was
asked for, a missing key — either crashed the phase that consumed it or, worse,
flowed into memory as data. This module makes that impossible: nothing reaches
the core that has not matched a declared shape.

The pipeline is: extract JSON from the text → validate against a schema →
coerce what is safely coercible → on failure, hand the error back to the model
for exactly one repair attempt → on second failure, return None and let the
caller take its deterministic path.

The validator is deliberately small and dependency-free. It implements the
subset of JSON Schema the core actually declares (types, required, properties,
items, enum, ranges, additionalProperties) and nothing else — a fuller
implementation would be more code to trust for no benefit here.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("aegis.cortex.schemas")

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class SchemaError(Exception):
    """A payload did not match its declared schema."""


# ── extraction ───────────────────────────────────────────────────────

def extract_json(text: str) -> object | None:
    """Pull a JSON value out of model prose.

    Three strategies, in order of reliability: a fenced block, the whole text,
    then the outermost brace/bracket span. The last one is what rescues the
    common "Sure! Here is the JSON: {...} Let me know if..." response, which is
    otherwise a total loss.
    """
    # Only real response text is parsed. Accepting anything with a ``__str__``
    # meant a stray ``0`` argument rendered as "0", which is valid JSON — a
    # caller's bug would have become a plausible-looking model answer.
    if not isinstance(text, str):
        return None
    raw = text
    if not raw.strip():
        return None

    for candidate in _fenced_blocks(raw) + [raw.strip()] + _brace_spans(raw):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _fenced_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FENCE.finditer(text) if m.group(1).strip()]


def _brace_spans(text: str) -> list[str]:
    """Outermost ``{...}`` and ``[...]`` spans, longest first.

    Longest first is the point: a reply containing both an incidental list and
    the real object ("here are the options [1,2] and my answer {...}") would
    otherwise hand back the list, which parses perfectly and means nothing.
    """
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            spans.append(text[start:end + 1])
    spans.sort(key=len, reverse=True)
    return spans


# ── coercion ─────────────────────────────────────────────────────────

def coerce_number(value, default: float | None = None) -> float | None:
    """A float if the value can honestly be read as one, else ``default``.

    Booleans are refused: ``True`` is not the number 1 in any schema the core
    declares, and letting it through would make a confidence of ``true`` score
    as 1.0.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def coerce_int(value, default: int | None = None) -> int | None:
    number = coerce_number(value, None)
    if number is None:
        return default
    try:
        return int(number)
    except (OverflowError, ValueError):
        return default


# ── validation ───────────────────────────────────────────────────────

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Every way ``value`` fails ``schema``. Empty list means it matches.

    All errors are collected rather than raising on the first, because the list
    is fed straight back to the model as the repair instruction — telling it
    about one problem at a time would need one round-trip per problem.
    """
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(t, lambda _v: True)(value) for t in types):
            got = type(value).__name__
            return [f"{path}: expected {'/'.join(types)}, got {got}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    # Booleans are excluded deliberately: Python makes `True < 5` true, so a
    # range check would silently accept a boolean wherever a number belongs.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: {value} is below the minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: {value} is above the maximum {maximum}")

    if isinstance(value, str):
        min_len, max_len = schema.get("minLength"), schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"{path}: shorter than {min_len} characters")
        if max_len is not None and len(value) > max_len:
            errors.append(f"{path}: longer than {max_len} characters")

    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in value:
                errors.extend(validate(value[key], sub_schema, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in sorted(value):
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key!r}")

    if isinstance(value, list):
        min_items, max_items = schema.get("minItems"), schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: needs at least {min_items} items")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path}: allows at most {max_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate(item, item_schema, f"{path}[{index}]"))

    return errors


def coerce_to_schema(value, schema: dict):
    """Repair the type mistakes that are safe to repair, in place of failing.

    Only lossless, unambiguous conversions: a numeric string to a number, a
    number to a string when a string was asked for, a single value to a
    one-element array. Anything requiring a guess about intent is left alone so
    that :func:`validate` rejects it and the model is asked again.
    """
    if not isinstance(schema, dict):
        return value

    expected = schema.get("type")
    types = expected if isinstance(expected, list) else ([expected] if expected else [])

    if isinstance(value, dict) and "object" in types:
        properties = schema.get("properties", {})
        return {k: (coerce_to_schema(v, properties[k]) if k in properties else v)
                for k, v in value.items()}

    if "array" in types:
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        if not isinstance(value, list):
            # A model asked for a list of one thing often sends the thing.
            value = [] if value is None else [value]
        return [coerce_to_schema(v, item_schema) if item_schema else v for v in value]

    if types and "number" in types:
        number = coerce_number(value)
        if number is not None:
            clamped = number
            if schema.get("minimum") is not None:
                clamped = max(schema["minimum"], clamped)
            if schema.get("maximum") is not None:
                clamped = min(schema["maximum"], clamped)
            return clamped

    if types and "integer" in types:
        number = coerce_int(value)
        if number is not None:
            if schema.get("minimum") is not None:
                number = max(int(schema["minimum"]), number)
            if schema.get("maximum") is not None:
                number = min(int(schema["maximum"]), number)
            return number

    if types and "string" in types and isinstance(value, (int, float)) \
            and not isinstance(value, bool):
        return str(value)

    return value


def parse_and_validate(text: str, schema: dict) -> tuple[object | None, list[str]]:
    """Extract, coerce and validate in one step.

    Returns ``(payload, errors)``. A non-empty error list always comes with a
    ``None`` payload — there is no such thing as partially accepted data here.
    """
    payload = extract_json(text)
    if payload is None:
        return None, ["response contained no JSON value"]
    coerced = coerce_to_schema(payload, schema)
    errors = validate(coerced, schema)
    return (None, errors) if errors else (coerced, [])


def repair_instruction(errors: list[str], schema_name: str) -> str:
    """The follow-up message asking the model to fix its own output."""
    listed = "\n".join(f"- {e}" for e in errors[:12])
    return (
        f"Your previous response did not match the required '{schema_name}' "
        f"schema. Problems:\n{listed}\n\n"
        f"Reply with ONLY the corrected JSON object, no prose and no code fence."
    )


# ── the declared shapes (spec M8.5) ──────────────────────────────────

SCHEMAS: dict[str, dict] = {
    "answer": {
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {},                       # any JSON type: tasks vary
            "reasoning": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    "sufficiency": {
        "type": "object",
        "required": ["sufficient"],
        "properties": {
            "sufficient": {"type": "boolean"},
            "missing": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
    },
    "state_eval": {
        "type": "object",
        "required": ["assessment"],
        "properties": {
            "assessment": {"type": "string"},
            "strengths": {"type": "array", "items": {"type": "string"}},
            "weaknesses": {"type": "array", "items": {"type": "string"}},
            "suggested_goals": {"type": "array", "items": {"type": "string"}},
            "insight": {"type": "string"},
        },
    },
    "decision": {
        "type": "object",
        "required": ["chosen"],
        "properties": {
            "chosen": {"type": "integer", "minimum": 1},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "ethical_concerns": {"type": "string"},
        },
    },
    "reflection": {
        "type": "object",
        "required": ["learning"],
        "properties": {
            "learning": {"type": "string"},
            "pattern": {"type": "string"},
            "knowledge": {
                "type": "object",
                "properties": {"concept": {"type": "string"},
                               "definition": {"type": "string"}},
            },
            "self_assessment": {"type": "string"},
            "next_priority": {"type": "string"},
        },
    },
    "curiosity": {
        "type": "object",
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string", "maxLength": 200},
            "question": {"type": "string"},
            "expected_insight": {"type": "string"},
            "connection": {"type": "string"},
        },
    },
    "skill_code": {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "minLength": 1},
            "explanation": {"type": "string"},
        },
    },
    # M11.5.2 step 6: the cortex narrates an already-computed attribution and
    # names the mechanism it *believes* — a guess the code compares against the
    # computed one, never adopts. The narrative is stored for the human.
    "meta_explanation": {
        "type": "object",
        "required": ["narrative"],
        "properties": {
            "narrative": {"type": "string"},
            "mechanism": {"type": "string"},
        },
    },
    "code_change": {
        "type": "object",
        "required": ["should_modify"],
        "properties": {
            "should_modify": {"type": "boolean"},
            "reason": {"type": "string"},
            "description": {"type": "string"},
            "modified_code": {"type": "string"},
        },
    },
    "param_adjust": {
        "type": "object",
        "required": ["adjustments"],
        "properties": {
            "adjustments": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": ["parameter", "direction"],
                    "properties": {
                        "parameter": {"type": "string"},
                        "direction": {"type": "string",
                                      "enum": ["increase", "decrease"]},
                        "magnitude": {"type": "number",
                                      "minimum": 0.0, "maximum": 0.1},
                        "reason": {"type": "string"},
                    },
                },
            },
            "assessment": {"type": "string"},
        },
    },
    "plan_rerank": {
        "type": "object",
        "required": ["order"],
        "properties": {
            # Indices into the shortlist the router was given. The planner
            # re-checks them; the model cannot introduce an action this way.
            "order": {"type": "array", "maxItems": 3,
                      "items": {"type": "integer", "minimum": 0}},
            "rationale": {"type": "string"},
        },
    },
    "reasoning_strategy": {
        "type": "object",
        "required": ["name", "steps"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 60},
            "steps": {"type": "array", "minItems": 1, "maxItems": 24,
                      "items": {"type": "object",
                                "required": ["op"],
                                "properties": {"op": {"type": "string"}}}},
            "applies_to": {"type": "object"},
            "rationale": {"type": "string"},
        },
    },
    "hypothesis": {
        "type": "object",
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array", "maxItems": 10,
                "items": {
                    "type": "object",
                    "required": ["target", "predictors"],
                    "properties": {
                        "statement": {"type": "string"},
                        "target": {"type": "string"},
                        "predictors": {"type": "array", "minItems": 1,
                                       "items": {"type": "string"}},
                        "lags": {"type": "object"},
                        "kind": {"type": "string",
                                 "enum": ["association", "causal", "law"]},
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    },
    "genome_proposal": {
        "type": "object",
        "required": ["genome"],
        "properties": {
            "genome": {"type": "object"},
            "rationale": {"type": "string"},
        },
    },
    "chain_refine": {
        "type": "object",
        "required": ["objective"],
        "properties": {
            "objective": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array"},
            "plan": {"type": "array"},
            "expected_result": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}


def schema_for(name: str) -> dict:
    """Look up a declared schema; unknown names are a programming error."""
    try:
        return SCHEMAS[name]
    except KeyError:
        raise SchemaError(
            f"unknown schema {name!r}; declared: {sorted(SCHEMAS)}") from None


def schema_names() -> list[str]:
    return sorted(SCHEMAS)
