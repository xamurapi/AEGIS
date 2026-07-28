"""The strategy DSL: thinking expressed as data (spec M6.3).

A reasoning strategy is a declarative pipeline, **not Python**. That is the
whole security position of this contour: strategies are synthesised — some of
them by a language model — and a synthesised strategy that were Python would be
arbitrary code execution with a friendly name. Here it is a list of records
drawn from a fixed vocabulary, interpreted by an interpreter that can only do
the twelve things listed below.

Three limits are structural rather than advisory:

* **A fixed vocabulary.** An unknown operation is refused at *admission*, not
  at execution — a strategy the library accepted must be one the interpreter
  can run, or the refusal happens somewhere nobody is watching.
* **A step budget.** Total steps, counted through nesting, may not exceed
  ``REASON_MAX_STEPS``. ``LOOP`` additionally carries its own ``max_iter``, so
  a loop with an always-true condition terminates by construction rather than
  by hope.
* **A price.** Every operation declares a ``ResourceCost``, and a whole
  strategy runs under one lease. A strategy that cannot be paid for does not
  run — the same rule as every other action (M4).

``$``-references are the only indirection: ``$last`` is the previous step's
result, and ``$gene:<name>`` is a genome value read at run time, which is what
ties M5 to M6 (Appendix E).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import aegis.config as cfg
from aegis.layers.motivation.resources import ResourceCost

logger = logging.getLogger("aegis.reasoning")


@dataclass(frozen=True)
class OpSpec:
    """One operation: what it needs, what it costs, whether it nests."""

    name: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    cost: ResourceCost = field(default_factory=ResourceCost)
    #: Fields holding nested step lists, walked when counting and validating.
    bodies: tuple[str, ...] = ()


def _cost(tok=0, ms=0, proc=0) -> ResourceCost:
    return ResourceCost(llm_tokens=tok, wall_ms=ms, subprocess_slots=proc,
                        llm_calls=1 if tok else 0)


#: The vocabulary (M6.3). Adding an operation means adding it here *and* in the
#: interpreter; a strategy naming anything else is refused.
OPS: dict[str, OpSpec] = {
    "DECOMPOSE": OpSpec("DECOMPOSE", optional=("max_parts",), cost=_cost(ms=20)),
    "RETRIEVE": OpSpec("RETRIEVE", required=("source",), optional=("k",),
                       cost=_cost(ms=30)),
    "PREDICT": OpSpec("PREDICT", optional=("horizon",), cost=_cost(ms=15)),
    "SOLVE": OpSpec("SOLVE", optional=("kind",), cost=_cost(ms=300, proc=1)),
    "COMPUTE": OpSpec("COMPUTE", required=("expr",), cost=_cost(ms=50, proc=1)),
    "LLM_STEP": OpSpec("LLM_STEP", required=("template",),
                       optional=("schema", "role"), cost=_cost(tok=400, ms=2000)),
    "VERIFY": OpSpec("VERIFY", optional=("checker",), cost=_cost(ms=20)),
    "VOTE": OpSpec("VOTE", required=("body",), optional=("n", "agg"),
                   cost=_cost(ms=10), bodies=("body",)),
    "BRANCH": OpSpec("BRANCH", required=("cond",), optional=("then", "else"),
                     cost=_cost(ms=5), bodies=("then", "else")),
    "LOOP": OpSpec("LOOP", required=("body",), optional=("while", "max_iter"),
                   cost=_cost(ms=5), bodies=("body",)),
    "REFLECT": OpSpec("REFLECT", cost=_cost(ms=10)),
    "ABSTAIN": OpSpec("ABSTAIN", optional=("reason",), cost=_cost(ms=1)),
}

#: Where ``RETRIEVE`` may look. A closed set, for the same reason the operation
#: vocabulary is closed.
RETRIEVE_SOURCES = ("memory", "graph", "skills")

#: How ``VOTE`` combines its runs.
AGGREGATORS = ("majority", "first", "unanimous")

#: What ``VERIFY`` may check. Notably absent: the benchmark's grader. A strategy
#: that could consult the right answer would not be reasoning, and a closed set
#: is the only way to be sure none ever does.
CHECKERS = ("type", "confidence", "consistency", "task")

#: Hard ceiling on a LOOP, independent of what the strategy asks for. A loop
#: whose condition never goes false is the obvious failure, and the obvious
#: failure is the one worth making impossible rather than unlikely.
MAX_LOOP_ITERATIONS = 8


class DSLError(ValueError):
    """A strategy that cannot be admitted, with the reason."""


def validate(steps, *, max_steps: int | None = None) -> list[str]:
    """Every problem with a strategy. Empty means admissible.

    Returns a list rather than raising on the first fault, because a
    synthesiser handed one error at a time will fix one error at a time.
    """
    max_steps = int(cfg.REASON_MAX_STEPS if max_steps is None else max_steps)
    problems: list[str] = []
    if not isinstance(steps, list):
        return ["a strategy is a list of steps"]
    if not steps:
        return ["a strategy with no steps does nothing"]

    total = count_steps(steps)
    if total > max_steps:
        problems.append(f"{total} steps exceeds the budget of {max_steps}")
    _validate_block(steps, problems, path="")
    return problems


def _validate_block(steps, problems: list[str], path: str) -> None:
    if not isinstance(steps, list):
        problems.append(f"{path or 'strategy'}: expected a list of steps")
        return
    for index, step in enumerate(steps):
        where = f"{path}[{index}]"
        if not isinstance(step, dict):
            problems.append(f"{where}: a step is an object")
            continue
        name = step.get("op")
        spec = OPS.get(str(name))
        if spec is None:
            problems.append(f"{where}: unknown operation {name!r}")
            continue
        for field_name in spec.required:
            if field_name not in step:
                problems.append(f"{where}: {name} needs {field_name!r}")
        allowed = set(spec.required) | set(spec.optional) | {"op"}
        for field_name in step:
            if field_name not in allowed:
                problems.append(f"{where}: {name} has no field {field_name!r}")

        if name == "RETRIEVE" and str(step.get("source")) not in RETRIEVE_SOURCES:
            problems.append(f"{where}: RETRIEVE source must be one of "
                            f"{list(RETRIEVE_SOURCES)}")
        if name == "VOTE":
            aggregator = str(step.get("agg", "majority"))
            if aggregator not in AGGREGATORS:
                problems.append(f"{where}: VOTE agg must be one of "
                                f"{list(AGGREGATORS)}")
        if name == "VERIFY" and str(step.get("checker", "type")) not in CHECKERS:
            problems.append(f"{where}: VERIFY checker must be one of "
                            f"{list(CHECKERS)}")
        if name == "LOOP":
            iterations = step.get("max_iter", MAX_LOOP_ITERATIONS)
            if not _is_bounded_int(iterations, 1, MAX_LOOP_ITERATIONS):
                problems.append(f"{where}: LOOP max_iter must be 1.."
                                f"{MAX_LOOP_ITERATIONS}")

        for body_field in spec.bodies:
            body = step.get(body_field)
            if body is None:
                continue
            _validate_block(body, problems, f"{where}.{body_field}")


def _is_bounded_int(value, low: int, high: int) -> bool:
    if isinstance(value, str) and value.startswith("$gene:"):
        return True                    # resolved at run time, clamped there
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    return low <= number <= high


def count_steps(steps) -> int:
    """Total operations, counting through every nested body.

    Counting the nesting is the point: a strategy whose top level is three
    steps and whose loop body is twenty is a twenty-three step strategy, and
    budgeting only the top level would be budgeting nothing.
    """
    total = 0
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        total += 1
        spec = OPS.get(str(step.get("op")))
        if spec is None:
            continue
        for body_field in spec.bodies:
            body = step.get(body_field)
            if isinstance(body, list):
                multiplier = 1
                if step.get("op") == "LOOP":
                    multiplier = _static_int(step.get("max_iter"),
                                             MAX_LOOP_ITERATIONS)
                elif step.get("op") == "VOTE":
                    multiplier = _static_int(step.get("n"), 1)
                total += count_steps(body) * max(1, multiplier)
    return total


def _static_int(value, default: int) -> int:
    """A step count needs a number now; a ``$gene`` reference gets the ceiling.

    Taking the ceiling is the safe direction: a budget computed from an
    optimistic guess would admit a strategy that then overran it.
    """
    if isinstance(value, str) and value.startswith("$gene:"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def cost_of(steps) -> ResourceCost:
    """What running this strategy is expected to cost, before it runs."""
    total = ResourceCost()
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        spec = OPS.get(str(step.get("op")))
        if spec is None:
            continue
        total = total + spec.cost
        for body_field in spec.bodies:
            body = step.get(body_field)
            if not isinstance(body, list):
                continue
            multiplier = 1
            if step.get("op") == "LOOP":
                multiplier = _static_int(step.get("max_iter"), MAX_LOOP_ITERATIONS)
            elif step.get("op") == "VOTE":
                multiplier = _static_int(step.get("n"), 1)
            inner = cost_of(body)
            for _ in range(max(1, multiplier)):
                total = total + inner
    return total


def normalise(steps):
    """A canonical form, so two spellings of one strategy are one strategy.

    Keys sorted, missing optionals left out, nested bodies normalised. The
    digest of this is what deduplicates synthesiser output: without it a
    generation proposes the same strategy four times with the fields in
    different orders and evaluates all four.
    """
    out = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        spec = OPS.get(str(step.get("op")))
        if spec is None:
            continue
        rendered = {"op": spec.name}
        for field_name in sorted(set(spec.required) | set(spec.optional)):
            if field_name not in step:
                continue
            value = step[field_name]
            rendered[field_name] = (normalise(value) if field_name in spec.bodies
                                    else value)
        out.append(rendered)
    return out


def digest(steps) -> str:
    """Stable identity of a strategy's shape."""
    from aegis.util.canonical import digest_of

    return digest_of(normalise(steps))[:16]
