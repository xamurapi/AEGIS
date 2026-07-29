"""Writing a new strategy for a class the system is bad at (spec M6.7).

Two paths, and the order matters. The **template path** is six pure functions
over the DSL — add a check, break the problem up, ask more times, compute
instead of asking, look before leaping, know when not to answer. It needs no
model, it always runs, and it is what makes this contour work in a deployment
with no cortex at all. The **cortex path** is a model shown the weakness, the
failing examples, the current best strategy and the grammar, and asked for a
new strategy in schema.

Both produce candidates and neither is trusted. A candidate is a *proposal*:
the library refuses anything the interpreter could not run, the arena refuses
anything that does not actually help, and only then does it get traffic.

Candidates are deduplicated by the normalised digest of their steps, so a
transformation that produced something the library already has, or two
transformations that converged, cost one evaluation rather than two.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aegis.layers.reasoning.dsl import (
    MAX_LOOP_ITERATIONS, digest, normalise, validate,
)
from aegis.util.quasirandom import hash_index

logger = logging.getLogger("aegis.reasoning")

#: How many candidates one round proposes. Every one costs an arena run, and an
#: arena run is the expensive part of this contour.
MAX_CANDIDATES = 6

#: Confidence threshold the ``add_abstain`` transformation branches on.
ABSTAIN_CONDITION = "insufficient"


@dataclass
class Candidate:
    """A proposed strategy, and where it came from."""

    name: str
    steps: list
    parent: str = ""
    weakness: str = ""
    origin: str = "synth"
    transform: str = ""
    created_tick: int = 0
    #: Set by the arena.
    verdict: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return digest(self.steps)

    def as_dict(self) -> dict:
        return {"name": self.name, "steps": normalise(self.steps),
                "parent": self.parent, "weakness": self.weakness,
                "origin": self.origin, "transform": self.transform,
                "created_tick": self.created_tick, "verdict": dict(self.verdict)}


# ── the six transformations (M6.7) ───────────────────────────────────
#
# Each is a pure function from a step list to a step list, or to ``None`` when
# it does not apply. Returning ``None`` rather than the input unchanged is what
# keeps the deduplicator from having to notice that nothing happened.

def _has(steps, operation: str) -> bool:
    """Whether an operation appears anywhere, including inside a body."""
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        if step.get("op") == operation:
            return True
        for value in step.values():
            if isinstance(value, list) and _has(value, operation):
                return True
    return False


def add_verify(steps):
    """Check the answer before standing behind it."""
    if _has(steps, "VERIFY"):
        return None
    return list(steps) + [{"op": "VERIFY", "checker": "confidence"}]


def add_decompose(steps):
    """Break the problem up before working it.

    Prepended, not appended: decomposition that happens after the answer has
    been produced changes nothing, and a transformation that reliably produced
    a no-op would spend an arena run per round proving it.
    """
    if _has(steps, "DECOMPOSE"):
        return None
    return [{"op": "DECOMPOSE", "max_parts": "$gene:reason_decompose_parts"}] \
        + list(steps)


def raise_vote(steps):
    """Ask more times and take the majority."""
    out, changed = [], False
    for step in steps:
        if isinstance(step, dict) and step.get("op") == "VOTE":
            current = step.get("n")
            current = current if isinstance(current, int) else 1
            if current < 5:
                step = {**step, "n": current + 2}
                changed = True
        out.append(step)
    if changed:
        return out
    if _has(steps, "VOTE"):
        return None                     # already at the ceiling
    # No vote at all: wrap the whole strategy in one. The body is what was
    # there, so the transformation stays a transformation rather than a rewrite.
    return [{"op": "VOTE", "n": 3, "agg": "majority", "body": list(steps)}]


def compute_instead_of_llm(steps):
    """Replace a model step with a sandboxed computation.

    The point is cost and determinism, not capability: a step that can be
    computed should not be a model call, and the arena's cost gate is what
    notices when it can.
    """
    if not _has(steps, "LLM_STEP"):
        return None
    out = []
    for step in steps:
        if isinstance(step, dict) and step.get("op") == "LLM_STEP":
            out.append({"op": "COMPUTE", "expr": "$last"})
            continue
        out.append(step)
    return out


def add_predict(steps):
    """Ask whether this is winnable before spending anything on it."""
    if _has(steps, "PREDICT"):
        return None
    return [{"op": "PREDICT", "horizon": 1}] + list(steps)


def add_abstain(steps):
    """Refuse when the answer would be a guess.

    The transformation that matters most on this benchmark, and the one whose
    absence is measurable: without it a strategy answers every unanswerable
    question with a confident number.
    """
    if _has(steps, "ABSTAIN"):
        return None
    tail = list(steps)
    if not _has(steps, "VERIFY"):
        tail.append({"op": "VERIFY", "checker": "confidence"})
    tail.append({"op": "BRANCH", "cond": ABSTAIN_CONDITION,
                 "then": [{"op": "ABSTAIN",
                           "reason": "the answer would be a guess"}]})
    return tail


#: In a fixed order, so two runs propose the same candidates in the same order.
TRANSFORMS = (
    ("add_abstain", add_abstain),
    ("add_decompose", add_decompose),
    ("add_verify", add_verify),
    ("add_predict", add_predict),
    ("raise_vote", raise_vote),
    ("compute_instead_of_llm", compute_instead_of_llm),
)


class Synthesiser:
    """Proposes strategies for a weakness. Refuses to guarantee any of them."""

    def __init__(self, *, cortex=None, max_candidates: int = MAX_CANDIDATES):
        self.cortex = cortex
        self.max_candidates = int(max_candidates)
        self.proposed = 0
        self.duplicates = 0
        self.refused = 0
        self.from_cortex = 0

    def parent_for(self, weakness, library):
        """The strategy a transformation starts from.

        The best strategy for the weak class — and for a weakness that spans
        classes, the best overall. Exactly the rule the arena uses to pick the
        incumbent, which is the point: transforming one strategy and then
        judging the result against a different, better one made every candidate
        lose by a wide margin for a reason that had nothing to do with the
        candidate.
        """
        family = getattr(weakness, "family", "") or ""
        return library.best_for(family) or library.get("direct")

    def propose(self, weakness, library, tick: int = 0) -> list[Candidate]:
        """The template path: candidates from six transformations.

        Synchronous and model-free on purpose. This is the path that has to
        work, so it is the path with no dependencies — a deployment with no
        cortex still improves its own reasoning.
        """
        parent = self.parent_for(weakness, library)
        if parent is None:
            return []
        label = getattr(weakness, "label", str(weakness))
        seen = {strategy.digest for strategy in library.strategies.values()}
        candidates: list[Candidate] = []
        for transform_name, transform in TRANSFORMS:
            if len(candidates) >= self.max_candidates:
                break
            try:
                steps = transform(list(parent.steps))
            except Exception:
                logger.exception("Transformation %s failed", transform_name)
                continue
            candidate = self._admit(steps, parent.name, label, transform_name,
                                    "synth", seen, tick)
            if candidate is not None:
                candidates.append(candidate)
        self.proposed += len(candidates)
        return candidates

    async def propose_with_cortex(self, weakness, library, tick: int = 0,
                                  existing=()) -> list[Candidate]:
        """The model path, on top of whatever the templates produced.

        Kept separate rather than folded into ``propose`` because it is the half
        that can be slow, absent or wrong, and the half that has to work should
        not be able to fail with it.
        """
        parent = self.parent_for(weakness, library)
        if parent is None:
            return []
        steps = await self._ask_cortex(weakness, parent)
        if steps is None:
            return []
        seen = {strategy.digest for strategy in library.strategies.values()}
        seen.update(candidate.digest for candidate in existing)
        candidate = self._admit(steps, parent.name,
                                getattr(weakness, "label", str(weakness)),
                                "cortex", "cortex", seen, tick)
        if candidate is None:
            return []
        self.from_cortex += 1
        self.proposed += 1
        return [candidate]

    # ── internals ────────────────────────────────────────────────────

    def _admit(self, steps, parent: str, label: str, transform: str,
               origin: str, seen: set, tick: int) -> Candidate | None:
        """Validate, deduplicate, name. Anything questionable is dropped here."""
        if not steps:
            return None
        problems = validate(steps)
        if problems:
            self.refused += 1
            logger.debug("Refused a synthesised strategy: %s", problems[0])
            return None
        shape = digest(steps)
        if shape in seen:
            self.duplicates += 1
            return None
        seen.add(shape)
        return Candidate(name=self._name(parent, transform, shape),
                         steps=normalise(steps), parent=parent,
                         weakness=label, origin=origin, transform=transform,
                         created_tick=int(tick))

    @staticmethod
    def _name(parent: str, transform: str, shape: str) -> str:
        """A name that says what was done to what, plus enough to be unique.

        The digest suffix is not decoration: the same transformation applied to
        the same parent in two different generations produces two different
        strategies only if the parent changed, and a collision on the name would
        merge their records.
        """
        return f"{parent}+{transform}-{shape[:6]}"

    async def _ask_cortex(self, weakness, parent):
        """Ask a model for a strategy. Returns steps, or None."""
        if self.cortex is None or not hasattr(self.cortex, "structured"):
            return None
        try:
            if not self.cortex.role_available("deep"):
                return None
            reply = await self.cortex.structured(
                "deep",
                [{"role": "user", "content": self._prompt(weakness, parent)}],
                "reasoning_strategy")
        except Exception:
            logger.exception("The cortex strategy path failed")
            return None
        steps = (reply or {}).get("steps") if isinstance(reply, dict) else None
        return steps if isinstance(steps, list) else None

    @staticmethod
    def _prompt(weakness, parent) -> str:
        from aegis.layers.reasoning.dsl import AGGREGATORS, CHECKERS, OPS, RETRIEVE_SOURCES

        examples = "\n".join(f"  - {example}"
                             for example in getattr(weakness, "examples", ())[:5])
        return (
            "Write one reasoning strategy that does better on this class of "
            "problem.\n\n"
            f"Weak class: {getattr(weakness, 'label', weakness)}\n"
            f"Failure rate: {getattr(weakness, 'fail_rate', 0.0):.2f} against a "
            f"baseline of {getattr(weakness, 'base_rate', 0.0):.2f} over "
            f"{getattr(weakness, 'support', 0)} attempts\n"
            f"Failing examples:\n{examples}\n\n"
            f"Current best strategy for this class ({parent.name}):\n"
            f"{normalise(parent.steps)}\n\n"
            "Answer with a name and a JSON list of steps. Operations: "
            f"{sorted(OPS)}. RETRIEVE sources: {list(RETRIEVE_SOURCES)}. "
            f"VOTE agg: {list(AGGREGATORS)}. VERIFY checkers: {list(CHECKERS)} "
            "— note that no checker can tell you the right answer. "
            f"LOOP max_iter is at most {MAX_LOOP_ITERATIONS}."
        )

    def status(self) -> dict:
        return {"proposed": self.proposed, "duplicates": self.duplicates,
                "refused": self.refused, "from_cortex": self.from_cortex,
                "transforms": [name for name, _ in TRANSFORMS]}


def traffic_share(name: str, key: str, every: int) -> bool:
    """Whether this request is the trial's, by a deterministic rule (M6.8).

    Every k-th request of the class, decided by hashing the request rather than
    by a counter: a counter would make the split depend on how many other
    strategies happened to run first, and two runs of one experiment would
    divide the traffic differently.
    """
    every = max(1, int(every))
    return hash_index(every, "reason_trial", name, key) == 0
