"""Auto-composition of arbitrary depth — discover a skill pipeline by SEARCH.

Where composite.py runs a hand-declared pipeline, this searches for one: given a
start string and a target, breadth-first over the available string->string
primitive skills, applying each in the sandbox and branching, until some chain
produces the target (within a depth bound). This is automatic hierarchical
planning — the agent composes primitives it has, to depth it discovers, with no
pipeline specified in advance. A target needing a primitive the agent hasn't
learned yet is simply unreachable until that primitive exists.
"""
from collections import deque
from dataclasses import dataclass, field

# Kinds whose skills follow the {"s": str} -> str contract and can be chained.
TRANSFORM_KINDS = ("reverse", "sort_csv", "upper")


@dataclass(frozen=True)
class AutoComposeTask:
    id: str
    start: str
    target: str
    max_depth: int = 3
    kinds: tuple = field(default=TRANSFORM_KINDS)


# auto1 is reachable with seeded primitives (upper + reverse); auto2 needs the
# sort_csv primitive, so it stays unreachable until that skill is learned.
AUTOCOMPOSE_BENCHMARK: list[AutoComposeTask] = [
    AutoComposeTask("auto_upper_rev", "abc", "CBA", max_depth=3),
    AutoComposeTask("auto_sort_rev", "3,1,2", "3,2,1", max_depth=3),
]
