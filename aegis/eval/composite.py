"""Step 3 — composite tasks solved by COMPOSING primitive skills (hierarchy).

A composite task names a pipeline of primitive kinds. The solver threads a
string through them (output of one becomes ``{"s": output}`` for the next),
reusing skills already in the library. No new atomic skill is needed — capability
emerges from composition. If any primitive in the pipeline is unsolved, the
composite fails, which makes the capability *hierarchy* explicit: composites
unlock only once their building blocks are learned.
"""
from dataclasses import dataclass
from typing import Any

from aegis.eval.benchmark import _norm


@dataclass(frozen=True)
class CompositeTask:
    id: str
    pipeline: tuple        # ordered primitive kinds, each str->str on {"s": ...}
    payload: dict          # initial input, must contain "s"
    expected: Any

    @property
    def kind(self) -> str:
        return "compose"

    def verify(self, answer: Any) -> bool:
        try:
            return _norm(answer) == _norm(self.expected)
        except Exception:
            return False


# Both pipelines depend on sort_csv (a synthesis target), so they stay unsolved
# until that primitive is learned — demonstrating the dependency hierarchy.
COMPOSITE_BENCHMARK: list[CompositeTask] = [
    # sort_csv("3,1,2")="1,2,3" -> reverse -> "3,2,1"
    CompositeTask("comp_sort_rev", ("sort_csv", "reverse"), {"s": "3,1,2"}, "3,2,1"),
    # reverse("30,1,2")="2,1,03" -> sort_csv -> "1,2,3"
    CompositeTask("comp_rev_sort", ("reverse", "sort_csv"), {"s": "30,1,2"}, "1,2,3"),
]
