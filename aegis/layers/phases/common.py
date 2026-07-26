"""Values and helpers shared by more than one phase.

They live here rather than in ``substrate.py`` so a phase module never has to
import the substrate — that would be a cycle, since the substrate imports the
phases.
"""

# External learning sources — cycled in order, not random.
_LEARNING_SOURCES = ["wikipedia", "arxiv", "quotes"]

# Meta-knowledge domains — cycled in order.
_META_DOMAINS = ["reasoning", "memory", "ethics", "planning", "creativity"]

# Concept seeds — cycled in order.
_CONCEPT_SEEDS = ["pattern", "cycle", "adaptation", "learning", "stability"]


# ── LLM-output coercion (audit M5) ───────────────────────────────────
# Models routinely return "almost right" JSON — numbers as strings, an object
# where a scalar was asked for, a scalar where an object was asked for. Using
# those values raw (``parsed["chosen"] - 1``, ``knowledge.get(...)`` on a str)
# raises inside a cognitive phase and drops the rest of the tick. These helpers
# coerce defensively so a malformed field falls back instead of crashing.

def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
