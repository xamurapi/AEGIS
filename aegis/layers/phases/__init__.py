"""The five phases of the cognitive cycle (spec §3.9).

``Substrate`` keeps the state and the schedule; each phase keeps its own logic.
The substrate had reached 1763 lines with seven more systems still to land,
which is where a file stops being reviewable. Splitting it was the precondition
for every later stage, so it happens first and changes nothing else: the bodies
were moved verbatim and the existing suite passes untouched.
"""
from aegis.layers.phases.context import TickContext

__all__ = ["TickContext"]
