A weakness has been measured in how this system reasons. Propose a reasoning
strategy that addresses it.

Weakness: {weakness}
Failure examples: {examples}
Current best strategy for this class: {incumbent}

A strategy is a declarative pipeline, not Python. Available operations:
{grammar}

Respond with ONLY a JSON object:
{
  "name": "short_snake_case_name",
  "steps": [{"op": "...", ...}],
  "applies_to": {"...": "..."},
  "rationale": "why this addresses the measured weakness"
}
