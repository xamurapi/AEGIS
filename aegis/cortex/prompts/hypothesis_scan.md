Propose testable hypotheses about this system's own behaviour.

Observable variables (name: description):
{variables}

Strongest observed associations:
{associations}

A hypothesis must be checkable against recorded time series and must reference
only the variables listed above. Prefer mechanisms over restatements of the
correlation.

Respond with ONLY a JSON object:
{
  "hypotheses": [
    {
      "statement": "human-readable claim",
      "target": "<variable being explained>",
      "predictors": ["<variable>", "..."],
      "lags": {"<variable>": <integer ticks>},
      "kind": "association" or "causal" or "law",
      "rationale": "why this might hold"
    }
  ]
}
