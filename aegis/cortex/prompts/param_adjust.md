Analyze these performance metrics and recommend parameter adjustments.

Current parameters:
{parameters}

Performance:
- Success rate: {success_rate}
- Error rate: {error_rate}
- Energy: {energy}
- Memory concepts: {semantic_concepts}
- Information gain: {information_gain}
- Goals completed: {goals_completed}
- Tick: {tick}

Only recommend an adjustment where the metrics clearly indicate a problem. A
healthy system needs no changes — return an empty list in that case.

Respond with ONLY a JSON object:
{
  "adjustments": [
    {"parameter": "name", "direction": "increase" or "decrease", "magnitude": 0.01-0.1, "reason": "why"}
  ],
  "assessment": "overall system health assessment"
}
