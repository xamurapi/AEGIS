Refine this causal chain for an objective.

Objective: {objective}
Constraints: {constraints}
Known risks from the world model: {risks}
Deterministic plan: {plan}

Respond with ONLY a JSON object:
{
  "objective": "restated objective",
  "constraints": ["..."],
  "risks": [{"cause": "...", "effect": "...", "failure_rate": <0.0-1.0>}],
  "plan": [{"action": "...", "expected": "...", "confidence": <0.0-1.0>}],
  "expected_result": "what should happen if the plan works",
  "confidence": <0.0-1.0>
}
