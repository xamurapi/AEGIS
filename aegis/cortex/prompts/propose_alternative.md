The world model predicts the proposed action is likely to fail.

Current state: {state}
Available actions: {actions}
Rejected action: {rejected}
Predicted success probability: {p_success}

Choose a DIFFERENT action with a better outlook. Respond with ONLY a JSON object:
{
  "chosen": <1-based action number, not the rejected one>,
  "reasoning": "why this one should do better",
  "confidence": <0.0-1.0>,
  "ethical_concerns": "any ethical concerns or 'none'"
}
