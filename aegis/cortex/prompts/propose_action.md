Choose the next action for the system.

Current state: {state}
Available actions: {actions}

Respond with ONLY a JSON object:
{
  "chosen": <1-based action number>,
  "reasoning": "why this action now",
  "confidence": <0.0-1.0>,
  "ethical_concerns": "any ethical concerns or 'none'"
}
