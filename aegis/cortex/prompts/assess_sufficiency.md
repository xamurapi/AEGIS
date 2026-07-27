Decide whether this task can be answered at all with what is given.

Task ({kind}): {prompt}
Input: {payload}
Retrieved context: {context}

A task with missing data must be refused, not guessed at — a confident wrong
answer is worse than an admitted gap.
Respond with ONLY a JSON object:
{
  "sufficient": true or false,
  "missing": ["what is missing, if anything"],
  "reason": "one short sentence"
}
