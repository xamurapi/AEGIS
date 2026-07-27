The task was split into parts and each was solved separately.

Whole task: {prompt}
Input: {payload}
Part results: {parts}

Combine them into the final answer. Respond with ONLY a JSON object:
{
  "answer": <the final answer to the whole task>,
  "reasoning": "how the parts combine",
  "confidence": <0.0-1.0>
}
