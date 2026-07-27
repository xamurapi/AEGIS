Solve this task by analogy with something already known.

Task ({kind}): {prompt}
Input: {payload}
Related knowledge from the graph: {related}

Identify which related item shares structure with this task, then transfer that solution.
Respond with ONLY a JSON object:
{
  "answer": <the answer>,
  "reasoning": "which analogy was used and why it holds",
  "confidence": <0.0-1.0>
}
