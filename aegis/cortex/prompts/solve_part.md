This is one part of a larger task.

Whole task: {prompt}
This part: {part}
Known so far: {context}

Solve ONLY this part. Respond with ONLY a JSON object:
{
  "answer": <the answer to this part>,
  "reasoning": "one short sentence",
  "confidence": <0.0-1.0>
}
