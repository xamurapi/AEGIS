Your previous answer was checked and rejected.

Task ({kind}): {prompt}
Input: {payload}
Your answer: {last_answer}
Verification result: {error}

Work out where that went wrong, then answer again. Respond with ONLY a JSON object:
{
  "answer": <the corrected answer>,
  "reasoning": "what was wrong the first time",
  "confidence": <0.0-1.0>
}
