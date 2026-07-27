Write code that computes the answer instead of reasoning it out.

Task ({kind}): {prompt}
Input (available as the dict `payload`): {payload}

Rules:
- Pure computation only. Imports limited to: math, statistics, itertools, functools, re, json, collections, string.
- No eval/exec/open/__import__, no I/O, no print.
- Define exactly one function: def solve(payload): ...

Respond with ONLY a JSON object:
{
  "code": "def solve(payload):\n    ...",
  "explanation": "what it computes"
}
