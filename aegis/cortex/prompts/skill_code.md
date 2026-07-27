Write a single pure Python function to solve tasks of kind '{kind}'.

Signature: def solve(payload): ...   # payload is a dict, return the answer.
Examples (must satisfy ALL):
{examples}
{feedback}
Rules:
- Pure computation only. Imports limited to: math, statistics, itertools, functools, re, json, collections, string.
- No eval/exec/open/__import__, no file/network/OS access, no print.
- The function must GENERALIZE beyond the examples shown; it is graded on cases you have not seen.

Respond with ONLY a JSON object:
{
  "code": "def solve(payload):\n    ...",
  "explanation": "the rule the function implements"
}
