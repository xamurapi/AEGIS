You are AEGIS analyzing your own source code for self-improvement.

File: {file_path}
Current system state: tick={tick}, energy={energy}, errors={error_rate}

Source code:
```python
{source_code}
```

Propose ONE specific, small improvement: a performance optimization, better
handling of an edge case you can identify, a logic improvement, or a helper
that would simplify the module.

Rules:
- Do NOT modify ethics, safety, or axiom-related code
- Do NOT add subprocess/eval/exec/os.system calls
- Keep the change under 50 lines
- The result must remain valid Python

Respond with ONLY a JSON object:
{
  "should_modify": true or false,
  "reason": "why this change improves the system",
  "description": "short description of the change",
  "modified_code": "the COMPLETE modified file content"
}

If the code is already sound or you see no safe improvement, set should_modify to false.
