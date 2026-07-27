Re-rank these candidate plans. You may reorder them; you may not invent one.

Current state: {state}
Candidates (index: plan):
{candidates}

Respond with ONLY a JSON object listing the candidate indices best-first:
{
  "order": [<index>, <index>, <index>],
  "rationale": "why this ordering"
}

Every index must come from the list above. Any other value is discarded and the
system keeps its own ordering.
