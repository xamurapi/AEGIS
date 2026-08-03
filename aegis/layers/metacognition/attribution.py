"""Attribution: "why did this strategy win", as a measurement (spec M11.5).

An explanation is the easiest thing in the world to fake — a language model
will produce a plausible reason for any outcome, including a random one, and
no reading of the text can tell. So "why" here is a **testable claim**: remove
the part of the strategy the explanation credits, and the win must disappear.

The pipeline, in the order the spec fixes it:

1. ``diff(incumbent, candidate)`` — the atomic edits, in canonical form. More
   than ``META_MAX_EDITS`` of them and the strategy is ``unsupported``:
   ablating one edit out of many would need subset enumeration and would not
   be a measurement. A limit on explainability, never on acceptance.
2. For each edit, ``S∖e`` — the candidate with that edit reverted — is scored
   against ``S`` on **held-out** problems of the weak class, sampled by hash
   indexing from ``(strategy, edit, position)`` (§M11.8). Held-out is not
   hygiene but the point: measured on the tuning set, the "effect" would be
   the candidate's fit to the examples it was selected on.
3. Effect, a Wilson-style interval on the difference of proportions
   (Newcombe's method over the two Wilson intervals), and a two-sided
   proportion test — all from ``util/stats.py``, no external libraries.
4. **Benjamini–Hochberg over the whole family of edits**, not over the ones
   that looked promising. Confirmed needs all three: BH at ``META_FDR_Q``,
   ``effect ≥ META_MIN_EFFECT``, ``wilson_low > 0``.
5. The mechanism label comes from the largest confirmed effect through the
   declared mapping. No confirmed edit — no mechanism, ``unsupported``.
6. The cortex may add a narrative and its own mechanism *guess*; a guess that
   contradicts the computed mechanism makes the explanation ``contested`` and
   changes no number. ``narrative`` is stored for the human and read by
   nothing (a test greps this module for exactly that).

The heavy step (2) runs in the evaluation pool via the module-level, picklable
:func:`ablation_worker` — ``META_MAX_EDITS × META_ABLATION_N`` interpreter runs
are seconds, and §3.4 requires seconds to leave the tick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import aegis.config as cfg
from aegis.layers.metacognition.distance import canonical_hash, canonicalize
from aegis.layers.metacognition.mechanism import mechanism_for
from aegis.util.quasirandom import hash_index
from aegis.util.stats import benjamini_hochberg, two_proportion_z, wilson_interval

logger = logging.getLogger("aegis.metacognition")

#: Task indices for ablation live here — disjoint from the arena's three bases
#: (1M/2M/3M), the working queue (up from 0) and the holdout probe (down from
#: 10M), so an effect is never measured on anything the candidate was tuned on.
ABLATION_BASE = 4_000_000
ABLATION_SPAN = 1_000_000

STATUSES = ("supported", "unsupported", "contested")


@dataclass(frozen=True)
class Edit:
    """One atomic edit distinguishing the candidate from the incumbent."""

    kind: str          # "insert" | "remove" | "param" | "wrap" | "reorder"
    position: int      # step index in the canonical form
    op: str            # the operation the edit concerns
    key: str = ""      # for kind == "param"
    before: object = None
    after: object = None

    def signature(self) -> str:
        """Stable identity, the hash-index seed of the ablation sample."""
        return f"{self.kind}:{self.position}:{self.op}:{self.key}"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "position": self.position, "op": self.op,
                "key": self.key, "before": self.before, "after": self.after}

    @classmethod
    def from_dict(cls, data: dict) -> "Edit":
        return cls(kind=str(data.get("kind", "")),
                   position=int(data.get("position", 0)),
                   op=str(data.get("op", "")), key=str(data.get("key", "")),
                   before=data.get("before"), after=data.get("after"))


@dataclass(frozen=True)
class EditAttribution:
    edit: Edit
    effect: float          # pass(S) − pass(S∖e)
    n: int                 # held-out sample size
    wilson_low: float      # lower 95% bound of the effect
    p_value: float
    confirmed: bool

    def as_dict(self) -> dict:
        return {"edit": self.edit.as_dict(), "effect": round(self.effect, 6),
                "n": self.n, "wilson_low": round(self.wilson_low, 6),
                "p_value": round(self.p_value, 6), "confirmed": self.confirmed}

    @classmethod
    def from_dict(cls, data: dict) -> "EditAttribution":
        return cls(edit=Edit.from_dict(data.get("edit") or {}),
                   effect=float(data.get("effect", 0.0)),
                   n=int(data.get("n", 0)),
                   wilson_low=float(data.get("wilson_low", 0.0)),
                   p_value=float(data.get("p_value", 1.0)),
                   confirmed=bool(data.get("confirmed")))


@dataclass(frozen=True)
class Explanation:
    strategy: str
    incumbent: str
    weakness: str
    gain: float
    edits: tuple[EditAttribution, ...] = ()
    mechanism: str = ""
    narrative: str = ""    # the cortex's free text; stored, never read by code
    status: str = "supported"
    created_tick: int = 0

    def confirmed_edits(self) -> tuple[EditAttribution, ...]:
        return tuple(edit for edit in self.edits if edit.confirmed)

    def as_dict(self) -> dict:
        return {"strategy": self.strategy, "incumbent": self.incumbent,
                "weakness": self.weakness, "gain": round(self.gain, 6),
                "edits": [edit.as_dict() for edit in self.edits],
                "mechanism": self.mechanism, "narrative": self.narrative,
                "status": self.status, "created_tick": self.created_tick}

    @classmethod
    def from_dict(cls, data: dict) -> "Explanation":
        status = str(data.get("status", "supported"))
        return cls(strategy=str(data.get("strategy", "")),
                   incumbent=str(data.get("incumbent", "")),
                   weakness=str(data.get("weakness", "")),
                   gain=float(data.get("gain", 0.0)),
                   edits=tuple(EditAttribution.from_dict(row)
                               for row in data.get("edits") or []
                               if isinstance(row, dict)),
                   mechanism=str(data.get("mechanism", "")),
                   narrative=str(data.get("narrative", "")),
                   status=status if status in STATUSES else "supported",
                   created_tick=int(data.get("created_tick", 0)))


# ── the diff (M11.5.2 step 1) ────────────────────────────────────────

def _scalar_fields(step: dict) -> dict:
    from aegis.layers.reasoning.dsl import OPS

    spec = OPS.get(str(step.get("op")))
    bodies = set(spec.bodies) if spec else set()
    return {key: value for key, value in step.items()
            if key != "op" and key not in bodies}


def _body_fields(step: dict) -> dict:
    from aegis.layers.reasoning.dsl import OPS

    spec = OPS.get(str(step.get("op")))
    bodies = set(spec.bodies) if spec else set()
    return {key: step[key] for key in step if key in bodies}


def _render(step: dict) -> str:
    import json

    return json.dumps(step, sort_keys=True, ensure_ascii=False)


def diff(incumbent_steps, candidate_steps) -> tuple[Edit, ...]:
    """Atomic edits from the incumbent to the candidate, canonical form.

    Top-level alignment by longest common subsequence over operation names.
    Special cases first, because they are single edits wearing several steps:
    a candidate that is the incumbent wrapped in one ``VOTE`` is a ``wrap``;
    the same steps in a different order are one ``reorder``; an ``LLM_STEP``
    replaced in place by a ``COMPUTE`` is one ``param`` edit (the guess/compute
    substitution the mechanism table names).
    """
    old = canonicalize(incumbent_steps)
    new = canonicalize(candidate_steps)
    if canonical_hash(old) == canonical_hash(new):
        return ()

    # wrap: the whole incumbent inside a single VOTE.
    if len(new) == 1 and new[0].get("op") == "VOTE" \
            and canonical_hash(new[0].get("body") or []) == canonical_hash(old):
        return (Edit(kind="wrap", position=0, op="VOTE",
                     before=None, after=_scalar_fields(new[0])),)

    # reorder: same steps, different order.
    if sorted(map(_render, old)) == sorted(map(_render, new)):
        return (Edit(kind="reorder", position=0,
                     op=str(new[0].get("op", "")) if new else "",
                     before=old, after=new),)

    ops_old = [str(step.get("op")) for step in old]
    ops_new = [str(step.get("op")) for step in new]
    aligned = _lcs_pairs(ops_old, ops_new)

    edits: list[Edit] = []
    matched_old = {i for i, _ in aligned}
    matched_new = {j for _, j in aligned}

    # Aligned steps whose fields changed → param edits.
    for i, j in aligned:
        before_step, after_step = old[i], new[j]
        scalars_before, scalars_after = (_scalar_fields(before_step),
                                         _scalar_fields(after_step))
        for key in sorted(set(scalars_before) | set(scalars_after)):
            if scalars_before.get(key) != scalars_after.get(key):
                edits.append(Edit(kind="param", position=j,
                                  op=str(after_step.get("op")), key=key,
                                  before=scalars_before.get(key),
                                  after=scalars_after.get(key)))
        bodies_before, bodies_after = (_body_fields(before_step),
                                       _body_fields(after_step))
        for key in sorted(set(bodies_before) | set(bodies_after)):
            if canonical_hash(bodies_before.get(key) or []) \
                    != canonical_hash(bodies_after.get(key) or []):
                edits.append(Edit(kind="param", position=j,
                                  op=str(after_step.get("op")), key=key,
                                  before=bodies_before.get(key),
                                  after=bodies_after.get(key)))

    removed = [(i, old[i]) for i in range(len(old)) if i not in matched_old]
    inserted = [(j, new[j]) for j in range(len(new)) if j not in matched_new]

    # LLM_STEP → COMPUTE in place is one substitution, not two edits.
    for r_index, (i, gone) in enumerate(list(removed)):
        if str(gone.get("op")) != "LLM_STEP":
            continue
        for a_index, (j, came) in enumerate(list(inserted)):
            if str(came.get("op")) == "COMPUTE":
                edits.append(Edit(kind="param", position=j, op="COMPUTE",
                                  key="op", before=gone, after=came))
                removed.pop(r_index)
                inserted.pop(a_index)
                break
        break

    for j, step in inserted:
        edits.append(Edit(kind="insert", position=j, op=_insert_op(step),
                          before=None, after=step))
    for i, step in removed:
        edits.append(Edit(kind="remove", position=i, op=str(step.get("op")),
                          before=step, after=None))

    edits.sort(key=lambda edit: (edit.position, edit.kind, edit.op, edit.key))
    return tuple(edits)


def _insert_op(step: dict) -> str:
    """What an inserted step *is*, for the mechanism mapping.

    A ``BRANCH`` whose entire body is abstention is an abstention edit wearing
    the branch it needs to fire — the shape ``add_abstain`` (M6.7) produces.
    Labelling it ``BRANCH`` would file the benchmark's most important edit
    under prediction, and the credit table would learn the wrong lesson.
    """
    from aegis.layers.metacognition.distance import op_multiset

    op = str(step.get("op"))
    if op == "BRANCH":
        inner = {name for name in op_multiset([step]) if name != "BRANCH"}
        if inner == {"ABSTAIN"}:
            return "ABSTAIN"
    return op


def _lcs_pairs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Index pairs of a longest common subsequence, leftmost alignment."""
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            rows[i][j] = (rows[i + 1][j + 1] + 1 if a[i] == b[j]
                          else max(rows[i + 1][j], rows[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif rows[i + 1][j] >= rows[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


# ── the revert (S∖e) ─────────────────────────────────────────────────

def revert(candidate_steps, edit: Edit) -> list:
    """The candidate with one edit rolled back — what ablation runs."""
    steps = canonicalize(candidate_steps)
    if edit.kind == "insert":
        return [step for index, step in enumerate(steps)
                if index != edit.position]
    if edit.kind == "remove":
        position = max(0, min(len(steps), edit.position))
        return steps[:position] + [edit.before] + steps[position:]
    if edit.kind == "wrap":
        if steps and steps[0].get("op") == "VOTE":
            return canonicalize(steps[0].get("body") or [])
        return steps
    if edit.kind == "reorder":
        return canonicalize(edit.before or [])
    if edit.kind == "param":
        out = []
        for index, step in enumerate(steps):
            if index != edit.position:
                out.append(step)
                continue
            if edit.key == "op":
                # The whole-step substitution (LLM_STEP → COMPUTE): put the
                # replaced step back.
                out.append(edit.before)
                continue
            reverted = dict(step)
            if edit.before is None:
                reverted.pop(edit.key, None)
            else:
                reverted[edit.key] = edit.before
            out.append(reverted)
        return out
    raise ValueError(f"unknown edit kind {edit.kind!r}")


# ── the measurement (M11.5.2 steps 2-4) ──────────────────────────────

def ablation_tasks(family: str, strategy_digest: str, edit_signature: str,
                   count: int) -> list:
    """Held-out tasks for one edit, by hash indexing (§M11.8).

    Each position's index is a blake2b of ``(strategy, edit, position)``
    mapped into the ablation range — deterministic, structurally disjoint from
    everything the candidate was selected on, and different per edit so a
    lucky sample cannot flatter every edit at once.
    """
    from aegis.eval import reasoning_bench as bench

    builder = bench.BUILDERS.get(str(family))
    tasks = []
    for position in range(max(1, int(count))):
        offset = hash_index(ABLATION_SPAN, "meta_ablation", strategy_digest,
                            edit_signature, position)
        if builder is not None:
            tasks.append(builder(ABLATION_BASE + offset))
        else:
            # A weakness that spans families is measured on the full mix.
            tasks.append(bench.build(ABLATION_BASE + offset))
    return tasks


def newcombe_low(successes_a: int, trials_a: int,
                 successes_b: int, trials_b: int) -> float:
    """Lower 95% bound on ``p_a − p_b`` from the two Wilson intervals.

    Newcombe's score method: the interval of a difference is assembled from
    the two one-sample Wilson intervals — the same estimator every other
    proportion in this system uses, so "confident" means one thing.
    """
    p_a = successes_a / trials_a if trials_a else 0.0
    p_b = successes_b / trials_b if trials_b else 0.0
    low_a, _ = wilson_interval(successes_a, trials_a)
    _, high_b = wilson_interval(successes_b, trials_b)
    return (p_a - p_b) - ((p_a - low_a) ** 2 + (high_b - p_b) ** 2) ** 0.5


def ablation_worker(payload: dict) -> dict:
    """Score the candidate and every ``S∖e`` — picklable, pool-runnable.

    Builds a bare interpreter: no cortex, no memory, no sandbox. That is not a
    shortcut but the deterministic path the arena itself scores on, and the
    only kind of dependency a subprocess can be handed (M9.1).
    """
    from aegis.layers.reasoning.interpreter import Interpreter

    interpreter = Interpreter(genome=dict(payload.get("genome") or {}))
    budget = payload.get("budget")
    family = str(payload.get("family", ""))
    strategy_digest = str(payload.get("digest", ""))
    count = int(payload.get("n", cfg.META_ABLATION_N))

    def passes(steps, tasks) -> int:
        solved = 0
        for task in tasks:
            trace = interpreter.run(steps, task, budget=budget)
            solved += 1 if trace.solved else 0
        return solved

    rows = []
    for entry in payload.get("edits") or []:
        tasks = ablation_tasks(family, strategy_digest,
                               str(entry.get("signature", "")), count)
        rows.append({
            "signature": str(entry.get("signature", "")),
            "candidate_solved": passes(payload.get("candidate") or [], tasks),
            "reverted_solved": passes(entry.get("reverted") or [], tasks),
            "n": len(tasks),
        })
    return {"rows": rows}


def attribute_edits(edits: tuple[Edit, ...], measured: dict, *,
                    fdr_q: float | None = None,
                    min_effect: float | None = None) -> tuple[EditAttribution, ...]:
    """Fold measured pass counts into attributions, BH over the whole family.

    ``measured`` maps ``edit.signature()`` to a row of the worker's output.
    The BH correction runs over **every** edit's p-value — correcting only the
    promising ones is how noise becomes a finding (M11.5.2 step 4).
    """
    fdr_q = float(cfg.META_FDR_Q if fdr_q is None else fdr_q)
    min_effect = float(cfg.META_MIN_EFFECT if min_effect is None else min_effect)

    stats = []
    for edit in edits:
        row = measured.get(edit.signature()) or {}
        n = int(row.get("n", 0))
        solved_s = int(row.get("candidate_solved", 0))
        solved_r = int(row.get("reverted_solved", 0))
        effect = (solved_s / n - solved_r / n) if n else 0.0
        low = newcombe_low(solved_s, n, solved_r, n) if n else -1.0
        p_value = two_proportion_z(solved_s, n, solved_r, n).p_value if n else 1.0
        stats.append((edit, effect, n, low, p_value))

    survives = benjamini_hochberg([p for *_ , p in stats], fdr_q)
    out = []
    for (edit, effect, n, low, p_value), significant in zip(stats, survives):
        confirmed = bool(significant and effect >= min_effect and low > 0.0)
        out.append(EditAttribution(edit=edit, effect=effect, n=n,
                                   wilson_low=low, p_value=p_value,
                                   confirmed=confirmed))
    return tuple(out)


def conclude(strategy: str, incumbent: str, weakness: str, gain: float,
             attributions: tuple[EditAttribution, ...], *,
             too_many_edits: bool = False, tick: int = 0) -> Explanation:
    """Assemble the explanation the numbers actually support (step 5)."""
    if too_many_edits or not attributions:
        return Explanation(strategy=strategy, incumbent=incumbent,
                           weakness=weakness, gain=gain, edits=attributions,
                           mechanism="", status="unsupported",
                           created_tick=int(tick))
    confirmed = [a for a in attributions if a.confirmed]
    if not confirmed:
        return Explanation(strategy=strategy, incumbent=incumbent,
                           weakness=weakness, gain=gain, edits=attributions,
                           mechanism="", status="unsupported",
                           created_tick=int(tick))
    best = max(confirmed, key=lambda a: (a.effect, -a.p_value,
                                         a.edit.signature()))
    return Explanation(strategy=strategy, incumbent=incumbent,
                       weakness=weakness, gain=gain, edits=attributions,
                       mechanism=mechanism_for(best.edit.op, best.edit.kind),
                       status="supported", created_tick=int(tick))


def apply_narrative(explanation: Explanation, narrative: str,
                    proposed_mechanism: str) -> Explanation:
    """Attach the cortex's story (step 6). The numbers do not move.

    A proposed mechanism that contradicts the computed one makes the
    explanation ``contested`` — the disagreement is recorded, and ordering
    keeps using the computed mechanism. The cortex cannot rewrite a
    conclusion; it can only be seen to disagree with one.
    """
    contested = bool(explanation.mechanism and proposed_mechanism
                     and proposed_mechanism != explanation.mechanism)
    return replace(explanation, narrative=str(narrative or ""),
                   status="contested" if contested else explanation.status)
