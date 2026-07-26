"""Self-contained mutation testing for the five higher-order cognitive systems.

Why not mutmut/mutatest? mutmut has no native Windows support (requires WSL) and
mutatest 3.1.0 crashes on Python 3.11 (`random.sample` over a set). This harness
is dependency-free, cross-platform and deterministic.

How it works (the standard mutation-testing loop):
  1. Parse each target module's AST and enumerate MUTANTS — small, semantics-
     changing edits (flip a comparison, swap +/-, negate a boolean, etc.).
  2. For each mutant: write the mutated source over the real file, run that
     module's test file in a subprocess, then ALWAYS restore the original.
  3. A mutant is KILLED if the tests fail (good — the suite caught the change)
     and SURVIVED if they still pass (a coverage/assertion gap to fix).
  4. Report the mutation score = killed / total. Surviving mutants are listed
     with file:line so they can be turned into new test cases.

Usage:
    python scripts/mutation_test.py            # all five systems
    python scripts/mutation_test.py evolution  # one system by keyword
"""
import ast
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source module, test file(s) that exercise it). The test entry may be a single
# path or a space-separated list of paths — all are run together per mutant.
TARGETS = [
    # The five new cognitive systems. Each entry also runs the round-3 audit
    # regression tests, which are what cover the hardening added to these
    # modules (shape coercion, degree index, torn-log recovery, ...).
    ("aegis/layers/world_model.py",
     "tests/test_world_model.py tests/test_audit_round3.py "
     "tests/test_capacity_and_retention.py"),
    ("aegis/layers/cognitive_graph.py",
     "tests/test_cognitive_graph.py tests/test_audit_round3.py tests/test_mutation_gaps.py"),
    ("aegis/layers/evolution_engine.py",
     "tests/test_evolution_engine.py tests/test_audit_round3.py"),
    ("aegis/layers/goal_intelligence.py", "tests/test_goal_intelligence.py"),
    ("aegis/layers/feedback_loop.py",
     "tests/test_feedback_loop.py tests/test_audit_round3.py tests/test_mutation_gaps.py"),
    # The sandbox is the single highest-consequence module in the tree: it is
    # the only thing standing between self-written skill code and the host.
    ("aegis/eval/sandbox.py",
     "tests/test_sandbox_security.py tests/test_sandbox_mutation.py "
     "tests/test_audit_round2.py tests/test_audit_round3.py "
     "tests/test_bdd_safety_resilience.py"),
    # Stage-0 foundations of the development spec. These are load-bearing for
    # everything the later stages verify: if the clock, the safety contract or
    # the state digest can be mutated without a test noticing, then every
    # "before/after" comparison built on top of them is decoration.
    ("aegis/safety/immutable.py", "tests/test_immutable_params.py"),
    ("aegis/util/canonical.py",
     "tests/test_canonical.py tests/test_determinism_e2e.py"),
    ("aegis/clock.py", "tests/test_clock.py"),
    ("aegis/telemetry/store.py", "tests/test_telemetry.py"),
    ("aegis/layers/phases/context.py", "tests/test_tick_context.py"),
    # Safety-critical / core deterministic modules (highest audit risk).
    ("aegis/event_bus.py", "tests/test_event_bus.py tests/test_mutation_gaps.py"),
    ("aegis/layers/ethics_core.py",
     "tests/test_ethics.py tests/test_ethics_core_ext.py tests/test_ethics_core_mut.py"),
    ("aegis/layers/self_preservation.py",
     "tests/test_self_preservation.py tests/test_self_preservation_ext.py "
     "tests/test_self_preservation_mut.py tests/test_mutation_gaps.py"),
    ("aegis/layers/goal_engine.py",
     "tests/test_goal_engine.py tests/test_goal_engine_mut.py"),
    ("aegis/layers/emotions.py",
     "tests/test_emotions.py tests/test_behaviour_mutation.py"),
    ("aegis/layers/health_monitor.py",
     "tests/test_health_monitor.py tests/test_behaviour_mutation.py"),
    ("aegis/layers/meta_regulation.py",
     "tests/test_meta_regulation.py tests/test_behaviour_mutation.py"),
]

# Comparison operator flips: each maps to a semantically different operator.
_CMP_FLIP = {
    ast.Lt: ast.GtE, ast.GtE: ast.Lt,
    ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_BINOP_FLIP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult,
}
_BOOLOP_FLIP = {ast.And: ast.Or, ast.Or: ast.And}


class _Mutant:
    def __init__(self, lineno, description, transform):
        self.lineno = lineno
        self.description = description
        self.transform = transform  # (tree) -> mutated tree


# Boolean-constant keyword arguments that only affect logging/serialization
# cosmetics — flipping them does not change program logic, so they are
# EQUIVALENT MUTANTS by construction and are excluded from the score.
_COSMETIC_KWARGS = {"exc_info", "ensure_ascii", "indent", "sort_keys", "reload"}


def _cosmetic_bool_constants(tree):
    """Return the set of id()s of bool Constant nodes that are values of a
    cosmetic keyword argument (equivalent mutants to skip)."""
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in _COSMETIC_KWARGS and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, bool):
                    skip.add((kw.value.lineno, kw.value.col_offset))
    return skip


def _enumerate_mutants(source: str):
    """Yield (lineno, description, mutated_source) for each candidate mutation."""
    tree = ast.parse(source)
    cosmetic = _cosmetic_bool_constants(tree)
    # Unparsed baseline — a mutant whose unparse matches this changed nothing.
    _baseline = ast.unparse(tree)
    mutants = []

    # Index every node by its ORDINAL position in a deterministic walk. This is
    # what identifies a mutation site — not (lineno, col_offset), which collide
    # for nested same-type nodes like the two BinOps in `x / 2 * 100`.
    for walk_idx, node in enumerate(ast.walk(tree)):
        # 1. Comparison operator flips.
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                repl = _CMP_FLIP.get(type(op))
                if repl:
                    mutants.append((walk_idx, node.lineno, "cmp", i, repl))
        # 2. Binary arithmetic flips.
        elif isinstance(node, ast.BinOp):
            repl = _BINOP_FLIP.get(type(node.op))
            if repl:
                mutants.append((walk_idx, node.lineno, "binop", None, repl))
        # 3. Boolean operator flips (and <-> or).
        elif isinstance(node, ast.BoolOp):
            repl = _BOOLOP_FLIP.get(type(node.op))
            if repl:
                mutants.append((walk_idx, node.lineno, "boolop", None, repl))
        # 4. Boolean constant flips (skip cosmetic logging/serialization flags).
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            if (node.lineno, node.col_offset) in cosmetic:
                continue
            mutants.append((walk_idx, node.lineno, "const", None, not node.value))

    out = []
    for walk_idx, lineno, kind, sub, repl in mutants:
        # Re-parse a fresh tree per mutant (so edits don't accumulate) and pick
        # the node at the SAME ordinal walk position — unambiguous even for
        # nested nodes that share a source position.
        fresh = ast.parse(source)
        target = next((n for i, n in enumerate(ast.walk(fresh)) if i == walk_idx), None)
        if target is None:
            continue
        desc = _apply(target, kind, sub, repl)
        try:
            mutated = ast.unparse(fresh)
        except Exception:
            continue
        # Skip degenerate mutants: if unparsing the mutated tree yields the same
        # source as the unparsed original, the "mutation" changed nothing
        # (e.g. an operator replaced by itself) — it is not a real mutant.
        if mutated == _baseline:
            continue
        out.append((lineno, f"{kind}: {desc}", mutated))
    return out


def _apply(node, kind, sub, repl):
    if kind == "cmp":
        old = type(node.ops[sub]).__name__
        node.ops[sub] = repl()
        return f"{old}->{repl.__name__} @L{node.lineno}"
    if kind == "binop":
        old = type(node.op).__name__
        node.op = repl()
        return f"{old}->{repl.__name__} @L{node.lineno}"
    if kind == "boolop":
        old = type(node.op).__name__
        node.op = repl()
        return f"{old}->{repl.__name__} @L{node.lineno}"
    if kind == "const":
        old = node.value
        node.value = repl
        return f"{old}->{repl} @L{node.lineno}"
    return "?"


def _run_tests(test_files: str) -> bool:
    """Return True if the tests PASS (mutant survived), False if they fail.

    `test_files` may be a single path or a space-separated list of paths."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *test_files.split(), "-x", "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode == 0


def _restore_any_leftover_backups():
    """Crash recovery: if a previous run was hard-killed mid-mutation, a `.mut.bak`
    holds the pristine source. Restore every such backup before doing anything —
    this GUARANTEES no mutated source is ever left on disk, even after SIGKILL."""
    for src_rel, _ in TARGETS:
        bak = (ROOT / src_rel).with_suffix(".py.mut.bak")
        if bak.exists():
            # Restore EXACT bytes (binary) so line endings / encoding are
            # preserved and the file does not show as modified afterwards.
            (ROOT / src_rel).write_bytes(bak.read_bytes())
            bak.unlink()
            print(f"[recovery] restored {src_rel} from leftover backup")


def mutate_module(src_rel: str, test_rel: str) -> dict:
    src_path = ROOT / src_rel
    backup_path = src_path.with_suffix(".py.mut.bak")
    # Keep the EXACT original bytes for restoration; decode a copy for AST work.
    original_bytes = src_path.read_bytes()
    original = original_bytes.decode("utf-8")
    # Persist a pristine backup to disk so a hard kill (which skips `finally`)
    # is still recoverable on the next run via _restore_any_leftover_backups().
    backup_path.write_bytes(original_bytes)
    mutants = _enumerate_mutants(original)
    killed, survived, errored = 0, 0, []
    print(f"\n== {src_rel}: {len(mutants)} mutants (tests: {test_rel}) ==")
    try:
        for lineno, desc, mutated in mutants:
            src_path.write_text(mutated, encoding="utf-8", newline="\n")
            try:
                passed = _run_tests(test_rel)
            except Exception:
                passed = True
            if passed:
                survived += 1
                errored.append(f"  SURVIVED L{lineno}: {desc}")
            else:
                killed += 1
    finally:
        # ALWAYS restore the pristine source bytes exactly, even on Ctrl-C /
        # exception — no encoding or line-ending translation.
        src_path.write_bytes(original_bytes)
        backup_path.unlink(missing_ok=True)

    total = killed + survived
    score = round(100 * killed / total, 1) if total else 100.0
    print(f"   killed={killed} survived={survived} score={score}%")
    for line in errored:
        print(line)
    return {"module": src_rel, "killed": killed, "survived": survived,
            "total": total, "score": score, "surviving": errored}


def main():
    _restore_any_leftover_backups()  # crash recovery from a prior hard kill
    keyword = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [t for t in TARGETS if not keyword or keyword in t[0]]
    if not targets:
        print(f"No target matches '{keyword}'. Options: {[t[0] for t in TARGETS]}")
        return 1

    t0 = time.time()
    results = [mutate_module(src, test) for src, test in targets]
    tot_killed = sum(r["killed"] for r in results)
    tot = sum(r["total"] for r in results)
    overall = round(100 * tot_killed / tot, 1) if tot else 100.0

    print("\n" + "=" * 60)
    print("MUTATION TESTING SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['module']:<42} {r['score']:>5}%  "
              f"({r['killed']}/{r['total']} killed)")
    print("-" * 60)
    print(f"  {'OVERALL MUTATION SCORE':<42} {overall:>5}%  "
          f"({tot_killed}/{tot} killed)")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    # Non-zero exit if any mutant survived — usable as a CI gate.
    return 0 if tot_killed == tot else 2


if __name__ == "__main__":
    sys.exit(main())
