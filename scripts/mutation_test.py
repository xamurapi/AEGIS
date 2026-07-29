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
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A report must not be able to kill the run. The equivalence arguments below are
# written with real mathematical notation (−, Σ, ·), and on a Windows console
# claiming cp1251 the first `print` of one raised UnicodeEncodeError *after* the
# sources had been restored but *before* the summary — so a sweep that had
# already done an hour of work died at the point of telling anyone what it found,
# and the surviving mutants it had located were lost. Encoding is a property of
# the terminal, not of the result.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Wall-clock ceiling for one mutant's test run. Generous enough that a slow but
# finite suite is never mistaken for a hang, tight enough that a non-terminating
# mutant costs seconds rather than the whole build.
MUTANT_TIMEOUT = float(os.environ.get("MUTANT_TIMEOUT", "180"))

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
    ("aegis/safety/immutable.py",
     "tests/test_immutable_params.py tests/test_immutable_across_contours.py"),
    ("aegis/util/canonical.py",
     "tests/test_canonical.py tests/test_determinism_e2e.py"),
    ("aegis/clock.py", "tests/test_clock.py"),
    ("aegis/telemetry/store.py",
     "tests/test_telemetry.py tests/test_sigterm_resilience.py"),
    ("aegis/layers/phases/context.py", "tests/test_tick_context.py"),
    # Determinism substitutes and versioned persistence. Both are load-bearing
    # for every later stage: a quasirandom sequence that silently stopped
    # spreading, or a migration that silently dropped a field, would corrupt
    # results everywhere downstream without failing anything locally.
    ("aegis/util/quasirandom.py", "tests/test_quasirandom.py"),
    ("aegis/store/migrations.py",
     "tests/test_migrations.py tests/test_legacy_state.py"),
    # Stage 1 — the cortex. The schema validator is the boundary that stops
    # malformed model output from entering state, and the breaker is what keeps
    # a dead provider from eating a phase budget in timeouts; a surviving
    # mutant in either is a hole in a guarantee the whole system rests on.
    ("aegis/cortex/schemas.py",
     "tests/test_cortex_schemas.py tests/test_cortex_fuzz.py tests/test_cortex_edges.py"),
    ("aegis/cortex/breaker.py", "tests/test_cortex_breaker.py"),
    ("aegis/cortex/cache.py",
     "tests/test_cortex_cache.py tests/test_cortex_edges.py"),
    ("aegis/cortex/router.py",
     "tests/test_cortex_router.py tests/test_cortex_budget.py "
     "tests/test_cortex_telemetry.py tests/test_cortex_edges.py"),
    # Stage 2 — resources. The lease is what makes motivation binding, and the
    # safety floor is what keeps health checks funded when everything else is
    # competing for the same budget; a surviving mutant in either would mean
    # the guarantee is decorative.
    ("aegis/layers/motivation/resources.py", "tests/test_resources.py"),
    ("aegis/layers/motivation/roi.py", "tests/test_roi.py"),
    ("aegis/layers/motivation/priority.py", "tests/test_priority.py"),
    ("aegis/layers/actions.py",
     "tests/test_action_registry.py tests/test_action_preconditions.py"),
    # Stage 3 — the predictive world model. Every later contour reads these
    # estimates: the planner ranks on them, the policy measures against them,
    # the discovery engine fits laws to them. A silently wrong probability here
    # is a wrong decision everywhere downstream.
    ("aegis/util/stats.py",
     "tests/test_prediction_scoring.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py "
     "tests/test_stats_inference.py"),
    ("aegis/layers/world/state.py",
     "tests/test_world_state.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py"),
    ("aegis/layers/world/transition.py",
     "tests/test_transition_model.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py"),
    ("aegis/layers/world/outcome.py",
     "tests/test_outcome_model.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py"),
    ("aegis/layers/world/prediction.py",
     "tests/test_prediction_scoring.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py"),
    ("aegis/layers/world/simulate.py",
     "tests/test_simulate.py tests/test_world_edges.py "
     "tests/test_world_formulas.py tests/test_world_guards.py"),
    # Stage 4 — the planner and the gate sequence. The scoring is where the
    # world model actually changes a decision, and Appendix J's ordering is a
    # safety property rather than a style: a mutant that moves the ethics gate
    # or lets the cortex reach past the shortlist must not survive.
    ("aegis/layers/planner.py",
     "tests/test_planner.py tests/test_planner_gates.py "
     "tests/test_planner_mutation.py tests/test_stage4_edges.py "
     "tests/test_bdd_planning.py"),
    ("aegis/layers/phases/decide.py",
     "tests/test_planner_gates.py tests/test_decide_mutation.py "
     "tests/test_stage4_edges.py tests/test_bdd_planning.py "
     "tests/test_audit_stage4_fixes.py"),
    ("aegis/layers/executors.py", "tests/test_stage4_edges.py"),
    # Stage 5 — the behaviour policy. This is the contour that can *remove* an
    # option, so its thresholds are load-bearing: a mutant that weakened the
    # false-discovery control, the support floor or the safety-critical
    # exemption would let the policy suppress things on evidence that does not
    # exist.
    ("aegis/layers/policy/store.py",
     "tests/test_policy_store.py tests/test_policy_mutation.py"),
    ("aegis/layers/policy/rules.py",
     "tests/test_rule_miner.py tests/test_rule_lifecycle.py "
     "tests/test_policy_integration.py tests/test_policy_mutation.py"),
    ("aegis/layers/policy/counterfactual.py",
     "tests/test_shadow_evaluator.py tests/test_policy_integration.py "
     "tests/test_policy_mutation.py"),
    ("aegis/layers/policy/__init__.py",
     "tests/test_policy_integration.py tests/test_bdd_behaviour_change.py "
     "tests/test_policy_mutation.py"),
    # Stage 6 — evaluation infrastructure. The split decides what "held out"
    # means, the generators decide whether the benchmark can be memorised, and
    # the pool decides whether a generation's result depends on machine load.
    # A silent mutant in any of the three corrupts every measurement built on
    # top without failing anything locally.
    ("aegis/eval/generators.py", "tests/test_generators_and_splits.py"),
    ("aegis/eval/pool.py", "tests/test_eval_pool.py"),
    ("aegis/eval/isolated.py", "tests/test_isolated_eval.py"),
    ("aegis/eval/benchmark.py",
     "tests/test_generators_and_splits.py tests/test_no_leakage.py "
     "tests/test_eval_layer.py"),
    ("aegis/eval/solver.py",
     "tests/test_isolated_eval.py tests/test_eval_layer.py "
     "tests/test_generators_and_splits.py"),
    # Stage 7 — population evolution. The genome decides what evolution can
    # even search over, the operators decide whether two runs agree, and the
    # rollback is the only thing standing between a benchmark-friendly genome
    # and a permanently worse system.
    ("aegis/layers/evolution/genome.py", "tests/test_evolution_population.py"),
    ("aegis/layers/evolution/operators.py", "tests/test_evolution_population.py"),
    ("aegis/layers/evolution/population.py", "tests/test_evolution_population.py"),
    ("aegis/layers/evolution_engine.py",
     "tests/test_evolution_engine.py tests/test_evolution_population.py "
     "tests/test_audit_round3.py"),
    # Stage 8 — the reasoning contour. The grammar and the interpreter are a
    # security boundary rather than a convenience: strategies are synthesised,
    # some of them by a language model, and a mutation that quietly widened
    # what may be admitted or how long a loop may run would be a hole nobody
    # would notice from the outside.
    ("aegis/layers/reasoning/dsl.py",
     "tests/test_reasoning_dsl.py tests/test_bdd_reasoning.py"),
    ("aegis/layers/reasoning/interpreter.py",
     "tests/test_interpreter.py tests/test_interpreter_operations.py "
     "tests/test_bdd_reasoning.py"),
    ("aegis/layers/reasoning/reasoner.py", "tests/test_reasoning_bench.py"),
    ("aegis/layers/reasoning/library.py",
     "tests/test_strategy_library.py tests/test_bdd_reasoning.py"),
    ("aegis/layers/reasoning/__init__.py",
     "tests/test_strategy_library.py tests/test_arena.py "
     "tests/test_reasoning_selection.py tests/test_bdd_reasoning.py"),
    ("aegis/eval/reasoning_bench.py", "tests/test_reasoning_bench.py"),
    # Stage 9 — the improvement loop. The detector decides what gets worked on,
    # the arena decides what is believed. A mutation that weakened either would
    # let the system change its own reasoning on evidence that does not exist.
    ("aegis/layers/reasoning/weakness.py",
     "tests/test_weakness.py tests/test_bdd_reasoning.py"),
    ("aegis/layers/reasoning/synthesis.py",
     "tests/test_synthesis.py tests/test_bdd_reasoning.py"),
    ("aegis/layers/reasoning/arena.py",
     "tests/test_arena.py tests/test_bdd_reasoning.py"),
    # Stage 10 — the discovery contour. Every module here decides what the
    # system comes to believe about itself. A weakened correction, a lag that
    # leaked, a preregistration that could be edited after the fact, or a
    # refutation that could be forgotten would each let the engine accumulate
    # laws that are not there — and unlike a bug, that failure looks like
    # success while it is happening.
    ("aegis/layers/discovery/datapool.py", "tests/test_datapool.py"),
    ("aegis/layers/discovery/statistics.py", "tests/test_discovery_statistics.py"),
    ("aegis/layers/discovery/hypothesis.py",
     "tests/test_hypothesis_scan.py tests/test_bdd_discovery.py"),
    ("aegis/layers/discovery/symbolic.py",
     "tests/test_symbolic.py tests/test_bdd_discovery.py"),
    ("aegis/layers/discovery/experiment.py",
     "tests/test_experiment_prereg.py tests/test_intervention_safety.py "
     "tests/test_bdd_discovery.py"),
    ("aegis/layers/discovery/ledger.py",
     "tests/test_ledger.py tests/test_bdd_discovery.py"),
    ("aegis/layers/discovery/__init__.py",
     "tests/test_discovery_engine.py tests/test_bdd_discovery.py"),
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


# Mutants that are EQUIVALENT to the original by argument rather than by
# construction — the program behaves identically, so no test can kill them and
# a surviving one is not a gap in the suite.
#
# The spec's gate is "a surviving *non-equivalent* mutant fails the build", so
# equivalence has to be recordable. Three rules keep this from becoming a place
# to hide real gaps:
#
#   1. Every entry carries a written argument for why no observable behaviour
#      differs. "The test is hard to write" is not one.
#   2. The entry is keyed on the *text* of the line, not its number. Edit the
#      line and the annotation stops applying, so the mutant comes back rather
#      than silently covering something else.
#   3. The count of skipped mutants is printed, so a run can never quietly
#      excuse more than a reader expects.
#
# Keyed by (path suffix, mutant description, stripped source line).
KNOWN_EQUIVALENT: dict[tuple[str, str, str], str] = {
    (
        "discovery/hypothesis.py",
        "boolop: Or->And",
        "if not names or len(frame) < self.min_rows:",
    ): (
        "A fast path, fully subsumed by the per-pair guard below it. With `and` "
        "the scan proceeds instead of returning early, but every lagged frame it "
        "then builds is no longer than `frame` — `numeric()` and `lag()` only "
        "ever drop rows — so each one trips `len(lagged) < self.min_rows` and is "
        "skipped. Both versions return [], leave `tested` and `rejected` "
        "untouched, and record the same `scans`. There is nothing to observe."
    ),
    (
        "discovery/symbolic.py",
        "boolop: Or->And",
        "if not rows or not predictors:",
    ): (
        "Both arms already end in None by another route, confirmed by running "
        "the mutant: with no rows `usable` is empty and the train/valid split "
        "fails its minimum-size guard; with no predictors the library is empty, "
        "the first size produces no candidates, and the frontier empties on the "
        "first pass. `fit` returns None either way and touches nothing else."
    ),
    (
        "discovery/symbolic.py",
        "binop: Sub->Add",
        "residual_sd = (sum((value - residual_mean) ** 2 for value in residuals)",
    ): (
        "Equivalent by the algebra of the fit, not by luck. Least squares with "
        "an intercept forces the residuals to sum to zero, so `residual_mean` "
        "is zero to within floating-point noise (~1e-16) and `value - mean` "
        "and `value + mean` differ by twice that. No dataset can separate them "
        "while the design carries an intercept column, and it always does — "
        "`_design_matrix` prepends 1.0 to every row."
    ),
    (
        "discovery/statistics.py",
        "binop: Sub->Add",
        "cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))",
    ): (
        "Algebraically identical, for either factor. Expanding "
        "Σ(a + mx)(b − my) gives Σab − my·Σa + mx·Σb − n·mx·my, and since "
        "my·Σa = n·mx·my = mx·Σb the middle terms cancel exactly, leaving "
        "Σab − n·mx·my — which is the covariance. The same cancellation holds "
        "for Σ(a − mx)(b + my). Verified numerically as well: 75.0 against "
        "74.99999999999997, a difference of one floating-point ulp."
    ),
    (
        "discovery/statistics.py",
        "boolop: Or->And",
        "if high_x <= low_x or high_y <= low_y:",
    ): (
        "A fast path that computes the same answer the long way. With `and`, a "
        "constant series is no longer caught by the guard, but `_bin_index` "
        "returns 0 for a degenerate range, so that series occupies exactly one "
        "bin; its marginal probability is 1, every log term becomes log(1) = 0, "
        "and the information is 0.0. Its margin then has one entry, so "
        "`df = (1 − 1) · (k − 1)` is zero and the function returns "
        "(0.0, 1.0) — byte for byte what the guard returns."
    ),
}


def _equivalence_reason(src_rel: str, description: str, source: str,
                        lineno: int) -> str | None:
    """The recorded argument for this mutant being equivalent, if there is one."""
    lines = source.splitlines()
    if not 1 <= lineno <= len(lines):
        return None
    line = lines[lineno - 1].strip()
    # The description carries the site (" @L144"); the key does not, so that
    # moving a line does not need the annotation rewritten.
    bare = description.split(" @L")[0].strip()
    normalised = src_rel.replace("\\", "/")
    for (path_suffix, wanted, wanted_line), reason in KNOWN_EQUIVALENT.items():
        if (normalised.endswith(path_suffix) and bare == wanted
                and line == wanted_line):
            return reason
    return None


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

    `test_files` may be a single path or a space-separated list of paths.

    The timeout is load-bearing, not defensive. Mutating an arithmetic operator
    inside a loop routinely produces code that never terminates — turning
    ``n //= base`` into ``n *= base`` is enough — and without a bound the whole
    gate hangs forever on the first such mutant instead of failing. A mutant
    that hangs is a mutant the tests did NOT accept, so a timeout counts as a
    kill: the change was detected, by wall clock rather than by assertion.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *test_files.split(), "-x", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, timeout=MUTANT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False
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
    equivalent: list[str] = []
    print(f"\n== {src_rel}: {len(mutants)} mutants (tests: {test_rel}) ==")
    try:
        for lineno, desc, mutated in mutants:
            reason = _equivalence_reason(src_rel, desc, original, lineno)
            if reason is not None:
                # Not run at all: an equivalent mutant cannot be killed, and
                # spending a test cycle proving that on every run is waste.
                equivalent.append(f"  EQUIVALENT L{lineno}: {desc} — {reason}")
                continue
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
    excused = f", equivalent={len(equivalent)}" if equivalent else ""
    print(f"   killed={killed} survived={survived}{excused} score={score}%")
    for line in errored:
        print(line)
    for line in equivalent:
        print(line)
    return {"module": src_rel, "killed": killed, "survived": survived,
            "total": total, "score": score, "surviving": errored,
            "equivalent": len(equivalent)}


def main():
    _restore_any_leftover_backups()  # crash recovery from a prior hard kill
    keyword = sys.argv[1] if len(sys.argv) > 1 else None
    targets = [t for t in TARGETS if not keyword or keyword in t[0]]
    if not targets:
        print(f"No target matches '{keyword}'. Options: {[t[0] for t in TARGETS]}")
        return 1

    t0 = time.time()
    # One module must not be able to discard the whole sweep. A full run costs
    # over an hour, and the first version lost every result it had gathered
    # because module sixty-one raised while printing its own report. A module
    # that fails to run is recorded as a failure and named in the summary — the
    # run continues, and the build still goes red.
    results, broken = [], []
    for src, test in targets:
        try:
            results.append(mutate_module(src, test))
        except Exception as exc:
            broken.append((src, f"{type(exc).__name__}: {exc}"))
            print(f"   !! {src} could not be analysed: {type(exc).__name__}: {exc}")
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
    if broken:
        print("-" * 60)
        for src, why in broken:
            print(f"  NOT ANALYSED  {src:<42} {why}")
    print(f"  elapsed: {time.time() - t0:.1f}s")
    # Non-zero exit if any mutant survived, or if any module could not be
    # analysed at all: an unanalysed module is an unmeasured one, and reporting
    # a green sweep over a subset of the targets would be the worst outcome here.
    return 0 if (tot_killed == tot and not broken) else 2


if __name__ == "__main__":
    sys.exit(main())
