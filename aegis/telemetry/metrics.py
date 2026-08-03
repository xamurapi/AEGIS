"""Canonical metric names (spec Appendix G).

Metric names are a contract, not labels. The discovery engine fits models over
series looked up by name, the dashboard plots them by name, and
``test_metrics_contract`` asserts that every name here actually appears during
a run. A typo in a string literal at the call site would silently create a
second, empty series and break all three — so the names live here once and the
call sites import them.

Format: ``aegis.<contour>.<metric>``. Tagged metrics (per phase, per kind, per
provider) are recorded under the base name with tags attached, which keeps one
file per metric while preserving the breakdown.
"""
from __future__ import annotations

# ── tick and reward ──────────────────────────────────────────────────
TICK_DURATION_MS = "aegis.tick.duration_ms"
TICK_PHASE_MS = "aegis.tick.phase_ms"              # tag: phase
REWARD_VALUE = "aegis.reward.value"
REWARD_ENV_ROLLING = "aegis.reward.env_rolling"
BENCH_SCORE = "aegis.bench.score"
BENCH_PER_KIND = "aegis.bench.per_kind"            # tag: kind

# ── M1 predictive world model ────────────────────────────────────────
WM_BRIER = "aegis.wm.brier"
WM_ECE = "aegis.wm.ece"
WM_REWARD_MAE = "aegis.wm.reward_mae"
WM_NLL_NEXT = "aegis.wm.nll_next"
WM_SURPRISE = "aegis.wm.surprise"
WM_COVERAGE = "aegis.wm.coverage"
WM_STATES = "aegis.wm.states"
WM_TRANSITIONS = "aegis.wm.transitions"
WM_ROLLOUT_MS = "aegis.wm.rollout_ms"

# ── M2 planner ───────────────────────────────────────────────────────
PLAN_OVERRIDE_RATE = "aegis.plan.override_rate"
PLAN_EV_GAP = "aegis.plan.ev_gap"
PLAN_LATENCY_MS = "aegis.plan.latency_ms"
PLAN_BLOCKED = "aegis.plan.blocked"                # tag: reason
PLAN_DEPTH = "aegis.plan.depth"

# ── M3 behaviour policy ──────────────────────────────────────────────
POLICY_BEHAVIOUR_DELTA_RATE = "aegis.policy.behaviour_delta_rate"
POLICY_ACTIVE_RULES = "aegis.policy.active_rules"
POLICY_REFUTED = "aegis.policy.refuted"
POLICY_REGRET = "aegis.policy.regret"
POLICY_CANDIDATES = "aegis.policy.candidates"

# ── M4 resources ─────────────────────────────────────────────────────
RES_SPENT = "aegis.res.spent"                      # tag: kind
RES_DENIED = "aegis.res.denied"
RES_STARVATION_TICKS = "aegis.res.starvation_ticks"
RES_ROI = "aegis.res.roi"                          # tag: activity
RES_SHARE = "aegis.res.share"                      # tag: drive

# ── M5 evolution ─────────────────────────────────────────────────────
EVO_GENERATION = "aegis.evo.generation"
EVO_CHAMPION_FITNESS = "aegis.evo.champion_fitness"
EVO_VALID_TEST_GAP = "aegis.evo.valid_test_gap"
EVO_PROMOTIONS = "aegis.evo.promotions"
EVO_ROLLBACKS = "aegis.evo.rollbacks"
EVO_NOVELTY_SKIPS = "aegis.evo.novelty_skips"

# ── M6 reasoning ─────────────────────────────────────────────────────
REASON_PASS_HOLDOUT = "aegis.reason.pass_holdout"
#: In-sample accuracy over the live queue — deliberately a SEPARATE name from
#: the holdout above. The two were published under one name for a long time,
#: which made every downstream claim about held-out reasoning performance a
#: claim about the data the engine trains on (audit R5). Both are worth having:
#: the gap between them is what overfitting looks like from the outside.
REASON_ACCURACY = "aegis.reason.accuracy"
REASON_STRATEGIES_ACTIVE = "aegis.reason.strategies_active"
REASON_WIN_RATE = "aegis.reason.win_rate"          # tag: strategy
REASON_ABSTAIN_RATE = "aegis.reason.abstain_rate"
REASON_CONFIDENT_ERROR = "aegis.reason.confident_error"

# ── M7 discovery ─────────────────────────────────────────────────────
DISC_HYPOTHESES_TESTED = "aegis.disc.hypotheses_tested"
DISC_SUPPORTED = "aegis.disc.supported"
DISC_REPLICATED = "aegis.disc.replicated"
DISC_FDR_REJECTIONS = "aegis.disc.fdr_rejections"
DISC_EXPERIMENTS = "aegis.disc.experiments"

# ── M8 cortex ────────────────────────────────────────────────────────
CORTEX_CALLS = "aegis.cortex.calls"                # tag: role
CORTEX_TOKENS = "aegis.cortex.tokens"              # tag: provider
CORTEX_SCHEMA_FAILURES = "aegis.cortex.schema_failures"
CORTEX_REPAIRS = "aegis.cortex.repairs"
CORTEX_BREAKER_TRIPS = "aegis.cortex.breaker_trips"
CORTEX_CACHE_HIT_RATE = "aegis.cortex.cache_hit_rate"

# ── memory, graph, health ────────────────────────────────────────────
MEM_SEMANTIC = "aegis.mem.semantic"
MEM_EPISODIC = "aegis.mem.episodic"
GRAPH_NODES = "aegis.graph.nodes"
HEALTH_STATUS_CODE = "aegis.health.status_code"
HEALTH_CONSECUTIVE_ERRORS = "aegis.health.consecutive_errors"

# Health is a word on the wire and a number in a time series; the mapping is
# fixed here so a chart of it means the same thing in every consumer.
HEALTH_STATUS_CODES: dict[str, int] = {
    "healthy": 0, "ok": 0,
    "warning": 1, "degraded": 1,
    "critical": 2,
    "unknown": 3,
}


def health_code(status: str) -> int:
    return HEALTH_STATUS_CODES.get(str(status), 3)


#: Every metric the running system is required to publish (Appendix G).
#: ``test_metrics_contract`` walks this list against a real run.
REQUIRED_METRICS: tuple[str, ...] = (
    TICK_DURATION_MS, TICK_PHASE_MS, REWARD_VALUE, REWARD_ENV_ROLLING,
    BENCH_SCORE, BENCH_PER_KIND,
    WM_BRIER, WM_ECE, WM_REWARD_MAE, WM_NLL_NEXT, WM_SURPRISE, WM_COVERAGE,
    WM_STATES, WM_TRANSITIONS, WM_ROLLOUT_MS,
    PLAN_OVERRIDE_RATE, PLAN_EV_GAP, PLAN_LATENCY_MS, PLAN_BLOCKED, PLAN_DEPTH,
    POLICY_BEHAVIOUR_DELTA_RATE, POLICY_ACTIVE_RULES, POLICY_REFUTED,
    POLICY_REGRET, POLICY_CANDIDATES,
    RES_SPENT, RES_DENIED, RES_STARVATION_TICKS, RES_ROI, RES_SHARE,
    EVO_GENERATION, EVO_CHAMPION_FITNESS, EVO_VALID_TEST_GAP, EVO_PROMOTIONS,
    EVO_ROLLBACKS, EVO_NOVELTY_SKIPS,
    REASON_PASS_HOLDOUT, REASON_ACCURACY, REASON_STRATEGIES_ACTIVE, REASON_WIN_RATE,
    REASON_ABSTAIN_RATE, REASON_CONFIDENT_ERROR,
    DISC_HYPOTHESES_TESTED, DISC_SUPPORTED, DISC_REPLICATED,
    DISC_FDR_REJECTIONS, DISC_EXPERIMENTS,
    CORTEX_CALLS, CORTEX_TOKENS, CORTEX_SCHEMA_FAILURES, CORTEX_REPAIRS,
    CORTEX_BREAKER_TRIPS, CORTEX_CACHE_HIT_RATE,
    MEM_SEMANTIC, MEM_EPISODIC, GRAPH_NODES,
    HEALTH_STATUS_CODE, HEALTH_CONSECUTIVE_ERRORS,
)
