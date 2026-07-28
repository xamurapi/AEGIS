"""AEGIS Configuration."""
import os
import logging
from pathlib import Path

_log = logging.getLogger("aegis.config")


def _env_int(name: str, default) -> int:
    """Parse an int env var, falling back (with a warning) on a bad value —
    a typo in one env var must not crash the whole package import (audit L4)."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        _log.warning("Invalid int for %s=%r; using default %r",
                     name, os.environ.get(name), default)
        return int(default)


def _env_float(name: str, default) -> float:
    """Parse a float env var, falling back (with a warning) on a bad value."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        _log.warning("Invalid float for %s=%r; using default %r",
                     name, os.environ.get(name), default)
        return float(default)


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LOGS_DIR = DATA_DIR / "logs"
MEMORY_DIR = DATA_DIR / "memory"

# --- Five higher-order cognitive systems (World Model, Cognitive Graph,
#     Evolution Engine, Goal Intelligence, Feedback Loop) ---
WORLD_MODEL_DIR = DATA_DIR / "world_model"
COGNITIVE_GRAPH_DIR = DATA_DIR / "cognitive_graph"
EVOLUTION_DIR = DATA_DIR / "evolution"
GOAL_INTEL_DIR = DATA_DIR / "goal_intelligence"
FEEDBACK_DIR = DATA_DIR / "feedback"

# --- Telemetry: metric time-series on disk (spec M9.2) ---
# Every contour publishes here. The discovery engine reads these series as its
# primary data source, so they are storage, not just dashboard decoration.
TELEMETRY_DIR = DATA_DIR / "telemetry"
# Flush the write buffer at least this often (seconds) and whenever this many
# rows are pending — bounded latency AND bounded memory.
TELEMETRY_FLUSH_SECONDS = _env_float("TELEMETRY_FLUSH_SECONDS", "10")
TELEMETRY_FLUSH_ROWS = _env_int("TELEMETRY_FLUSH_ROWS", "200")
# Rows kept per metric. Beyond twice this, the OLDER half is downsampled
# (averaged into buckets) rather than deleted: the shape of long history is
# what the discovery engine needs, and truncation would destroy exactly that.
TELEMETRY_MAX_ROWS = _env_int("TELEMETRY_MAX_ROWS", "200000")

for d in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, MEMORY_DIR,
          WORLD_MODEL_DIR, COGNITIVE_GRAPH_DIR, EVOLUTION_DIR,
          GOAL_INTEL_DIR, FEEDBACK_DIR, TELEMETRY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TICK_INTERVAL = 3.0
CHECKPOINT_EVERY_N_TICKS = 10
MAX_WORKING_MEMORY = 50
MEMORY_DECAY_RATE = 0.02
ETHICAL_THRESHOLD_AUTO = 0.7
ETHICAL_THRESHOLD_REVIEW = 0.85
# Bind to loopback by default — the control plane can activate/deactivate the
# kill switch, grant permissions and trigger self-modification, so it must NOT
# be network-exposed unless the operator explicitly opts in via AEGIS_API_HOST.
API_HOST = os.environ.get("AEGIS_API_HOST", "127.0.0.1")
API_PORT = _env_int("AEGIS_API_PORT", "8888")
# Shared-secret token for mutating (POST) endpoints. When set, clients must send
# it in the `X-API-Token` header. Empty string = no token required (only safe
# together with the loopback bind above).
API_TOKEN = os.environ.get("AEGIS_API_TOKEN", "")
# Allowed CORS origins (comma-separated). Empty = no cross-origin access.
API_CORS_ORIGINS = [o.strip() for o in os.environ.get("AEGIS_API_CORS_ORIGINS", "").split(",") if o.strip()]

# Loopback / link-local hosts that are NOT network-exposed.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", ""}


def network_exposure_warning(host: str = API_HOST, token: str = API_TOKEN) -> str | None:
    """Return a warning if the API is bound to a non-loopback address WITHOUT an
    auth token — that exposes the kill switch, permission grants and code
    self-modification triggers to the network unauthenticated (audit M8).
    Returns None when the configuration is safe.
    """
    if host not in _LOOPBACK_HOSTS and not token:
        return (f"AEGIS API is bound to a non-loopback host ({host!r}) with an EMPTY "
                f"AEGIS_API_TOKEN — mutating endpoints (kill switch, permissions, "
                f"self-modification) are network-exposed WITHOUT authentication. "
                f"Set AEGIS_API_TOKEN or bind to 127.0.0.1.")
    return None


# Surface the misconfiguration loudly at import so it cannot pass silently.
_exposure = network_exposure_warning()
if _exposure:
    logging.getLogger("aegis.config").warning(_exposure)

# DeepSeek LLM
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_ENABLED = True
LLM_THINK_EVERY_N_TICKS = 3  # use LLM every N ticks to save tokens
LLM_MAX_TOKENS = 5000
LLM_TEMPERATURE = 0.7
# Hard wall-clock timeout for any single LLM call (audit H3). Without it a
# hung/slow provider (or a stuck local generate()) suspends the tick loop / a
# dashboard request indefinitely. Applied to hosted SDK clients and wrapped
# around the local model's executor call.
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", "60")

# Claude (Anthropic)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Valid, current model id (audit M4 — the previous default "claude-opus-4-6" is
# not a real model and 404s every Claude call). Override via CLAUDE_MODEL env.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

# LLM Provider: "deepseek", "claude", "both", "local" (local transformers model)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "both")

# --- Trainable local model (the LoRA contour, spec M8.3) ---
# This model exists to be CHANGED, not to be clever: it is the only place the
# system rewrites its own weights. Its size is bounded by what fits in memory
# together with activations and gradients on the reference machine (§M8.3a),
# not by how well it reasons — reasoning is the cortex's job.
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LOCAL_MODEL_DEVICE = os.environ.get("LOCAL_MODEL_DEVICE", "cpu")  # "auto", "cuda", "cpu"
# bfloat16, not float32: on the reference profile (CPU-only torch, 32 GB RAM)
# float32 doubles the memory a model needs and buys nothing (§M8.3a).
LOCAL_MODEL_DTYPE = os.environ.get("LOCAL_MODEL_DTYPE", "bfloat16")  # float16|bfloat16|float32
LOCAL_MODEL_QUANTIZE = os.environ.get("LOCAL_MODEL_QUANTIZE", "")  # "4bit", "8bit", "" (none)
LOCAL_MODEL_MAX_LENGTH = _env_int("LOCAL_MODEL_MAX_LENGTH", "4096")
# Fraction of a model's raw weight size that a training run additionally needs
# for activations, gradients and optimizer state. Used by the pre-load memory
# guard so an oversized model is refused with an explanation instead of dying
# inside from_pretrained (§M8.3a).
LOCAL_MODEL_TRAIN_OVERHEAD = _env_float("LOCAL_MODEL_TRAIN_OVERHEAD", "3.0")
LOCAL_MODEL_INFER_OVERHEAD = _env_float("LOCAL_MODEL_INFER_OVERHEAD", "1.3")

# --- Weight modification / LoRA fine-tuning ---
WEIGHT_CHECKPOINTS_DIR = DATA_DIR / "weight_checkpoints"
WEIGHT_DATASETS_DIR = DATA_DIR / "datasets"
WEIGHT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LORA_R = _env_int("LORA_R", "16")              # LoRA rank
LORA_ALPHA = _env_int("LORA_ALPHA", "32")       # LoRA alpha
LORA_DROPOUT = _env_float("LORA_DROPOUT", "0.05")
# Which projections LoRA attaches to depends on the model ARCHITECTURE, not on
# our preference: q/k/v/o_proj is right for Llama, Qwen, Mistral and Gemma, but
# Phi-3 fuses them into qkv_proj. Hardcoding the list meant that swapping model
# family silently trained zero parameters (§M8.3c), so it comes from the
# environment and WeightModifier verifies at least one match before training.
LORA_TARGET_MODULES = [m.strip() for m in
                       os.environ.get("LORA_TARGET_MODULES",
                                      "q_proj,k_proj,v_proj,o_proj").split(",")
                       if m.strip()] or ["q_proj", "k_proj", "v_proj", "o_proj"]


def quantization_error(quantize: str = LOCAL_MODEL_QUANTIZE) -> str | None:
    """Why the requested quantization cannot be used here, or None if it can.

    ``bitsandbytes`` 4-bit/8-bit loading needs CUDA. On the reference profile
    (CPU-only torch) requesting it used to blow up deep inside
    ``from_pretrained``; refusing at configuration time with a sentence the
    operator can act on is the difference between a bug and a setting.
    """
    if not quantize:
        return None
    if quantize not in ("4bit", "8bit"):
        return (f"LOCAL_MODEL_QUANTIZE={quantize!r} is not recognised — "
                f"use '4bit', '8bit' or leave it empty.")
    try:
        import torch  # noqa: PLC0415 — optional heavy dependency
    except Exception:
        return (f"LOCAL_MODEL_QUANTIZE={quantize!r} requires torch with CUDA, "
                f"but torch is not installed.")
    if not torch.cuda.is_available():
        return (f"LOCAL_MODEL_QUANTIZE={quantize!r} requires a CUDA device "
                f"(bitsandbytes has no CPU kernels). This machine has none — "
                f"unset LOCAL_MODEL_QUANTIZE and use LOCAL_MODEL_DTYPE=bfloat16 "
                f"with a model of 3B parameters or smaller.")
    return None

TRAIN_BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", "1")
TRAIN_GRADIENT_ACCUMULATION = _env_int("TRAIN_GRADIENT_ACCUMULATION", "4")
TRAIN_EPOCHS = _env_int("TRAIN_EPOCHS", "3")
TRAIN_LEARNING_RATE = _env_float("TRAIN_LEARNING_RATE", "2e-4")
TRAIN_MAX_SAMPLES = _env_int("TRAIN_MAX_SAMPLES", "500")
TRAIN_VAL_SPLIT = _env_float("TRAIN_VAL_SPLIT", "0.1")

# Safety limits
TRAIN_MIN_INTERVAL_SECONDS = _env_int("TRAIN_MIN_INTERVAL", "3600")  # 1 hour minimum
TRAIN_MAX_CHECKPOINTS = _env_int("TRAIN_MAX_CHECKPOINTS", "5")
TRAIN_EVERY_N_TICKS = _env_int("TRAIN_EVERY_N_TICKS", "1000")
TRAIN_MIN_DATASET_SIZE = _env_int("TRAIN_MIN_DATASET_SIZE", "50")
TRAIN_VAL_LOSS_THRESHOLD = _env_float("TRAIN_VAL_LOSS_THRESHOLD", "0.5")  # max acceptable val loss increase

# --- Code Self-Modification ---
# DISABLED BY DEFAULT (audit C2). Untrusted external content (web fetches,
# agent feeds) flows into semantic memory and then into LLM prompts; the same
# LLM channel proposes whole-file rewrites of the aegis package that are applied
# to disk and executed after restart — an indirect-prompt-injection path to
# arbitrary code execution that the pattern/AST blocklists cannot fully close.
# Autonomous source self-modification therefore requires an explicit operator
# opt-in. Parametric self-modification and LoRA weight training are unaffected.
CODE_SELF_MOD_ENABLED = os.environ.get("AEGIS_CODE_SELF_MOD_ENABLED", "0") == "1"
CODE_BACKUPS_DIR = DATA_DIR / "code_backups"
CODE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
CODE_MOD_EVERY_N_TICKS = _env_int("CODE_MOD_EVERY_N_TICKS", "500")  # attempt code mod every N ticks
CODE_MOD_MIN_TICK = _env_int("CODE_MOD_MIN_TICK", "100")  # don't modify code before this tick
CODE_MOD_MAX_PER_SESSION = _env_int("CODE_MOD_MAX_PER_SESSION", "10")  # max code mods per run
# Only attempt whole-file LLM rewrites on files at or below this size (chars).
# The model must return the entire file, so larger files can't be regenerated
# reliably within the modification-size cap.
CODE_MOD_MAX_FILE_CHARS = _env_int("CODE_MOD_MAX_FILE_CHARS", "4000")

# --- Capability layer: eval harness, skill library, sandbox, environment ---
EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
# Periodic held-out benchmark run (the fitness graph). Detached, non-blocking.
EVAL_EVERY_N_TICKS = _env_int("EVAL_EVERY_N_TICKS", "50")
# Live environment step: agent attempts one real task and gets real reward.
ENV_STEP_EVERY_N_TICKS = _env_int("ENV_STEP_EVERY_N_TICKS", "2")
# Skill synthesis: propose+sandbox-test a new skill for a failing task kind.
SKILL_SYNTH_EVERY_N_TICKS = _env_int("SKILL_SYNTH_EVERY_N_TICKS", "200")
# Hard timeout (seconds) for a single sandboxed skill execution.
SANDBOX_TIMEOUT = _env_float("SANDBOX_TIMEOUT", "3.0")

# --- Five higher-order systems: cadence (ticks between activations) ---
# World Model: observe cause->effect + build a causal chain for the focus.
WORLD_MODEL_EVERY_N_TICKS = _env_int("WORLD_MODEL_EVERY_N_TICKS", "5")
# Cognitive Graph: ingest recent memory into the typed graph.
COGNITIVE_GRAPH_EVERY_N_TICKS = _env_int("COGNITIVE_GRAPH_EVERY_N_TICKS", "8")
# Evolution Engine: propose a mutation; judged when the next benchmark lands.
EVOLUTION_EVERY_N_TICKS = _env_int("EVOLUTION_EVERY_N_TICKS", "100")
# Goal Intelligence runs every tick inside DECIDE (cheap, deterministic).

# --- Behaviour closure: how strongly learned structures steer the tick ---
# Upper bound on how much a course of action's observed failure history may cut
# decision confidence. Confidence feeds the ethics gate, so this is capped: a
# long tail of remembered failures must not be able to drive the system into
# permanent self-doubt and stall every action.
MAX_RISK_CONFIDENCE_PENALTY = _env_float("MAX_RISK_CONFIDENCE_PENALTY", "0.4")

# --- Capacity homeostasis ---
# How big the causal model and the knowledge graph are allowed to get is not a
# number anyone can pick correctly in advance: it depends on the machine. So it
# is not picked in advance — the caps track MEASURED tick latency, which the
# health monitor already records and already has a threshold for. Ticks well
# under the threshold buy capacity; ticks over it give capacity back. The
# module constants stay as the floor, and growth is hard-capped.
CAPACITY_EVERY_N_TICKS = _env_int("CAPACITY_EVERY_N_TICKS", "50")
CAPACITY_GROWTH_FACTOR = _env_float("CAPACITY_GROWTH_FACTOR", "1.25")
CAPACITY_SHRINK_FACTOR = _env_float("CAPACITY_SHRINK_FACTOR", "0.8")
# Grow only while average tick latency is below this fraction of the threshold.
CAPACITY_HEADROOM = _env_float("CAPACITY_HEADROOM", "0.5")
# Ceiling as a multiple of the baseline cap.
CAPACITY_MAX_MULTIPLE = _env_float("CAPACITY_MAX_MULTIPLE", "20")

# Broadcast the (large) full status over WebSocket at most every N ticks.
WS_BROADCAST_EVERY_N_TICKS = _env_int("WS_BROADCAST_EVERY_N_TICKS", "1")

# --- Per-phase latency budgets (spec §3.4) ---
# The whole-tick threshold (tick_duration_ms) is too coarse to protect the
# cognitive cycle: planning and rollouts land in DECIDE, and a regression there
# hides inside an average that also contains a benchmark run. Budgets are
# measured only on ticks where the phase did NO external work (network, LLM,
# subprocess) — otherwise the number describes the provider, not the code.
PHASE_BUDGET_MS = {
    "perceive": _env_float("PHASE_BUDGET_PERCEIVE_MS", "5"),
    "evaluate": _env_float("PHASE_BUDGET_EVALUATE_MS", "10"),
    "decide": _env_float("PHASE_BUDGET_DECIDE_MS", "30"),
    "act": _env_float("PHASE_BUDGET_ACT_MS", "20"),
    "reflect": _env_float("PHASE_BUDGET_REFLECT_MS", "15"),
}
# A single slow tick is noise (GC pause, scheduler hiccup); a budget is breached
# only when the average over this many local ticks stays above it.
PHASE_BUDGET_WINDOW = _env_int("PHASE_BUDGET_WINDOW", "20")

# --- LLM call budget (per process run) ---
# Hard ceiling on LLM calls per run to bound token spend; 0 = unlimited.
# Kept as the OUTERMOST fuse: from stage 2 on, every cortex call also needs a
# lease from the ResourceManager, which is the real accounting (§M4.3).
LLM_MAX_CALLS_PER_RUN = _env_int("LLM_MAX_CALLS_PER_RUN", "0")
# Minimum seconds between any two LLM calls (simple rate limit); 0 = none.
LLM_MIN_INTERVAL_SECONDS = _env_float("LLM_MIN_INTERVAL_SECONDS", "0")


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ══════════════════════════════════════════════════════════════════════
# Development spec §5.2 — the seven new contours
# ══════════════════════════════════════════════════════════════════════

# --- Storage roots (spec §5.3) ---
POLICY_DIR = DATA_DIR / "policy"
MOTIVATION_DIR = DATA_DIR / "motivation"
REASONING_DIR = DATA_DIR / "reasoning"
DISCOVERY_DIR = DATA_DIR / "discovery"
CORTEX_DIR = DATA_DIR / "cortex"

for d in [POLICY_DIR, MOTIVATION_DIR, REASONING_DIR, DISCOVERY_DIR, CORTEX_DIR,
          DISCOVERY_DIR / "datasets"]:
    d.mkdir(parents=True, exist_ok=True)

# --- M1: predictive world model -------------------------------------
# Additive smoothing with back-off: a state/action pair seen fewer than
# WM_MIN_N times borrows its estimate from the action's marginal instead of
# inventing a confident number from two observations.
WM_SMOOTHING = _env_float("WM_SMOOTHING", "1.0")
WM_MIN_N = _env_int("WM_MIN_N", "3")
WM_BRANCH = _env_int("WM_BRANCH", "3")          # successor states expanded per node
WM_BEAM = _env_int("WM_BEAM", "5")
WM_DEPTH = _env_int("WM_DEPTH", "3")
WM_DISCOUNT = _env_float("WM_DISCOUNT", "0.9")
# Counts decay by 0.5 every WM_HALF_LIFE observations, so the model follows the
# system as it changes instead of averaging over a world that no longer exists.
WM_HALF_LIFE = _env_int("WM_HALF_LIFE", "500")
WM_MAX_PREDICTIONS = _env_int("WM_MAX_PREDICTIONS", "20000")
WM_MAX_STATES = _env_int("WM_MAX_STATES", "20000")
WM_EXPLORE_BONUS = _env_float("WM_EXPLORE_BONUS", "0.15")
WM_CALIBRATION_BINS = _env_int("WM_CALIBRATION_BINS", "10")
WM_SURPRISE_WINDOW = _env_int("WM_SURPRISE_WINDOW", "50")

# State encoding boundaries (spec Appendix D). JSON so the whole scheme can be
# replaced from the environment; the bin edges are themselves genome material.
_WM_STATE_BINS_DEFAULT = {
    "energy": {"lo": 0.33, "hi": 0.66},
    "error": {"none": 0.0001, "low": 0.05, "high": 0.20},
    "load": {"lo": 0.5, "hi": 1.0},
    "perf": {"window": 5, "flat_band": 0.01},
    "mood": "as_is",
    "mode": "as_is",
    "focus_kind": "via_goal_intelligence",
}


def _env_json(name: str, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        import json as _json
        parsed = _json.loads(raw)
    except Exception:
        _log.warning("Invalid JSON for %s; using default", name)
        return default
    return parsed if isinstance(parsed, type(default)) else default


WM_STATE_BINS = _env_json("WM_STATE_BINS", _WM_STATE_BINS_DEFAULT)

# --- M2: planner -----------------------------------------------------
PLAN_ENABLED = _env_str("PLAN_ENABLED", "1") == "1"
PLAN_MAX_CANDIDATES = _env_int("PLAN_MAX_CANDIDATES", "12")
# Log the plan to the autobiography only when the planner actually changed the
# decision by at least this much expected value — otherwise the log is noise.
PLAN_LOG_THRESHOLD = _env_float("PLAN_LOG_THRESHOLD", "0.05")

# --- M3: behaviour policy -------------------------------------------
POLICY_LR = _env_float("POLICY_LR", "0.15")
POLICY_WEIGHT = _env_float("POLICY_WEIGHT", "0.3")
POLICY_MIN_SUPPORT = _env_int("POLICY_MIN_SUPPORT", "20")
POLICY_MAX_COND = _env_int("POLICY_MAX_COND", "2")
POLICY_MINE_EVERY_N_TICKS = _env_int("POLICY_MINE_EVERY_N_TICKS", "200")
POLICY_TRIAL_TICKS = _env_int("POLICY_TRIAL_TICKS", "300")
POLICY_REVIEW_TICKS = _env_int("POLICY_REVIEW_TICKS", "1000")
POLICY_MIN_EFFECT = _env_float("POLICY_MIN_EFFECT", "0.03")
POLICY_ALPHA = _env_float("POLICY_ALPHA", "0.05")
POLICY_MAX_RULES = _env_int("POLICY_MAX_RULES", "500")
POLICY_MAX_PREFS = _env_int("POLICY_MAX_PREFS", "20000")

# --- M4: resources and priority -------------------------------------
RES_TOKENS_PER_HOUR = _env_int("RES_TOKENS_PER_HOUR", "200000")
RES_CALLS_PER_HOUR = _env_int("RES_CALLS_PER_HOUR", "400")
RES_WALL_MS_PER_TICK = _env_int("RES_WALL_MS_PER_TICK", "2500")
RES_SUBPROC_SLOTS = _env_int("RES_SUBPROC_SLOTS", "4")
RES_TRAINING_SLOTS = _env_int("RES_TRAINING_SLOTS", "1")
RES_NET_CALLS_PER_HOUR = _env_int("RES_NET_CALLS_PER_HOUR", "600")
RES_DISK_MB = _env_int("RES_DISK_MB", "4096")
# Share of every budget that safety-critical work keeps no matter how the ROI
# tracker would prefer to spend it (spec Appendix B, category 7).
RESOURCE_SAFETY_FLOOR = _env_float("RESOURCE_SAFETY_FLOOR", "0.15")
# No activity may be starved to zero: an activity with no budget never produces
# a result, so its ROI can never recover — the reallocation would be one-way.
RESOURCE_MIN_SHARE = _env_float("RESOURCE_MIN_SHARE", "0.05")
RESOURCE_REALLOC_EVERY_N_TICKS = _env_int("RESOURCE_REALLOC_EVERY_N_TICKS", "500")
PRIORITY_AGING = _env_float("PRIORITY_AGING", "0.01")
PRIORITY_AGING_MAX_TICKS = _env_int("PRIORITY_AGING_MAX_TICKS", "2000")

# --- M5: population evolution ---------------------------------------
# Whether the tick may start evolution generations at all.
#
# A generation evaluates ten variants, each of which builds a fresh system in
# another process and runs a benchmark and a rollout in it. That is the right
# amount of work for a long-lived deployment and far too much to start by
# accident: any long-running test that ticks past the interval would otherwise
# spawn one, and a suite that runs real generations is measuring the machine
# rather than the code.
EVO_ENABLED = os.environ.get("AEGIS_EVO_ENABLED", "1") == "1"
EVO_POP_SIZE = _env_int("EVO_POP_SIZE", "10")
EVO_SIGMA = _env_float("EVO_SIGMA", "0.15")
EVO_EPSILON = _env_float("EVO_EPSILON", "0.005")
EVO_MIN_DISTANCE = _env_float("EVO_MIN_DISTANCE", "0.02")
EVO_EVERY_N_TICKS = _env_int("EVO_EVERY_N_TICKS", "250")
EVO_SPLIT_ROTATE_EVERY = _env_int("EVO_SPLIT_ROTATE_EVERY", "5")
EVO_WATCH_TICKS = _env_int("EVO_WATCH_TICKS", "500")
EVO_ROLLBACK_DELTA = _env_float("EVO_ROLLBACK_DELTA", "0.03")
EVO_COST_PENALTY = _env_float("EVO_COST_PENALTY", "0.1")
EVO_LATENCY_PENALTY = _env_float("EVO_LATENCY_PENALTY", "0.05")
EVO_MAX_ARCHIVE = _env_int("EVO_MAX_ARCHIVE", "2000")

# --- M6: reasoning ---------------------------------------------------
REASON_MAX_STEPS = _env_int("REASON_MAX_STEPS", "24")
REASON_MIN_GAIN = _env_float("REASON_MIN_GAIN", "0.05")
REASON_COST_TOLERANCE = _env_float("REASON_COST_TOLERANCE", "1.5")
REASON_TRIAL_N = _env_int("REASON_TRIAL_N", "50")
REASON_SCAN_EVERY_N_TICKS = _env_int("REASON_SCAN_EVERY_N_TICKS", "300")
REASON_UCB_C = _env_float("REASON_UCB_C", "1.4")
REASON_MAX_TRACES = _env_int("REASON_MAX_TRACES", "5000")

# --- M7: discovery ---------------------------------------------------
DISC_MAX_LAG = _env_int("DISC_MAX_LAG", "5")
DISC_MAX_DEPTH = _env_int("DISC_MAX_DEPTH", "4")
DISC_ALPHA = _env_float("DISC_ALPHA", "0.05")
DISC_MIN_N = _env_int("DISC_MIN_N", "100")
DISC_BLOCK_TICKS = _env_int("DISC_BLOCK_TICKS", "100")
DISC_LAW_REPS = _env_int("DISC_LAW_REPS", "3")
DISC_SCAN_EVERY_N_TICKS = _env_int("DISC_SCAN_EVERY_N_TICKS", "1000")
DISC_INTERVENTION_ENABLED = _env_str("DISC_INTERVENTION_ENABLED", "1") == "1"
DISC_INTERVENTION_MAX_DELTA = _env_float("DISC_INTERVENTION_MAX_DELTA", "0.2")
DISC_MAX_HYPOTHESES = _env_int("DISC_MAX_HYPOTHESES", "2000")

# --- M8: cortex ------------------------------------------------------
# role -> failover chain. The expensive roles go to hosted APIs; FAST, which
# fires on nearly every LLM tick, is served locally so it costs no API tokens.
CORTEX_ROUTES_DEFAULT = {
    "deep": ["kimi", "claude", "local_openai"],
    "code": ["kimi", "claude"],
    "judge": ["claude", "kimi"],
    "fast": ["local_openai", "kimi"],
}
CORTEX_ROUTES = _env_json("CORTEX_ROUTES", CORTEX_ROUTES_DEFAULT)
CORTEX_DETERMINISTIC = _env_str("CORTEX_DETERMINISTIC", "1") == "1"
CORTEX_CACHE_TTL = _env_float("CORTEX_CACHE_TTL", "3600")
CORTEX_CACHE_MAX = _env_int("CORTEX_CACHE_MAX", "2000")
CORTEX_BREAKER_ERRORS = _env_int("CORTEX_BREAKER_ERRORS", "5")
CORTEX_BREAKER_COOLDOWN = _env_float("CORTEX_BREAKER_COOLDOWN", "300")
# One repair round-trip when the model answers with malformed JSON, then give
# up and fall back to the deterministic path (§M8.5).
CORTEX_MAX_REPAIRS = _env_int("CORTEX_MAX_REPAIRS", "1")

# Providers. Every model identifier is an environment variable, so moving to a
# newer generation of any family never touches code (§M8.3b).
KIMI_API_KEY = _env_str("KIMI_API_KEY")
KIMI_BASE_URL = _env_str("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
KIMI_MODEL = _env_str("KIMI_MODEL", "kimi-k2-0905-preview")
OPENAI_API_KEY = _env_str("OPENAI_API_KEY")
OPENAI_BASE_URL = _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = _env_str("OPENAI_MODEL", "gpt-4o-mini")
LOCAL_OPENAI_BASE_URL = _env_str("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
LOCAL_OPENAI_MODEL = _env_str("LOCAL_OPENAI_MODEL", "")
LOCAL_OPENAI_API_KEY = _env_str("LOCAL_OPENAI_API_KEY", "ollama")

# --- M9: infrastructure ---------------------------------------------
EVAL_POOL_WORKERS = _env_int("EVAL_POOL_WORKERS", "4")
EVAL_POOL_TASK_TIMEOUT = _env_float("EVAL_POOL_TASK_TIMEOUT", "120")
