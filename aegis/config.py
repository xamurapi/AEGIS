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

for d in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, MEMORY_DIR,
          WORLD_MODEL_DIR, COGNITIVE_GRAPH_DIR, EVOLUTION_DIR,
          GOAL_INTEL_DIR, FEEDBACK_DIR]:
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

# --- Local model (transformers) ---
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LOCAL_MODEL_DEVICE = os.environ.get("LOCAL_MODEL_DEVICE", "cpu")  # "auto", "cuda", "cpu"
LOCAL_MODEL_DTYPE = os.environ.get("LOCAL_MODEL_DTYPE", "float32")  # "float16", "bfloat16", "float32"
LOCAL_MODEL_QUANTIZE = os.environ.get("LOCAL_MODEL_QUANTIZE", "")  # "4bit", "8bit", "" (none)
LOCAL_MODEL_MAX_LENGTH = _env_int("LOCAL_MODEL_MAX_LENGTH", "2048")

# --- Weight modification / LoRA fine-tuning ---
WEIGHT_CHECKPOINTS_DIR = DATA_DIR / "weight_checkpoints"
WEIGHT_DATASETS_DIR = DATA_DIR / "datasets"
WEIGHT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LORA_R = _env_int("LORA_R", "16")              # LoRA rank
LORA_ALPHA = _env_int("LORA_ALPHA", "32")       # LoRA alpha
LORA_DROPOUT = _env_float("LORA_DROPOUT", "0.05")
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

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

# Broadcast the (large) full status over WebSocket at most every N ticks.
WS_BROADCAST_EVERY_N_TICKS = _env_int("WS_BROADCAST_EVERY_N_TICKS", "1")

# --- LLM call budget (per process run) ---
# Hard ceiling on LLM calls per run to bound token spend; 0 = unlimited.
LLM_MAX_CALLS_PER_RUN = _env_int("LLM_MAX_CALLS_PER_RUN", "0")
# Minimum seconds between any two LLM calls (simple rate limit); 0 = none.
LLM_MIN_INTERVAL_SECONDS = _env_float("LLM_MIN_INTERVAL_SECONDS", "0")
