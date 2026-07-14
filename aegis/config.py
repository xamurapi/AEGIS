"""AEGIS Configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LOGS_DIR = DATA_DIR / "logs"
MEMORY_DIR = DATA_DIR / "memory"

for d in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, MEMORY_DIR]:
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
API_PORT = int(os.environ.get("AEGIS_API_PORT", "8888"))
# Shared-secret token for mutating (POST) endpoints. When set, clients must send
# it in the `X-API-Token` header. Empty string = no token required (only safe
# together with the loopback bind above).
API_TOKEN = os.environ.get("AEGIS_API_TOKEN", "")
# Allowed CORS origins (comma-separated). Empty = no cross-origin access.
API_CORS_ORIGINS = [o.strip() for o in os.environ.get("AEGIS_API_CORS_ORIGINS", "").split(",") if o.strip()]

# DeepSeek LLM
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_ENABLED = True
LLM_THINK_EVERY_N_TICKS = 3  # use LLM every N ticks to save tokens
LLM_MAX_TOKENS = 5000
LLM_TEMPERATURE = 0.7

# Claude (Anthropic)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-6"

# LLM Provider: "deepseek", "claude", "both", "local" (local transformers model)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "both")

# --- Local model (transformers) ---
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
LOCAL_MODEL_DEVICE = os.environ.get("LOCAL_MODEL_DEVICE", "cpu")  # "auto", "cuda", "cpu"
LOCAL_MODEL_DTYPE = os.environ.get("LOCAL_MODEL_DTYPE", "float32")  # "float16", "bfloat16", "float32"
LOCAL_MODEL_QUANTIZE = os.environ.get("LOCAL_MODEL_QUANTIZE", "")  # "4bit", "8bit", "" (none)
LOCAL_MODEL_MAX_LENGTH = int(os.environ.get("LOCAL_MODEL_MAX_LENGTH", "2048"))

# --- Weight modification / LoRA fine-tuning ---
WEIGHT_CHECKPOINTS_DIR = DATA_DIR / "weight_checkpoints"
WEIGHT_DATASETS_DIR = DATA_DIR / "datasets"
WEIGHT_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LORA_R = int(os.environ.get("LORA_R", "16"))              # LoRA rank
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))       # LoRA alpha
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "1"))
TRAIN_GRADIENT_ACCUMULATION = int(os.environ.get("TRAIN_GRADIENT_ACCUMULATION", "4"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "3"))
TRAIN_LEARNING_RATE = float(os.environ.get("TRAIN_LEARNING_RATE", "2e-4"))
TRAIN_MAX_SAMPLES = int(os.environ.get("TRAIN_MAX_SAMPLES", "500"))
TRAIN_VAL_SPLIT = float(os.environ.get("TRAIN_VAL_SPLIT", "0.1"))

# Safety limits
TRAIN_MIN_INTERVAL_SECONDS = int(os.environ.get("TRAIN_MIN_INTERVAL", "3600"))  # 1 hour minimum
TRAIN_MAX_CHECKPOINTS = int(os.environ.get("TRAIN_MAX_CHECKPOINTS", "5"))
TRAIN_EVERY_N_TICKS = int(os.environ.get("TRAIN_EVERY_N_TICKS", "1000"))
TRAIN_MIN_DATASET_SIZE = int(os.environ.get("TRAIN_MIN_DATASET_SIZE", "50"))
TRAIN_VAL_LOSS_THRESHOLD = float(os.environ.get("TRAIN_VAL_LOSS_THRESHOLD", "0.5"))  # max acceptable val loss increase

# --- Code Self-Modification ---
CODE_BACKUPS_DIR = DATA_DIR / "code_backups"
CODE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
CODE_MOD_EVERY_N_TICKS = int(os.environ.get("CODE_MOD_EVERY_N_TICKS", "500"))  # attempt code mod every N ticks
CODE_MOD_MIN_TICK = int(os.environ.get("CODE_MOD_MIN_TICK", "100"))  # don't modify code before this tick
CODE_MOD_MAX_PER_SESSION = int(os.environ.get("CODE_MOD_MAX_PER_SESSION", "10"))  # max code mods per run
# Only attempt whole-file LLM rewrites on files at or below this size (chars).
# The model must return the entire file, so larger files can't be regenerated
# reliably within the modification-size cap.
CODE_MOD_MAX_FILE_CHARS = int(os.environ.get("CODE_MOD_MAX_FILE_CHARS", "4000"))

# --- Capability layer: eval harness, skill library, sandbox, environment ---
EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
# Periodic held-out benchmark run (the fitness graph). Detached, non-blocking.
EVAL_EVERY_N_TICKS = int(os.environ.get("EVAL_EVERY_N_TICKS", "50"))
# Live environment step: agent attempts one real task and gets real reward.
ENV_STEP_EVERY_N_TICKS = int(os.environ.get("ENV_STEP_EVERY_N_TICKS", "2"))
# Skill synthesis: propose+sandbox-test a new skill for a failing task kind.
SKILL_SYNTH_EVERY_N_TICKS = int(os.environ.get("SKILL_SYNTH_EVERY_N_TICKS", "200"))
# Hard timeout (seconds) for a single sandboxed skill execution.
SANDBOX_TIMEOUT = float(os.environ.get("SANDBOX_TIMEOUT", "3.0"))

# Broadcast the (large) full status over WebSocket at most every N ticks.
WS_BROADCAST_EVERY_N_TICKS = int(os.environ.get("WS_BROADCAST_EVERY_N_TICKS", "1"))

# --- LLM call budget (per process run) ---
# Hard ceiling on LLM calls per run to bound token spend; 0 = unlimited.
LLM_MAX_CALLS_PER_RUN = int(os.environ.get("LLM_MAX_CALLS_PER_RUN", "0"))
# Minimum seconds between any two LLM calls (simple rate limit); 0 = none.
LLM_MIN_INTERVAL_SECONDS = float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", "0"))
