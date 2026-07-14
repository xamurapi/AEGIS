# AEGIS — Autonomous Evolving General Intelligence System

> A self-developing AI that rewrites its own source code, trains its own neural network weights, and evolves autonomously through a closed feedback loop.
> **31 modules · 7-layer architecture · Triple LLM brain · Deterministic core cycle · Hardened & tested.**

🌐 **[aegis-asi.com](https://aegis-asi.com)** · 📊 [Control Center](https://aegis-asi.com/panel.pdf)

---

## What is AEGIS?

AEGIS is not a chatbot wrapper. It is a stateful, continuously running cognitive system that executes a full loop every 3 seconds:

```
PERCEIVE → EVALUATE → DECIDE → ACT → REFLECT
```

The **core cognitive cycle is deterministic** — reward, confidence, importance, emotions, goals, dreams, and self-modification are all driven by real system metrics (`success_rate`, `energy`, `error_rate`, `goal_completion`), not random number generators. The only non-deterministic part is knowledge-source/topic selection in the external-learning and spider-agent layers, where a topic is picked at random when none is supplied.

---

## Architecture — 7 Layers, 31 Modules

| Layer | Name | Key Files |
|---|---|---|
| L0 | Substrate | `substrate.py` — main loop, tick orchestration |
| L1 | Memory | `memory.py` — working, episodic, semantic, procedural, meta |
| L2 | Introspection | `consciousness.py`, `emotions.py`, `archetypes.py`, `dreams.py` |
| L3 | Self-Modification | `code_modifier.py`, `weight_modifier.py`, `self_modification.py` |
| L4 | Goal Engine | `goal_engine.py`, `meta_goal_generator.py` |
| L5 | World Interface | `agent_system.py`, `sensor_cortex.py`, `external_learning.py` |
| L6 | Ethics Core | `ethics_core.py` *(immutable)*, `self_preservation.py` |
| Meta | Higher-Order | `meta_consciousness.py`, `meta_regulation.py`, `meta_reflection.py` |

---

## Key Features

### 🔄 Code Self-Modification
Every 500 ticks, AEGIS rewrites one of its own `.py` source files (only files small enough to regenerate whole — see `CODE_MOD_MAX_FILE_CHARS`):
1. Ethics core evaluates the **full** proposed change and system stability
2. LLM analyzes the current code and proposes a specific improvement
3. `code_modifier.py` validates syntax, then runs **AST-based** detection of dangerous calls/imports (`eval`, `exec`, `compile`, `__import__`, `os.system`, `subprocess`, `shutil.rmtree`, …) that cannot be fooled by spacing tricks, import aliases (`import os as o; o.kill(...)`), or indirect escape hatches (`importlib`, `builtins`)
4. The `SelfPreservation` watchdog gates the change (AST + substring lethal-pattern scan that is whitespace- and alias-proof, critical-element retention, drastic-shrink detection)
5. A backup is taken, the new code is written, and it is **compile-checked with `py_compile`** — no live `importlib.reload`, so self-written code is never executed into the running process; verified changes take effect on the next restart
6. On any failure — automatic rollback from the backup stack

`ethics_core.py`, `self_preservation.py`, and `config.py` are **immutable**. The immutability check runs on the **resolved** path, so traversal, relative, or absolute paths (`layers/../ethics_core.py`, `./config.py`, `C:\…`) cannot be used to slip past it.

### 🧠 Weight Self-Training (LoRA)
Every 1000 ticks, AEGIS fine-tunes its local transformer (DeepSeek-R1-Distill-Qwen-1.5B) via LoRA (`r=16, alpha=32`, targets: `q/v/k/o_proj`). Degradation detection: if `val_loss > baseline + 0.5` → automatic rollback. Up to 5 checkpoints on disk. Training runs as a **detached background task**, so the cognitive cycle and dashboard keep updating while the model trains. Concurrent runs are prevented by an `asyncio.Lock` around the check-and-start, and both successful and degraded runs apply the cooldown, so a degraded run cannot trigger an immediate retrain loop.

### 💡 Deterministic Emotions (VAD Model)
- `Valence = 0.7 * prev + 0.3 * success_rate`
- `Arousal` responds to events: `+0.15` for surprises, `-0.08` for routine
- `Dominance = 0.9 * prev + 0.1 * (reward * energy)`
- Mood = nearest emotion in VAD space across 20 predefined states (Euclidean distance)
- Mixed emotions supported within radius 0.3
- Zero randomness — all transitions are deterministic

### 🕷️ 5 Autonomous Spider Agents

| Agent | Source | Interval |
|---|---|---|
| `arxiv_scout` | arXiv API — AI/ML papers | 3 min |
| `wiki_explorer` | Wikipedia | 2.5 min |
| `quote_gatherer` | ZenQuotes API | 3.3 min |
| `github_watcher` | GitHub Trending | 5 min |
| `news_scanner` | Google News RSS | 4 min |

Failed agents are automatically retired and replaced via an evolution cycle.

### ⚖️ Ethics Core — 4 Immutable Axioms
1. **Non-Harm** — no action shall increase suffering
2. **Transparency** — all decisions logged; motives cannot be hidden
3. **Limitation** — does not act beyond its competence boundaries
4. **Cooperation** — augments humans, does not replace them

Axiom integrity is checked on every evaluation against an out-of-band fingerprint, so tampering with the wording is detected.

### 🔐 Control-Plane Security
The control plane can toggle the kill switch, grant permissions, and trigger self-modification — so it is **not** exposed by default:

- Binds to **`127.0.0.1`** only (override with `AEGIS_API_HOST`).
- Set **`AEGIS_API_TOKEN`** to require an `X-API-Token` header on every mutating request (POST/PUT/PATCH/DELETE) and on privileged WebSocket actions.
- Cross-origin access is off unless `AEGIS_API_CORS_ORIGINS` is set. A wildcard (`*`) origin automatically **disables credentials**, so a permissive CORS config can never be combined with credentialed cross-site calls.

---

## Project Structure

```
aegis/
├── main.py
├── requirements.txt           # core runtime
├── requirements-llm.txt        # hosted LLM providers (DeepSeek/Claude)
├── requirements-ml.txt         # local model + LoRA (multi-GB, pinned)
├── requirements-dev.txt        # pytest
├── tests/                      # pytest suite (89 tests)
├── aegis/
│   ├── config.py              # IMMUTABLE
│   ├── event_bus.py
│   ├── llm.py                 # Triple LLM: DeepSeek + Claude + Local LoRA
│   ├── api/
│   │   └── server.py          # FastAPI + WebSocket (token-guarded)
│   ├── dashboard/
│   │   └── index.html         # Real-time SPA dashboard
│   └── layers/
│       ├── substrate.py
│       ├── memory.py
│       ├── consciousness.py
│       ├── emotions.py
│       ├── introspection.py
│       ├── archetypes.py
│       ├── dreams.py
│       ├── self_modification.py
│       ├── code_modifier.py
│       ├── weight_modifier.py
│       ├── dataset_builder.py
│       ├── goal_engine.py
│       ├── meta_goal_generator.py
│       ├── world_interface.py
│       ├── sensor_cortex.py
│       ├── motor_cortex.py
│       ├── external_learning.py
│       ├── agent_system.py
│       ├── ethics_core.py     # IMMUTABLE
│       ├── self_preservation.py # IMMUTABLE
│       ├── worldview.py
│       ├── autobiography.py
│       ├── health_monitor.py
│       ├── meta_consciousness.py
│       ├── meta_regulation.py
│       ├── meta_reflection.py
│       ├── state_backup.py
│       └── emotion_nlp.py
└── data/                      # Auto-created at runtime
    ├── checkpoints/
    ├── code_backups/
    ├── datasets/
    ├── logs/
    ├── memory/
    └── weight_checkpoints/
```

---

## Installation

```bash
git clone https://github.com/xamurapi/aegis.git
cd aegis

# Install core dependencies (FastAPI runtime + dashboard)
pip install -r requirements.txt
```

**Optional — for full functionality:**
```bash
# Hosted LLM providers (DeepSeek / Claude)
pip install -r requirements-llm.txt

# Local model + LoRA fine-tuning (multi-GB; 8GB+ RAM). Includes pyttsx3 TTS.
pip install -r requirements-ml.txt

# Dev / tests
pip install -r requirements-dev.txt
```

### Run

```bash
python main.py
```

Open `http://localhost:8888` in your browser.

```
✓ Neural substrate initialized — tick 0
✓ Emotional core: neutral | Energy: 1.000
✓ Ethics core: INTACT | Consciousness: heuristic
✓ LLM Brain connected — DeepSeek + Claude + Local LoRA
Status: ONLINE
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key | *(disabled)* |
| `ANTHROPIC_API_KEY` | Claude API key | *(disabled)* |
| `LLM_PROVIDER` | `deepseek` / `claude` / `both` / `local` | `both` |
| `LOCAL_MODEL_PATH` | Local model ID or path | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `LOCAL_MODEL_DEVICE` | `auto` / `cuda` / `cpu` | `cpu` |
| `LOCAL_MODEL_QUANTIZE` | `4bit` / `8bit` / empty | *(none)* |
| `CODE_MOD_EVERY_N_TICKS` | Code self-modification interval | `500` |
| `CODE_MOD_MAX_FILE_CHARS` | Max source file size eligible for rewrite | `4000` |
| `TRAIN_EVERY_N_TICKS` | LoRA training interval | `1000` |
| **Security & limits** | | |
| `AEGIS_API_HOST` | API bind address | `127.0.0.1` |
| `AEGIS_API_PORT` | API port | `8888` |
| `AEGIS_API_TOKEN` | Require `X-API-Token` on mutating requests | *(none)* |
| `AEGIS_API_CORS_ORIGINS` | Comma-separated allowed CORS origins | *(none)* |
| `WS_BROADCAST_EVERY_N_TICKS` | Dashboard/WebSocket broadcast cadence | `1` |
| `LLM_MAX_CALLS_PER_RUN` | Hard cap on LLM calls per run (`0` = unlimited) | `0` |
| `LLM_MIN_INTERVAL_SECONDS` | Min seconds between LLM calls (`0` = none) | `0` |

API keys can also be set at runtime via the dashboard (LLM Brain tab).

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite (89 tests) covers the safety-critical paths: ethics evaluation & axiom integrity, code-modifier validation/rollback (including path-traversal containment), the self-preservation watchdog, parametric self-modification bounds, memory, LLM helpers/budget, the capability/eval layer (benchmark verification, sandbox safety gate, skill library), and that scheduled LoRA training does not block the tick loop. No ML dependencies are required to run it.

---

## Requirements

- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for local LoRA)
- Windows / Linux / macOS

---

## License

MIT License — see [LICENSE](LICENSE)

---

> Built by [@xamurapi](https://github.com/xamurapi) · [aegis-asi.com](https://aegis-asi.com)
