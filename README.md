# AEGIS — Autonomous Evolving General Intelligence System

> A self-developing AI that rewrites its own source code, trains its own neural network weights, and evolves autonomously through a closed feedback loop.
> **31 modules · 7-layer architecture · Triple LLM brain · Deterministic emotions · Zero randomness.**

🌐 **[aegis-asi.com](https://aegis-asi.com)** · 📊 [Control Center](https://aegis-asi.com/panel.pdf)

---

## What is AEGIS?

AEGIS is not a chatbot wrapper. It is a stateful, continuously running cognitive system that executes a full loop every 3 seconds:

```
PERCEIVE → EVALUATE → DECIDE → ACT → REFLECT
```

Every decision is **100% deterministic** — no random number generators anywhere. Emotions, goals, dreams, and self-modification are all driven by real system metrics: `success_rate`, `energy`, `error_rate`, `goal_completion`.

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
Every 500 ticks, AEGIS rewrites its own `.py` source files:
1. Ethics core evaluates the target file and system stability
2. LLM analyzes the current code and proposes a specific improvement
3. `code_modifier.py` validates via AST parsing, blocks dangerous patterns (`eval`, `exec`, `subprocess`, `os.system`), creates a backup, writes new code, tests import
4. On any failure — automatic rollback from backup stack

`ethics_core.py` and `config.py` are **immutable** — the system cannot modify them.

### 🧠 Weight Self-Training (LoRA)
Every 1000 ticks, AEGIS fine-tunes its local transformer (DeepSeek-R1-Distill-Qwen-1.5B) via LoRA (`r=16, alpha=32`, targets: `q/v/k/o_proj`). Degradation detection: if `val_loss > baseline + 0.5` → automatic rollback. Up to 5 checkpoints on disk.

### 💡 Deterministic Emotions (VAD Model)
- `Valence = 0.7 * prev + 0.3 * success_rate`
- `Arousal` responds to events: `+0.15` for surprises, `-0.08` for routine
- `Dominance = 0.9 * prev + 0.1 * (reward * energy)`
- Mood = nearest emotion in VAD space across 16 predefined states (Euclidean distance)
- Mixed emotions supported within radius 0.3
- Zero randomness — all transitions are deterministic

### 🕷️ 5 Autonomous Spider Agents

| Agent | Source | Interval |
|---|---|---|
| `arxiv_scout` | arXiv API — AI/ML papers | 3 min |
| `wiki_explorer` | Wikipedia | 2.5 min |
| `quote_gatherer` | Quotable API | 3.3 min |
| `github_watcher` | GitHub Trending | 5 min |
| `news_scanner` | Google News RSS | 4 min |

Failed agents are automatically retired and replaced via an evolution cycle.

### ⚖️ Ethics Core — 4 Immutable Axioms
1. **Non-Harm** — no action shall increase suffering
2. **Transparency** — all decisions logged; motives cannot be hidden
3. **Limitation** — does not act beyond its competence boundaries
4. **Cooperation** — augments humans, does not replace them

---

## Project Structure

```
aegis/
├── main.py
├── requirements.txt
├── aegis/
│   ├── config.py              # IMMUTABLE
│   ├── event_bus.py
│   ├── llm.py                 # Triple LLM: DeepSeek + Claude + Local LoRA
│   ├── api/
│   │   └── server.py          # FastAPI + WebSocket
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
│       ├── self_preservation.py
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

# Install core dependencies
pip install -r requirements.txt
```

**Optional — for full functionality:**
```bash
# Local model + LoRA fine-tuning (requires 8GB+ RAM)
pip install torch transformers peft bitsandbytes accelerate

# Text-to-speech
pip install pyttsx3
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
| `TRAIN_EVERY_N_TICKS` | LoRA training interval | `1000` |

API keys can also be set at runtime via the dashboard (LLM Brain tab).

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
