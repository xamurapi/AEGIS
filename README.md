# AEGIS — Autonomous Evolving General Intelligence System

[![tests](https://img.shields.io/badge/tests-1363%20passing-2ea44f)](#testing--quality)
[![coverage](https://img.shields.io/badge/branch%20coverage-92%25-2ea44f)](#testing--quality)
[![mutation score](https://img.shields.io/badge/mutation%20score-99.8%25-2ea44f)](#testing--quality)
[![audit](https://img.shields.io/badge/audit-3%20rounds-blue)](docs/%D0%90%D0%A3%D0%94%D0%98%D0%A2.md)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> A self-developing AI that rewrites its own source code, trains its own neural network weights, and evolves autonomously through a closed feedback loop.
> **36 modules · 7-layer architecture + 5 higher-order systems · Triple LLM brain · Deterministic core cycle.**
> **1363 tests · 92% branch coverage · 99.8% mutation score · 3 audit rounds.**
> **Safe defaults:** source self-rewriting is opt-in (`AEGIS_CODE_SELF_MOD_ENABLED=1`), the control plane binds to `127.0.0.1`, and self-written skills run only in a child-process sandbox.

🌐 **[aegis-asi.com](https://aegis-asi.com)** · 📊 [Control Center](https://aegis-asi.com/panel.pdf)

---

## What is AEGIS?

AEGIS is not a chatbot wrapper. It is a stateful, continuously running cognitive system that executes a full loop every 3 seconds:

```
PERCEIVE → EVALUATE → DECIDE → ACT → REFLECT
```

The **core cognitive cycle is deterministic** — reward, confidence, importance, emotions, goals, dreams, and self-modification are all driven by real system metrics (`success_rate`, `energy`, `error_rate`, `goal_completion`), not random number generators. The only non-deterministic part is knowledge-source/topic selection in the external-learning and spider-agent layers, where a topic is picked at random when none is supplied.

On top of that cycle run five higher-order systems that give it a causal model of
the world, a connected knowledge graph, benchmark-gated evolution, value-driven
motivation, and a closed loop from real outcomes back into training data — see
[below](#the-five-higher-order-systems).

---

## Architecture — 7 Layers, 36 Modules

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

## The Five Higher-Order Systems

Layers L0–L6 make AEGIS *self-aware*. These five make it *effective in the world* —
they close the gap between "the model talks about itself" and "the system reasons
about causes, keeps only changes that measurably help, and learns from real results."

| # | System | File | What it does |
|---|---|---|---|
| 1 | **World Model** | `world_model.py` | Learns cause→effect links from observed outcomes (Laplace-smoothed frequencies) and builds causal chains: `objective → constraints → risks → plan → expected result`. Answers "what tends to fail around this topic" from memory, not from text generation. |
| 2 | **Cognitive Graph** | `cognitive_graph.py` | Typed graph of knowledge and experience (`concept / event / skill / goal / outcome`, edges `causes / requires / learned_from / led_to`). Gives path finding, relevance and centrality, so reasoning uses **connected** knowledge instead of a flat recency list. |
| 3 | **Evolution Engine** | `evolution_engine.py` | Natural selection over its own parameters: `champion → mutation → candidate → held-out benchmark → keep only if better`. This is what makes self-modification ≠ self-*improvement* an enforced distinction: a change survives only when an external, verifiable metric says so. |
| 4 | **Goal Intelligence** | `goal_intelligence.py` | Turns internal state into motivation: `goal → value → action choice → reward`. Keeps a learned utility per objective, updated from realized reward, and picks what to pursue by expected value under four intrinsic drives (competence, knowledge, coherence, stability). |
| 5 | **Feedback Loop** | `feedback_loop.py` | Closes the experience loop: `situation → decision → real result → evaluation → cause → new experience`. Every row is structured (it records **why** an outcome happened), so the LoRA dataset carries causes rather than raw text. |

All five are deterministic and dependency-free; an LLM may refine a plan, but is
never required for them to function. They persist atomically and are bounded in
size, so no structure grows without limit.

### Where each one changes behaviour

A learning system that only accumulates is a logger. Four of these five recorded
knowledge that nothing read — the chain `action → result → evaluation → new
knowledge` stopped one step short of `→ behaviour change`. This is the step that
closes it, and each row below is pinned by a test in
[`tests/test_behavior_closure.py`](tests/test_behavior_closure.py) that fails if
the wiring is removed:

| System | What it now changes |
|---|---|
| World Model | A course of action with an observed failure history is proposed with **lower confidence** — and confidence feeds the ethics gate. Capped by `MAX_RISK_CONFIDENCE_PENALTY` so remembered failure cannot stall the system. |
| Cognitive Graph | The **next learning topic** is the concept most connected to the current focus; the flat recency slice is only the fallback. |
| Goal Intelligence | Learned utility **selects the decision**, before ethics judges it and before the LLM may override it — so on non-LLM ticks motivation alone steers the action. |
| Feedback Loop | The experience log **reaches the LoRA dataset**, which is the only training source carrying *why* an outcome happened. |
| Evolution Engine | Already closed by construction (benchmark-gated) — see the limitation below. |

> **Known limitation, stated plainly.** The Evolution Engine's *mechanism* is
> sound — mutate, benchmark, keep or revert, with the champion never updated from
> self-report. But the parameters it currently mutates (`learning_rate`,
> `attention_heads`, …) are simulated knobs that do not causally drive the skill
> benchmark, so selection currently rides on benchmark drift from other changes.
> Making the loop meaningful requires wiring the genome to something that
> measurably moves the metric. This is an architectural gap, not a bug, and it is
> tracked in [`docs/АУДИТ.md`](docs/АУДИТ.md).

---

## Key Features

### 🔄 Code Self-Modification — **opt-in, off by default**

Autonomous source rewriting is **disabled unless you set `AEGIS_CODE_SELF_MOD_ENABLED=1`**.
The reason is a real attack path, not caution theatre: untrusted external content
(web fetches, agent feeds) flows into semantic memory and from there into LLM
prompts — while the *same* LLM channel proposes whole-file rewrites that get
written to disk and executed after restart. That is indirect prompt injection to
arbitrary code execution, and no pattern/AST blocklist closes it fully. So it
requires an explicit operator decision (audit finding C2). Parametric
self-modification and LoRA weight training are unaffected and stay on.

When enabled, every 500 ticks AEGIS rewrites one of its own `.py` source files
(only files small enough to regenerate whole — see `CODE_MOD_MAX_FILE_CHARS`):
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

- Binds to **`127.0.0.1`** only (override with `AEGIS_API_HOST`). Binding elsewhere without a token logs a loud warning at import.
- Set **`AEGIS_API_TOKEN`** to require an `X-API-Token` header on every mutating request (POST/PUT/PATCH/DELETE) and on privileged WebSocket actions. Comparison is constant-time.
- **Only authenticated WebSocket clients join the state broadcast.** The handshake check alone was not enough: an unauthorized socket used to stay in the fan-out list and receive the periodic `full_status()` push, so simply connecting and waiting leaked internal state past the token gate (audit R3-2).
- WebSockets are not covered by CORS, so the `Origin` header is checked **before** the handshake is accepted — a hostile page cannot open `ws://127.0.0.1:8888/ws` (CSWSH).
- Cross-origin access is off unless `AEGIS_API_CORS_ORIGINS` is set. A wildcard (`*`) origin automatically **disables credentials**, so a permissive CORS config can never be combined with credentialed cross-site calls.

**Using the dashboard with a token.** Open it as `http://127.0.0.1:8888/?token=YOUR_TOKEN`.
The token is remembered in `localStorage` and attached to every `/api` request;
an unauthorized session shows an explicit banner instead of silently freezing.

### 🧪 Sandboxed skill execution

Self-written skills never run in the main process: they are AST-checked against an
allowlist and then executed via `python -I` in a child process under a hard
wall-clock timeout. The static gate inspects **everything Python evaluates at
definition time** — decorators, default values, keyword-only defaults, and
parameter/return **annotations**. That last one matters: an unchecked annotation
(`def solve(p, _z: __import__('os').system('...'))`) passed the old gate and then
executed arbitrary code in the child (audit R3-1, fixed and proven by test).

This is defense-in-depth, **not** an OS-level jail. Two escapes were found in this
layer across two audit rounds, which is itself the argument: a production
deployment should add a container/seccomp sandbox with network egress blocked.

---

## Project Structure

```
AEGIS/
├── main.py
├── requirements.txt           # core runtime
├── requirements-llm.txt        # hosted LLM providers (DeepSeek/Claude)
├── requirements-ml.txt         # local model + LoRA (multi-GB, pinned)
├── requirements-dev.txt        # pytest, pytest-bdd, coverage, httpx
├── docs/
│   ├── АУДИТ.md               # 3 audit rounds — every finding, fix and test
│   ├── QA.md                  # QA procedures, quality metrics, gates
│   └── СИСТЕМЫ.md             # the five higher-order systems in detail
├── scripts/
│   └── mutation_test.py       # dependency-free mutation-testing harness
├── tests/                      # 1363 tests
│   └── features/              # executable Gherkin specifications
├── aegis/
│   ├── config.py              # IMMUTABLE
│   ├── event_bus.py
│   ├── llm.py                 # Triple LLM: DeepSeek + Claude + Local LoRA
│   ├── api/
│   │   └── server.py          # FastAPI + WebSocket (token-guarded)
│   ├── dashboard/
│   │   └── index.html         # Real-time SPA dashboard
│   ├── eval/                  # benchmark, skill library, isolated sandbox
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
│       ├── world_model.py     # System 1 — causal model
│       ├── cognitive_graph.py # System 2 — knowledge graph
│       ├── evolution_engine.py# System 3 — benchmark-gated evolution
│       ├── goal_intelligence.py # System 4 — value-driven motivation
│       ├── feedback_loop.py   # System 5 — learning from real outcomes
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
git clone https://github.com/xamurapi/AEGIS.git
cd AEGIS

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
| `CLAUDE_MODEL` | Claude model ID | `claude-sonnet-5` |
| `LLM_TIMEOUT_SECONDS` | Per-call LLM timeout | `60` |
| `LOCAL_MODEL_PATH` | Local model ID or path | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `LOCAL_MODEL_DEVICE` | `auto` / `cuda` / `cpu` | `cpu` |
| `LOCAL_MODEL_DTYPE` | `float16` / `bfloat16` / `float32` | `float32` |
| `LOCAL_MODEL_QUANTIZE` | `4bit` / `8bit` / empty | *(none)* |
| **Self-modification** | | |
| `AEGIS_CODE_SELF_MOD_ENABLED` | **Master switch for source self-rewriting** (`1` = on) | `0` *(off)* |
| `CODE_MOD_EVERY_N_TICKS` | Code self-modification interval | `500` |
| `CODE_MOD_MIN_TICK` | No source rewrite before this tick | `100` |
| `CODE_MOD_MAX_PER_SESSION` | Hard cap on source rewrites per run | `10` |
| `CODE_MOD_MAX_FILE_CHARS` | Max source file size eligible for rewrite | `4000` |
| **LoRA training** | | |
| `TRAIN_EVERY_N_TICKS` | LoRA training interval | `1000` |
| `TRAIN_MIN_INTERVAL` | Min seconds between training runs (cooldown) | `3600` |
| `TRAIN_MIN_DATASET_SIZE` | Min examples required to start training | `50` |
| `TRAIN_VAL_LOSS_THRESHOLD` | Val-loss rise that triggers rollback | `0.5` |
| `TRAIN_MAX_CHECKPOINTS` | Weight checkpoints kept on disk | `5` |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | LoRA rank / alpha / dropout | `16` / `32` / `0.05` |
| **Higher-order systems** | | |
| `WORLD_MODEL_EVERY_N_TICKS` | Observe cause→effect, build a causal chain | `5` |
| `COGNITIVE_GRAPH_EVERY_N_TICKS` | Ingest recent memory into the graph | `8` |
| `EVOLUTION_EVERY_N_TICKS` | Propose a mutation (judged by the next benchmark) | `100` |
| `MAX_RISK_CONFIDENCE_PENALTY` | Cap on how far observed failure history may cut decision confidence | `0.4` |
| `EVAL_EVERY_N_TICKS` | Held-out benchmark run (the fitness signal) | `50` |
| `ENV_STEP_EVERY_N_TICKS` | Live environment step — one real task, real reward | `2` |
| `SKILL_SYNTH_EVERY_N_TICKS` | Attempt to synthesize a skill for a failing task kind | `200` |
| `SANDBOX_TIMEOUT` | Hard timeout for one sandboxed skill execution (s) | `3.0` |
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

## Testing & Quality

```bash
pip install -r requirements-dev.txt

python -m pytest -q                                    # 1363 tests
python -m coverage run -m pytest -q && python -m coverage report   # gate: 90%
python scripts/mutation_test.py                        # gate: no survivors
```

No ML dependencies are required — the whole suite runs offline (no network, no LLM).

| Metric | Value | Gate |
|---|---:|---:|
| Tests | **1363** (+2 skipped) | all green |
| Branch coverage (whole package) | **92%** | **90%** |
| Mutation score | **99.8%** (446/447) | no survivors |
| Modules at 100% mutation score | **12 of 13** | — |

**Levels.** Unit tests per module · integration tests running the five systems
inside a real `Substrate` tick · executable **Gherkin** specifications
(`tests/features/*.feature`, driven by `pytest-bdd`) · audit regression tests ·
an API contract sweep derived from the app's own route table · a dashboard
contract test that checks all 161 field reads against the real status payload.

**Mutation testing** uses a dependency-free in-repo harness (`scripts/mutation_test.py`):
mutmut needs WSL on Windows and mutatest 3.1.0 is broken on Python 3.11. It flips
comparisons, arithmetic, boolean operators and boolean constants, runs the module's
tests per mutant, and always restores the source byte-for-byte. It exits non-zero if
any non-equivalent mutant survives.

Why it earns its keep: `event_bus` had 98% coverage, yet flipping its fail-closed
veto (`allowed = False` → `True`) — i.e. *a crashed safety check now lets the event
through* — left the entire suite green. Coverage measures execution; mutation score
measures whether anything actually asserts.

**Audit.** Three rounds are documented in [`docs/АУДИТ.md`](docs/АУДИТ.md), with every
finding tied to a test that was verified **red before the fix**. Round 3 found a
critical sandbox escape (proven RCE), an unauthenticated state leak over WebSocket,
and two dead API endpoints — all four in files that were excluded from the coverage
gate at the time. Those exclusions are gone. QA procedures and metrics live in
[`docs/QA.md`](docs/QA.md).

---

## Documentation

The engineering docs are written in Russian; this README is the English entry point.

| Document | What is in it |
|---|---|
| [`docs/АУДИТ.md`](docs/АУДИТ.md) | All three audit rounds — every finding with file:line, risk, failure scenario and the fix, plus the "red before the fix" proof for each regression test |
| [`docs/QA.md`](docs/QA.md) | Test levels, reproduction commands, coverage gate, mutation-testing methodology, test-isolation rules and the pre-merge checklist |
| [`docs/СИСТЕМЫ.md`](docs/СИСТЕМЫ.md) | The five higher-order systems in detail — data structures, tick integration, persistence |

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
