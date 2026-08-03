# AEGIS — Autonomous Evolving General Intelligence System

[![tests](https://img.shields.io/badge/tests-4735%20passing-2ea44f)](#testing--quality)
[![coverage](https://img.shields.io/badge/branch%20coverage-95%25-2ea44f)](#testing--quality)
[![mutation score](https://img.shields.io/badge/mutation-100%25%20on%20round--5%20modules-2ea44f)](#testing--quality)
[![audit](https://img.shields.io/badge/audit-5%20rounds-blue)](docs/%D0%90%D0%A3%D0%94%D0%98%D0%A2.md)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> A self-developing AI that predicts before it acts, changes its behaviour from measured experience, evolves a genome that provably moves its own benchmark, improves its own reasoning strategies, and derives laws about itself that it then has to defend against an experiment.
> **7-layer architecture + 5 higher-order systems + 7 contours of the development spec · provider-agnostic cortex · deterministic core cycle.**
> **4735 tests · 95% branch coverage · 5 audit rounds, 34 findings closed in the fifth.**
> **Safe defaults:** source self-rewriting is opt-in (`AEGIS_CODE_SELF_MOD_ENABLED=1`), the control plane binds to `127.0.0.1`, and self-written skills run only in a child-process sandbox.

🌐 **[aegis-asi.com](https://aegis-asi.com)** · 📊 [Control Center](https://aegis-asi.com/panel.pdf)

---

## What is AEGIS?

AEGIS is not a chatbot wrapper. It is a stateful, continuously running cognitive system that runs a loop every 3 seconds:

```
PERCEIVE → EVALUATE → DECIDE → ACT → REFLECT
```

The five phases run in that order, unconditionally — but *completes* would be the
wrong word, and the difference is load-bearing. A phase that raises takes the rest
of the cycle with it, and the tick is recorded as failed rather than counted as
work done. That is not a defect to paper over: audit round 5 found a list-shaped
model reply aborting EVALUATE and silently costing that tick its DECIDE, ACT and
REFLECT. A loop that always claimed to finish would have hidden it.

The **whole system is deterministic**, not just its core. Reward, confidence, importance, emotions, goals, dreams and self-modification are driven by real system metrics (`success_rate`, `energy`, `error_rate`, `goal_completion`); the last places that still called an RNG — topic selection, agent staggering, id assignment, dataset shuffling — now rotate through fixed lists or index by a `blake2b` hash of their own inputs, and quasi-random sampling uses Halton sequences. A test walks every file under `aegis/` and fails on an `import random`, an `np.random`, or any RNG call, and a second test proves the replacements still do real work rather than returning a constant.

This is what makes the rest of the README measurable. Every acceptance number below is a comparison of *before* against *after*; if two identical runs could differ, none of those differences would mean anything. Two 300-tick runs from one state produce a byte-identical state digest.

On top of that cycle run five higher-order systems that give it a causal model of
the world, a connected knowledge graph, benchmark-gated evolution, value-driven
motivation, and a closed loop from real outcomes back into training data — see
[below](#the-five-higher-order-systems).

Above **those** run the seven contours of the development specification, which
turn each of those capabilities from something the system *has* into something it
can be held to: a forecast written down before the action and scored after it, a
behaviour policy whose rules carry evidence and expire, motivation that spends a
real budget, evolution over a genome that provably moves the metric, reasoning
strategies the system writes and then has to prove on held-out problems, and a
discovery engine that produces laws about itself and has to defend them against a
preregistered experiment — see
[below](#the-seven-contours-of-the-development-spec).

---

## Architecture — 7 Layers, 5 Systems, 7 Contours

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

The tick itself is five phase modules (`layers/phases/*.py`) sharing one
`TickContext`, so a phase can be read, budgeted and tested on its own. Everything
time-dependent goes through an injectable `clock.py`, which is what makes a
300-tick determinism test possible at all.

| Package | What lives there |
|---|---|
| `cortex/` | Provider-agnostic LLM router — roles, JSON schemas with one repair attempt, response cache, circuit breaker, failover. The core runs with no provider at all. |
| `layers/world/` | State encoder, transition and outcome models, forecast scorer, deterministic rollout simulator |
| `layers/planner.py` | Plans compared on expected value, cost and risk — not the highest-priority goal |
| `layers/policy/` | Preferences, mined rules with a controlled trial, counterfactual regret |
| `layers/motivation/` | Resource leases, priority with anti-starvation, ROI-driven reallocation |
| `layers/evolution/` | Genome, deterministic operators, population, isolated variant evaluation |
| `layers/reasoning/` | Strategy DSL, interpreter, library, weakness detector, synthesiser, arena |
| `layers/discovery/` | Data pool, hypothesis scan with FDR control, symbolic regression, preregistered experiments, discovery ledger |
| `layers/metacognition/` | Ablation attribution, mechanism-credit table, strategy distance and archive, skeleton catalogue with permanent retirements |
| `telemetry/`, `store/`, `safety/` | Metric time series, versioned stores with migrations, the immutable-parameter contract |

---

## The Five Higher-Order Systems

Layers L0–L6 make AEGIS *self-aware*. These five make it *effective in the world* —
they close the gap between "the model talks about itself" and "the system reasons
about causes, keeps only changes that measurably help, and learns from real results."

| # | System | File | What it does |
|---|---|---|---|
| 1 | **World Model** | `world_model.py` | Learns cause→effect links from observed outcomes (Laplace-smoothed frequencies) and builds causal chains: `objective → constraints → risks → plan → expected result`. Answers "what tends to fail around this topic" from memory, not from text generation. |
| 2 | **Cognitive Graph** | `cognitive_graph.py` | Typed graph of knowledge and experience (`concept / event / skill / goal / outcome`, edges `causes / requires / learned_from / led_to`). Gives path finding, relevance and centrality, so reasoning uses **connected** knowledge instead of a flat recency list. |
| 3 | **Evolution Engine** | `evolution_engine.py`, `layers/evolution/` | Natural selection over its own parameters: `champion → mutation → candidate → held-out benchmark → keep only if better`, since M5 over a population of ten variants with the winner confirmed on a third, untouched split. This is what makes self-modification ≠ self-*improvement* an enforced distinction: a change survives only when an external, verifiable metric says so. |
| 4 | **Goal Intelligence** | `goal_intelligence.py` | Turns internal state into motivation: `goal → value → action choice → reward`. Keeps a learned utility per objective, updated from realized reward, and picks what to pursue by expected value under four intrinsic drives (competence, knowledge, coherence, stability). |
| 5 | **Feedback Loop** | `feedback_loop.py` | Closes the experience loop: `situation → decision → real result → evaluation → cause → new experience`. Every row is structured (it records **why** an outcome happened), so the LoRA dataset carries causes rather than raw text. |

All five are deterministic and dependency-free; an LLM may refine a plan, but is
never required for them to function. They persist atomically and are bounded in
size, so no structure grows without limit.

### Capacity is measured, not guessed

How much causal memory a machine can afford is not knowable in advance, so it is
not fixed in advance. `MAX_LINKS` / `MAX_NODES` are the **floor**; the live caps
track average tick latency, which the health monitor already records and already
has a threshold for. Ticks comfortably under it buy capacity, ticks over it hand
capacity back, bounded by `CAPACITY_MAX_MULTIPLE`. No benchmark is involved —
this is homeostasis on a directly measured cost, which is what makes it honest.

What gets forgotten matters as much as how much is kept. Pruning ranks a link by
**how much it tells you** — how far its success rate sits from a coin flip,
weighted by the evidence behind it, with a deliberate bias toward remembering
failure. Sorting by observation count alone (the previous rule) discarded the
rare decisive failure before the frequent unremarkable link, which is backwards
for a memory whose job is to anticipate what goes wrong.

Risk lookup goes through a token index rather than a scan of the cause table:
`risks_for` runs on every tick now, so its cost must not grow with everything
the system has ever learned.

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
| Evolution Engine | Closed by construction (benchmark-gated), and since M5 the genes it selects on are ones that provably move that benchmark — see the note below. |

> **The limitation this section used to declare is closed.** The Evolution
> Engine's mechanism was always sound — mutate, benchmark, keep or revert, with
> the champion never updated from self-report — but the genes it mutated
> (`learning_rate`, `attention_heads`, …) did not causally drive the benchmark,
> so selection rode on drift from other changes. The genome has been replaced by
> one made only of parameters that provably move the measured metric: beam width
> and rollout depth, the planner's scoring weights, the policy's influence and
> evidence threshold, exploration pressure, world-model smoothing and forgetting,
> the reasoning budget. Every gene carries a sensitivity test — a gene that
> cannot be shown to move the fitness by at least 0.01 is removed from the genome
> rather than left to spin. The LoRA hyper-parameters stay in
> `SelfModification.parameters`, where the training contour still uses them, and
> are out of the genome entirely. See [`docs/ЭВОЛЮЦИЯ.md`](docs/%D0%AD%D0%92%D0%9E%D0%9B%D0%AE%D0%A6%D0%98%D0%AF.md).

---

## The Contours of the Development Spec

The five systems above make the cycle *effective*. These contours make each of
its claims *checkable* — every one of them has a metric for "this changed
behaviour" and a metric for "the change helped", and both are asserted by a
test. M11 is the meta-level over M6: its inputs are the reasoning contour's
outcomes, its output is a change to how strategies are generated.

| # | Contour | The link it adds | Acceptance |
|---|---|---|---|
| M1 | **Predictive world model** | The forecast becomes an object: written before the action, scored after it, and its error drives curiosity. | Brier ≤ 0.18, ECE ≤ 0.08, strictly better than "always the mean rate" and "always 0.5" |
| M2 | **Planner** | Decisions come from comparing plans under the world model, not from the highest-priority goal. | **+113%** mean reward against the greedy baseline at equal budget; planning 18.3 ms |
| M3 | **Behaviour policy** | The fifth arrow — `experience → behaviour change` — as an object: rules with evidence, a controlled trial and an expiry. | Zero rules on noise; suppression and recovery scenarios; `behaviour_delta_rate > 0` |
| M4 | **Motivation with resources** | Priority as a number and resource as something that runs out. An action without a lease does not run. | Budget exhaustion, anti-starvation, safety floors, ROI never reaching zero |
| M5 | **Population evolution** | Ten variants a generation, isolated evaluation, selection on a validation split confirmed on a test split. | +15% champion fitness over 20 generations, `valid_test_gap ≤ 0.05` |
| M6 | **Thinking as data** | A reasoning strategy is a declarative pipeline the system writes, judges on held-out problems, and retires. | held-out **0.685 → 0.965** (+28.0 pp), confident errors 0.305 → **0.000** |
| M7 | **Discovery engine** | `data → hypothesis → formula → experiment → law`, with false-discovery control and a plan frozen before the data. | Planted law recovered at **R²_valid 0.9999**, 3 replicated discoveries, **0** from 1296 comparisons of noise |
| M11 | **Metacognition** | *Why* a strategy won, proven by ablation, and invention of structurally different strategies under a novelty quota — the meta-loop that changes the generator, not the strategies. | Planted cause found at precision/recall **1.0**, **0** confirmations from 200 noise comparisons, order_delta **1.0**, 2 far candidates accepted through unsoftened gates, byte-identical registries across runs |

Bold figures are measured by the acceptance scripts below; the rest are the
gates those runs had to clear.

Two design decisions run through all seven.

**A strategy is not Python.** M6 synthesises reasoning strategies, and one of the
synthesisers is a language model. A synthesised strategy that were Python would be
arbitrary code execution with a friendly name, so it is a list of records drawn
from twelve operations, run by an interpreter that can do exactly those twelve,
counts every step against one budget, and reaches only what it was handed.

**Not finding a law is the hard part.** A scan over a dozen metrics at six lags
under three measures performs hundreds of tests; at α = 0.05 roughly one in twenty
useless pairs clears an uncorrected threshold. So the correction is applied over
every comparison made rather than the ones that looked promising, the experiment's
plan is hashed before any data exists, and a refutation is kept forever — both
because it is knowledge and because otherwise the same appealing pattern is
rediscovered every thousand ticks.

Acceptance is reproducible, not asserted:

```bash
python scripts/ab_planner.py       # M2 — the planner against a greedy baseline
python scripts/ab_policy.py        # M3 — behaviour with and without the policy
python scripts/evo_bench.py        # M5 — 20 generations
python scripts/reason_bench.py     # M6 — 30 synthesis cycles
python scripts/discovery_soak.py   # M7 — a planted law, and pure noise
python scripts/meta_bench.py       # M11 — attribution, ordering, the far track

python scripts/soak.py             # VII.5 — the 24-hour run
```

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
3. `code_modifier.py` validates syntax, then runs **AST-based** detection of dangerous calls/imports (`eval`, `exec`, `compile`, `__import__`, `os.system`, `subprocess`, `shutil.rmtree`, …) that cannot be fooled by spacing tricks, import aliases (`import os as o; o.kill(...)`), or indirect escape hatches (`importlib`, `builtins`). The list knew `os.system` but **no member of the process-execution family** until audit R5-2: `os.execv`, `os.spawnl`, `os.posix_spawn`, `os.fork` and `os.startfile` all passed the gate that exists to stop exactly that, and `Path(x).write_text(y)` never reached the write-mode check because it required the receiver to be a plain name. Both are closed, on any receiver
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
- Set **`AEGIS_API_TOKEN`** to require an `X-API-Token` header on **every** `/api` request — reads included — and on privileged WebSocket actions. Comparison is constant-time. It gated only mutating methods until audit R5-4: `GET /api/status` serves `full_status()`, i.e. memory contents, goals and ethics state, and `network_exposure_warning()` was calling a token-plus-`0.0.0.0` bind *safe* while that was readable by anyone on the network.
- **Only authenticated WebSocket clients join the state broadcast.** The handshake check alone was not enough: an unauthorized socket used to stay in the fan-out list and receive the periodic `full_status()` push, so simply connecting and waiting leaked internal state past the token gate (audit R3-2).
- WebSockets are not covered by CORS, so the `Origin` header is checked **before** the handshake is accepted — a hostile page cannot open `ws://127.0.0.1:8888/ws` (CSWSH).
- Cross-origin access is off unless `AEGIS_API_CORS_ORIGINS` is set. A wildcard (`*`) origin automatically **disables credentials**, so a permissive CORS config can never be combined with credentialed cross-site calls.
- **The kill switch stops state-changing endpoints, not just the tick loop.** It used to guard three POSTs out of twenty, and neither `evaluate_action` nor `evaluate_weight_modification` consults it — so a "stopped" system still accepted `/api/self-mod/propose` and could start a LoRA run. Every mutating endpoint now answers `423` while the switch is active, except the switch and lockdown themselves (audit R5-3).

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

The gate also refuses a **name spelled as data**. Reading the AST is not enough:
`operator.attrgetter("__globals__")(json.dumps)` never mentions a dunder as a
name or an attribute, and it walked straight through to `__builtins__`,
`__import__` and `os` — verified end to end, with the child process returning
this machine's working directory (audit R5-1). Any dunder-shaped string literal,
and `operator.attrgetter`/`methodcaller` themselves, are now rejected outright;
`operator.add`, ordinary dict keys and non-dunder strings are untouched.

This is defense-in-depth, **not** an OS-level jail. Three escapes have been found
in this layer across three audit rounds, each one a different way for a static
gate to be looking at the wrong thing, which is itself the argument: a production
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
│   ├── АУДИТ.md               # 5 audit rounds — every finding, fix and test
│   ├── QA.md                  # QA procedures, quality metrics, gates
│   ├── СИСТЕМЫ.md             # the five higher-order systems in detail
│   ├── ТЗ-РАЗВИТИЕ.md         # the development specification
│   ├── ПЛАН-ЭТАПОВ.md         # its twelve stages and what each delivered
│   ├── ПРОГНОЗ.md             # M1–M2 — prediction and planning
│   ├── РЕСУРСЫ.md             # M3–M4 — behaviour policy and resources
│   ├── ЭВОЛЮЦИЯ.md            # M5 — the genome and why it was replaced
│   ├── МЫШЛЕНИЕ.md            # M6 — reasoning as data
│   ├── ОТКРЫТИЯ.md            # M7 — the discovery engine
│   └── МЕТАКОГНИЦИЯ.md        # M11 — attribution and invention
├── scripts/
│   ├── mutation_test.py       # dependency-free mutation-testing harness
│   ├── ab_planner.py          # M2 acceptance — planner vs greedy
│   ├── ab_policy.py           # M3 acceptance — behaviour with/without policy
│   ├── evo_bench.py           # M5 acceptance — 20 generations
│   ├── reason_bench.py        # M6 acceptance — 30 synthesis cycles
│   ├── discovery_soak.py      # M7 acceptance — planted law, and noise
│   ├── meta_bench.py          # M11 acceptance — attribution and the far track
│   ├── soak.py                # VII.5 — the 24-hour run
│   ├── check_no_stubs.py      # no `pass`-bodied production code
│   └── check_undefined_names.py
├── tests/                      # 4735 tests
│   └── features/              # 10 executable Gherkin specifications
├── aegis/
│   ├── config.py              # IMMUTABLE
│   ├── clock.py               # injectable clock — what makes determinism testable
│   ├── event_bus.py
│   ├── llm.py                 # Triple LLM: DeepSeek + Claude + Local LoRA
│   ├── cortex/                # provider-agnostic router, schemas, cache, breaker
│   ├── safety/                # immutable/bounded/monotonic parameter contract
│   ├── store/                 # versioned stores, migrations, atomic writes
│   ├── telemetry/             # metric time series, percentiles, compaction
│   ├── util/                  # deterministic stats — Wilson, BH-FDR, Welch, BIC
│   ├── api/
│   │   └── server.py          # FastAPI + WebSocket (token-guarded)
│   ├── dashboard/
│   │   └── index.html         # Real-time SPA dashboard
│   ├── eval/                  # benchmark, skill library, isolated sandbox
│   └── layers/
│       ├── substrate.py
│       ├── phases/            # PERCEIVE EVALUATE DECIDE ACT REFLECT, budgeted
│       ├── world/             # M1 — encoder, transition/outcome models, scorer
│       ├── planner.py         # M2 — plans compared on value, cost and risk
│       ├── policy/            # M3 — preferences, mined rules, regret
│       ├── motivation/        # M4 — leases, priority, ROI reallocation
│       ├── evolution/         # M5 — genome, operators, population
│       ├── reasoning/         # M6 — DSL, interpreter, weakness, synthesis, arena
│       ├── discovery/         # M7 — data pool, hypotheses, symbolic, experiments
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
| **Capacity homeostasis** | | |
| `CAPACITY_EVERY_N_TICKS` | How often the caps are re-sized from measured tick latency | `50` |
| `CAPACITY_HEADROOM` | Grow only while avg tick latency is below this fraction of the health threshold | `0.5` |
| `CAPACITY_GROWTH_FACTOR` / `CAPACITY_SHRINK_FACTOR` | Step up / step down multipliers | `1.25` / `0.8` |
| `CAPACITY_MAX_MULTIPLE` | Ceiling as a multiple of the baseline cap | `20` |
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

### The seven contours of the development spec

Every one of these has a deterministic path that works with no model attached;
the variables below tune that path rather than switch it on.

**Predictive world model (M1)**

| Variable | What it does | Default |
|---|---|---|
| `WM_SMOOTHING` | Additive smoothing for transition and outcome estimates | `1.0` |
| `WM_MIN_N` | Observations before a `(state, action)` pair is trusted | `3` |
| `WM_BRANCH` / `WM_BEAM` / `WM_DEPTH` | Successors per node, beam width, rollout depth | `3` / `5` / `3` |
| `WM_DISCOUNT` | Rollout horizon | `0.9` |
| `WM_HALF_LIFE` | Observations after which a transition count halves | `500` |
| `WM_EXPLORE_BONUS` | Reward for going where the model knows little | `0.15` |

**Planner (M2) and behaviour policy (M3)**

| Variable | What it does | Default |
|---|---|---|
| `PLAN_ENABLED` | Plan comparison on/off (off = greedy baseline) | `1` |
| `PLAN_MAX_CANDIDATES` | Objectives considered per tick | `12` |
| `POLICY_LR` / `POLICY_WEIGHT` | Preference learning rate, and its weight in scoring | `0.15` / `0.3` |
| `POLICY_MIN_SUPPORT` | Observations before a rule may be mined | `20` |
| `POLICY_MINE_EVERY_N_TICKS` | Rule mining cadence | `200` |
| `POLICY_TRIAL_TICKS` / `POLICY_REVIEW_TICKS` | Trial length, and re-judgement interval | `300` / `1000` |
| `POLICY_MIN_EFFECT` / `POLICY_ALPHA` | Effect and significance a rule must clear | `0.03` / `0.05` |

**Motivation and resources (M4)**

| Variable | What it does | Default |
|---|---|---|
| `RES_TOKENS_PER_HOUR` / `RES_CALLS_PER_HOUR` | Token and call allowance | `200000` / `400` |
| `RES_WALL_MS_PER_TICK` | Wall-clock allowance per tick | `2500` |
| `RES_SUBPROC_SLOTS` / `RES_TRAINING_SLOTS` | Concurrent subprocesses and training runs | `4` / `1` |
| `RESOURCE_SAFETY_FLOOR` | Share of budget nothing may take from safety work | `0.15` |
| `RESOURCE_MIN_SHARE` | Floor under any activity's share, so its ROI can still update | `0.05` |
| `PRIORITY_AGING` / `PRIORITY_AGING_MAX_TICKS` | Anti-starvation bonus and its cap | `0.01` / `2000` |

**Evolution (M5)**

| Variable | What it does | Default |
|---|---|---|
| `EVO_POP_SIZE` | Variants per generation | `10` |
| `EVO_SIGMA` / `EVO_EPSILON` | Mutation amplitude, and the gain a challenger must show | `0.15` / `0.005` |
| `EVO_MIN_DISTANCE` | Below this a variant is too similar to be worth evaluating | `0.02` |
| `EVO_EVERY_N_TICKS` | Minimum interval between generations | `250` |
| `EVO_WATCH_TICKS` / `EVO_ROLLBACK_DELTA` | Post-promotion watch window and the drop that triggers rollback | `500` / `0.03` |

**Thinking (M6)**

| Variable | What it does | Default |
|---|---|---|
| `REASON_MAX_STEPS` | Hard ceiling on interpreter steps per strategy | `24` |
| `REASON_MIN_GAIN` | Held-out gain a candidate strategy must show | `0.05` |
| `REASON_COST_TOLERANCE` | How much more a candidate may cost than the incumbent | `1.5` |
| `REASON_TRIAL_N` | Applications before a trial strategy is judged | `50` |
| `REASON_SCAN_EVERY_N_TICKS` | Weakness scan cadence | `300` |
| `REASON_UCB_C` | Exploration constant in strategy selection | `1.4` |

**Discovery (M7)**

| Variable | What it does | Default |
|---|---|---|
| `DISC_ALPHA` | False-discovery-rate level for the hypothesis scan | `0.05` |
| `DISC_MAX_LAG` | How far back a predictor may be lagged | `5` |
| `DISC_MIN_N` | Telemetry points before the first scan | `100` |
| `DISC_LAW_REPS` | Confirmations in separate windows before "law" | `3` |
| `DISC_SCAN_EVERY_N_TICKS` | Scan cadence | `1000` |
| `DISC_INTERVENTION_ENABLED` | Whether self-experiments may run at all | `1` |
| `DISC_INTERVENTION_MAX_DELTA` | Intervention amplitude, as a fraction of the range | `0.2` |
| `DISC_BLOCK_TICKS` | Block length in an ABAB series | `100` |

**Cortex (M8) and infrastructure (M9)**

| Variable | What it does | Default |
|---|---|---|
| `CORTEX_ROUTES` | Role → provider failover chains, as JSON | see `docs/ТЗ-РАЗВИТИЕ.md` §5.2 |
| `CORTEX_DETERMINISTIC` | `temperature=0` for every comparative run | `1` |
| `CORTEX_CACHE_TTL` / `CORTEX_CACHE_MAX` | Response cache lifetime and size | `3600` / `2000` |
| `CORTEX_BREAKER_ERRORS` / `CORTEX_BREAKER_COOLDOWN` | Consecutive errors that trip a provider, and for how long | `5` / `300` |
| `KIMI_API_KEY` / `KIMI_BASE_URL` / `KIMI_MODEL` | Kimi, as a first-class provider | *(none)* |
| `LOCAL_OPENAI_BASE_URL` / `LOCAL_OPENAI_MODEL` | Ollama / llama.cpp / vLLM for the `fast` role | `http://127.0.0.1:11434/v1` |
| `LORA_TARGET_MODULES` | LoRA target modules — must match the model's architecture | `q_proj,k_proj,v_proj,o_proj` |
| `EVAL_POOL_WORKERS` | Processes for isolated variant evaluation | `4` |
| `TELEMETRY_FLUSH_SECONDS` / `TELEMETRY_MAX_ROWS` | Telemetry buffering and retention | `10` / `200000` |

API keys can also be set at runtime via the dashboard (LLM Brain tab).

---

## Testing & Quality

```bash
pip install -r requirements-dev.txt

python -m pytest -q                                    # 4735 tests, ~5.5 min
python -m coverage run -m pytest -q && python -m coverage report   # gate: 90%
python scripts/mutation_test.py                        # gate: no survivors
python scripts/check_no_stubs.py                       # no `pass`-bodied production code
```

No ML dependencies are required — the whole suite runs offline (no network, no LLM).

| Metric | Value | Gate |
|---|---:|---:|
| Tests | **4735** passing (+2 skipped) | all green |
| Branch coverage (whole package) | **95%** | **90%** |
| Mutation score, modules verified at 100% | sandbox · policy (all four) · world/simulate · world/prediction · quasirandom · store/migrations · discovery · telemetry/store · safety/immutable · reasoning DSL, interpreter, library, weakness, synthesis, arena | no survivors |
| Mutation score over the modules round 5 changed | **100%** (286 mutants, 10 modules) | no survivors |
| Executable Gherkin specifications | **10 feature files · 98 scenarios** | — |

**Levels.** Unit tests per module · integration tests running the five systems
and the seven contours inside a real `Substrate` tick · executable **Gherkin**
specifications (`tests/features/*.feature`, driven by `pytest-bdd`) · audit
regression tests · an API contract sweep derived from the app's own route table ·
a dashboard contract test that checks every field read against the real status
payload · a contract test that reads the development specification itself and
asserts each numbered requirement against the code.

Four kinds of test earn their place by catching what unit tests cannot:

- **Determinism end-to-end** — two 300-tick runs from one state must produce a
  byte-identical digest, with a companion test that splices in a genuine
  `random.choice` and *requires* the comparison to fail. Without it, a digest
  that accidentally covered nothing would look like a passing guarantee.
- **Measured phase budgets** — the §3.4 ceilings (PERCEIVE ≤5 ms · EVALUATE
  ≤10 ms · DECIDE ≤30 ms · ACT ≤20 ms · REFLECT ≤15 ms) are asserted against a
  real wall clock in a subprocess. Run under the frozen clock the same test
  passes while measuring zero, which is how it was found to be vacuous.
- **Migration from a real v1 snapshot** — `tests/fixtures/legacy_state/` holds
  pre-spec files in their original shapes. It found a migration that named a
  function which does not exist: every v1 upgrade raised, was caught, logged,
  and silently discarded the evolution champion and its whole lineage.
- **Crash resilience** — SIGTERM mid-write, torn files, files from a future
  schema version. The system must come back up with either the old complete
  state or the new one, never half of each.

The sweep is a gate. Every module audit round 5 touched has been swept and is
clean — 286 mutants across ten modules, no survivors — and the round before it
found real gaps in modules that predate the development spec: `world_model.py`
scored 33% on twelve mutants. Two of them were decorative genes: `synth_attempts` bounded a loop whose repair branch was
capped by a literal, and `mem_retention_bias` was written onto `MemorySystem`,
which never read it, while its declared reader lived in the causal model. Both
passed the gate that exists to forbid exactly that, because the gate read the
genome back from the copy that had been handed out rather than from the contours
— it compared a value against itself. It reads from the contours now, and the two
genes whose effect is a decision rather than a setting have behavioural tests,
because no read-back can tell a value that was consumed from one that was stored.

**Mutation testing** uses a dependency-free in-repo harness (`scripts/mutation_test.py`):
mutmut needs WSL on Windows and mutatest 3.1.0 is broken on Python 3.11. It flips
comparisons, arithmetic, boolean operators and boolean constants, runs the module's
tests per mutant, and always restores the source byte-for-byte. It exits non-zero if
any non-equivalent mutant survives.

Why it earns its keep: `event_bus` had 98% coverage, yet flipping its fail-closed
veto (`allowed = False` → `True`) — i.e. *a crashed safety check now lets the event
through* — left the entire suite green. Coverage measures execution; mutation score
measures whether anything actually asserts.

**Audit.** Five rounds are documented in [`docs/АУДИТ.md`](docs/АУДИТ.md), with every
finding tied to a test that was verified **red before the fix**. Round 3 found a
critical sandbox escape (proven RCE), an unauthenticated state leak over WebSocket,
and two dead API endpoints — all four in files that were excluded from the coverage
gate at the time. Those exclusions are gone.

Round 5 swept the whole tree and closed **34 findings**, two of them critical, and
both of those were the same shape: *a gate that reads syntax cannot see a name
spelled as data.* `operator.attrgetter("__globals__")` walked straight past the
sandbox's AST checks to `__builtins__`, `__import__` and `os` — verified end to
end, with the child process returning the host's working directory — and the
self-modification blocklist knew `os.system` but not one member of the
`exec`/`spawn`/`fork`/`startfile` family, nor `Path(x).write_text(y)`.

The round also asked a question static analysis cannot: **which tests are closed
on themselves** — graded by the system's own output, with no external reference.
One assertion turned out to be a tautology by construction, and the two hardest
reasoning families had no answer anywhere that a human had worked out. Both are
now anchored by a hand-solved table with the prompts pinned verbatim. QA
procedures, metrics and the rule for anchoring a new test live in
[`docs/QA.md`](docs/QA.md).

---

## Documentation

The engineering docs are written in Russian; this README is the English entry point.

| Document | What is in it |
|---|---|
| [`docs/АУДИТ.md`](docs/АУДИТ.md) | All five audit rounds — every finding with file:line, risk, failure scenario and the fix, plus the "red before the fix" proof for each regression test |
| [`docs/QA.md`](docs/QA.md) | Test levels, reproduction commands, coverage gate, mutation-testing methodology, test-isolation rules and the pre-merge checklist |
| [`docs/СИСТЕМЫ.md`](docs/СИСТЕМЫ.md) | The five higher-order systems in detail — data structures, tick integration, persistence |
| [`docs/ПРОГНОЗ.md`](docs/ПРОГНОЗ.md) | The predictive world model — forecast written before the action, scored after it, and the error that drives curiosity |
| [`docs/РЕСУРСЫ.md`](docs/РЕСУРСЫ.md) | Motivation that costs something — priority as a number, resource as something that runs out, and the floors that keep safety funded |
| [`docs/ЭВОЛЮЦИЯ.md`](docs/ЭВОЛЮЦИЯ.md) | Population evolution — ten variants a generation, isolated evaluation, and a genome made of parameters that actually move the metric |
| [`docs/МЫШЛЕНИЕ.md`](docs/МЫШЛЕНИЕ.md) | Thinking as data — a strategy DSL that is not Python, weakness detection with false-discovery control, and the three gates of the arena |
| [`docs/ОТКРЫТИЯ.md`](docs/ОТКРЫТИЯ.md) | The discovery contour — how a hypothesis becomes a formula, an experiment and a registered law, and every gate that stops it registering noise |
| [`docs/ТЗ-РАЗВИТИЕ.md`](docs/ТЗ-РАЗВИТИЕ.md) | The development specification the seven new contours are built to |
| [`docs/ПЛАН-ЭТАПОВ.md`](docs/ПЛАН-ЭТАПОВ.md) | Its twelve stages — what each one delivered, and the acceptance number it had to clear |
| [`docs/ТЗ-МЕТАКОГНИЦИЯ.md`](docs/%D0%A2%D0%97-%D0%9C%D0%95%D0%A2%D0%90%D0%9A%D0%9E%D0%93%D0%9D%D0%98%D0%A6%D0%98%D0%AF.md) | Specification for M11 — attribution of *why* a strategy won, verified by ablation, and deliberate invention of structurally different ones |
| [`docs/МЕТАКОГНИЦИЯ.md`](docs/%D0%9C%D0%95%D0%A2%D0%90%D0%9A%D0%9E%D0%93%D0%9D%D0%98%D0%A6%D0%98%D0%AF.md) | The M11 implementation — ablation attribution, the mechanism-credit table that reorders synthesis, the skeleton catalogue with permanent retirements, and the honest deviations from the letter of the spec |

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
