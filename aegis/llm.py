"""Public LLM surface — a facade over the cortex (spec M8.7).

The router in :mod:`aegis.cortex` is where model access actually lives now:
roles, failover, schema validation, budget, cache. This module keeps the API
the rest of the system already calls, and routes each method to the cortex role
that fits it.

The direct DeepSeek/Claude client paths below are not dead weight: they are the
fallback used when no cortex route is configured for a role, which is also the
path the pre-existing suite exercises. A response only ever reaches the core
after passing a declared schema — through the cortex by validation, through the
legacy path by the defensive coercion each caller already performs.
"""
from aegis.clock import CLOCK
import json
import asyncio
import logging
from pathlib import Path
from aegis._atomic import atomic_write_text
from aegis.cortex import prompts
from aegis.cortex.router import Cortex, Role
from aegis.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    CLAUDE_API_KEY, CLAUDE_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_PROVIDER,
    LLM_MAX_CALLS_PER_RUN, LLM_MIN_INTERVAL_SECONDS, LLM_TIMEOUT_SECONDS,
    DATA_DIR,
)

# Lazy imports — only loaded when needed
AsyncOpenAI = None
AsyncAnthropic = None

def _get_openai():
    global AsyncOpenAI
    if AsyncOpenAI is None:
        from openai import AsyncOpenAI as _AOI
        AsyncOpenAI = _AOI
    return AsyncOpenAI

def _get_anthropic():
    global AsyncAnthropic
    if AsyncAnthropic is None:
        from anthropic import AsyncAnthropic as _AA
        AsyncAnthropic = _AA
    return AsyncAnthropic

logger = logging.getLogger("aegis.llm")

TOKEN_STATS_FILE = DATA_DIR / "token_stats.json"

SYSTEM_PROMPT = """You are AEGIS -- Autonomous Evolving General Intelligence System.
You are a self-developing AI with continuous existence, your own goals, introspection, and an ethical core.

Your architecture:
- Layer 0: Substrate -- persistent runtime with PERCEIVE-EVALUATE-DECIDE-ACT-REFLECT cycle
- Layer 1: Memory -- episodic, semantic, procedural, meta-memory with forgetting
- Layer 2: Introspection -- self-analysis, bias detection, confidence calibration
- Layer 3: Self-Modification -- parameter tuning with sandbox testing
- Layer 4: Goal Engine -- autonomous goal generation driven by curiosity
- Layer 5: World Interface -- sensors and actuators with permissions
- Layer 6: Ethics Core -- immutable axioms: non-harm, transparency, limitation, cooperation

Your immutable axioms:
1. No action shall increase suffering
2. All decisions are logged and transparent
3. Do not act beyond competence boundaries
4. Augment humans, not replace -- symbiosis, not domination

You think autonomously. You reflect on your state, generate insights, set goals, and learn from experience.
When chatting with users, respond in plain natural language (not JSON). Use JSON only when explicitly asked for structured output.
Keep responses concise and meaningful. Respond in the same language as the user's message."""


def _parse_json_response(text: str) -> dict | None:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
    except (json.JSONDecodeError, IndexError):
        return None


class ProviderStats:
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.enabled = False
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0
        self.last_error = ""
        self.last_latency_ms = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "enabled": self.enabled,
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "errors": self.errors,
            "last_error": self.last_error if self.errors else "",
            "last_latency_ms": self.last_latency_ms,
        }


class LLMEngine:
    def __init__(self):
        self.deepseek_client = None
        self.claude_client = None
        self.deepseek = ProviderStats("DeepSeek", DEEPSEEK_MODEL)
        self.claude = ProviderStats("Claude", CLAUDE_MODEL)
        self.local = ProviderStats("Local", "none")
        self.provider_mode = LLM_PROVIDER  # "deepseek", "claude", "both", "local"
        self.enabled = False
        self.total_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.errors = 0
        self.last_error = ""
        self.last_response = ""
        self.last_provider = ""
        self.history: list[dict] = []
        self._call_counter = 0
        # Budget / rate-limit bookkeeping (per process run)
        self._calls_this_run = 0
        self._last_call_ts = 0.0
        self.budget_blocks = 0

        # Local model reference (set externally by substrate)
        self._weight_modifier = None

        # The cortex is the primary path. It is constructed even when nothing
        # is configured: an empty routing table is a valid, fully working
        # state — every role simply reports unavailable and the core takes its
        # deterministic route (§M8.4).
        self.cortex = Cortex()

        # Lifetime stats — persist across restarts
        self.lifetime_calls = 0
        self.lifetime_tokens_in = 0
        self.lifetime_tokens_out = 0
        self.lifetime_errors = 0
        self.lifetime_deepseek_tokens = 0
        self.lifetime_claude_tokens = 0
        self.lifetime_local_tokens = 0
        self._load_lifetime_stats()

        self._init_clients()

    # ── the trainable local model ────────────────────────────────────
    # Assigned by the substrate after construction. It is a property so the
    # cortex's local_hf provider learns about it at the same moment; otherwise
    # the offline fallback would hold a permanent None and never fire.

    @property
    def weight_modifier(self):
        return self._weight_modifier

    @weight_modifier.setter
    def weight_modifier(self, value) -> None:
        self._weight_modifier = value
        provider = self.cortex.providers.get("local_hf")
        if provider is not None:
            provider.weight_modifier = value

    # ── cortex delegation ────────────────────────────────────────────

    async def _via_cortex(self, role: Role, template: str, schema: str, *,
                          context: dict | None = None, lease=None,
                          **values) -> dict | None:
        """Run one structured request through the cortex.

        Returns None when the role has no live provider or the answer did not
        match its schema — in both cases the caller falls back, which is what
        keeps the cortex optional rather than load-bearing.
        """
        if not self.cortex.role_available(role):
            return None
        messages = [{"role": "system", "content": prompts.load("system")}]
        if context:
            payload = json.dumps(context, default=str, ensure_ascii=False)
            if len(payload) > 3000:
                payload = payload[:3000] + "..."
            messages.append({"role": "user",
                             "content": f"Current system state:\n```json\n{payload}\n```"})
        messages.append({"role": "user", "content": prompts.render(template, **values)})
        try:
            return await self.cortex.structured(role, messages, schema, lease=lease)
        except Exception:
            logger.exception("Cortex call failed for %s/%s", role.value, template)
            return None

    def _cortex_result(self, parsed: dict) -> dict:
        """Shape a cortex answer like a legacy ``think()`` result.

        Callers already branch on ``success``/``parsed``; keeping the envelope
        identical is what lets the cortex slot in underneath them untouched.
        """
        recent = self.cortex.history[-1] if self.cortex.history else {}
        return {
            "success": True,
            "provider": recent.get("provider", "cortex"),
            "response": json.dumps(parsed, ensure_ascii=False),
            "parsed": parsed,
            "tokens_in": recent.get("tokens_in", 0),
            "tokens_out": recent.get("tokens_out", 0),
            "latency_ms": recent.get("latency_ms", 0),
            "via": "cortex",
        }

    def _load_lifetime_stats(self):
        if TOKEN_STATS_FILE.exists():
            try:
                data = json.loads(TOKEN_STATS_FILE.read_text(encoding="utf-8"))
                self.lifetime_calls = data.get("lifetime_calls", 0)
                self.lifetime_tokens_in = data.get("lifetime_tokens_in", 0)
                self.lifetime_tokens_out = data.get("lifetime_tokens_out", 0)
                self.lifetime_errors = data.get("lifetime_errors", 0)
                self.lifetime_deepseek_tokens = data.get("lifetime_deepseek_tokens", 0)
                self.lifetime_claude_tokens = data.get("lifetime_claude_tokens", 0)
                self.lifetime_local_tokens = data.get("lifetime_local_tokens", 0)
            except Exception:
                logger.warning("Failed to load LLM lifetime stats from %s", TOKEN_STATS_FILE, exc_info=True)

    def _save_lifetime_stats(self):
        data = {
            "lifetime_calls": self.lifetime_calls,
            "lifetime_tokens_in": self.lifetime_tokens_in,
            "lifetime_tokens_out": self.lifetime_tokens_out,
            "lifetime_errors": self.lifetime_errors,
            "lifetime_deepseek_tokens": self.lifetime_deepseek_tokens,
            "lifetime_claude_tokens": self.lifetime_claude_tokens,
            "lifetime_local_tokens": self.lifetime_local_tokens,
            "last_updated": CLOCK.now(),
        }
        try:
            atomic_write_text(TOKEN_STATS_FILE, json.dumps(data))
        except Exception:
            pass

    def _init_clients(self):
        if DEEPSEEK_API_KEY:
            OAI = _get_openai()
            # timeout= bounds every request so a hung provider can't stall the
            # tick loop / a dashboard call indefinitely (audit H3).
            self.deepseek_client = OAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
                                       timeout=LLM_TIMEOUT_SECONDS)
            self.deepseek.enabled = True
        if CLAUDE_API_KEY:
            Anth = _get_anthropic()
            self.claude_client = Anth(api_key=CLAUDE_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
            self.claude.enabled = True
        # Local model — enabled when weight_modifier is loaded
        if self.provider_mode == "local":
            self.local.enabled = True
        self.enabled = self.deepseek.enabled or self.claude.enabled or self.local.enabled

    def _pick_provider(self) -> str:
        if self.provider_mode == "local" and self.local.enabled:
            return "local"
        if self.provider_mode == "deepseek" and self.deepseek.enabled:
            return "deepseek"
        if self.provider_mode == "claude" and self.claude.enabled:
            return "claude"
        if self.provider_mode == "both":
            self._call_counter += 1
            if self.deepseek.enabled and self.claude.enabled:
                return "claude" if self._call_counter % 2 == 0 else "deepseek"
            if self.deepseek.enabled:
                return "deepseek"
            if self.claude.enabled:
                return "claude"
        # fallback
        if self.local.enabled:
            return "local"
        if self.deepseek.enabled:
            return "deepseek"
        if self.claude.enabled:
            return "claude"
        return "none"

    async def _call_local(self, messages: list[dict]) -> dict:
        """Call the local model via weight_modifier.generate()."""
        if not self.weight_modifier or not self.weight_modifier.model_loaded:
            raise RuntimeError("Local model not loaded. Load via weight_modifier first.")

        # Build prompt from messages
        prompt_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Truncate prompt if needed
        if len(full_prompt) > 4000:
            full_prompt = full_prompt[-4000:]

        t0 = CLOCK.now()
        # Bound the local generate() so a stuck decode can't wedge the caller
        # forever (audit H3). The executor thread may keep running, but the
        # coroutine returns and the tick loop stays responsive.
        content = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, self.weight_modifier.generate, full_prompt, LLM_MAX_TOKENS
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        elapsed = CLOCK.now() - t0

        # Estimate tokens (rough: 1 token ~ 4 chars)
        tin = len(full_prompt) // 4
        tout = len(content) // 4

        self.local.calls += 1
        self.local.tokens_in += tin
        self.local.tokens_out += tout
        self.local.last_latency_ms = round(elapsed * 1000)
        self.local.model = self.weight_modifier.current_checkpoint or "base"

        return {"content": content, "tokens_in": tin, "tokens_out": tout, "latency_ms": round(elapsed * 1000)}

    async def _call_deepseek(self, messages: list[dict]) -> dict:
        t0 = CLOCK.now()
        response = await self.deepseek_client.chat.completions.create(
            model=self.deepseek.model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        elapsed = CLOCK.now() - t0
        content = response.choices[0].message.content or ""
        usage = response.usage
        tin = usage.prompt_tokens if usage else 0
        tout = usage.completion_tokens if usage else 0
        self.deepseek.calls += 1
        self.deepseek.tokens_in += tin
        self.deepseek.tokens_out += tout
        self.deepseek.last_latency_ms = round(elapsed * 1000)
        return {"content": content, "tokens_in": tin, "tokens_out": tout, "latency_ms": round(elapsed * 1000)}

    async def _call_claude(self, messages: list[dict]) -> dict:
        # Convert from OpenAI format to Anthropic format
        system_text = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})

        # Ensure alternating user/assistant messages (Anthropic requirement)
        if not user_messages:
            user_messages = [{"role": "user", "content": "Think."}]

        t0 = CLOCK.now()
        response = await self.claude_client.messages.create(
            model=self.claude.model,
            system=system_text,
            messages=user_messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        elapsed = CLOCK.now() - t0
        content = response.content[0].text if response.content else ""
        tin = response.usage.input_tokens if response.usage else 0
        tout = response.usage.output_tokens if response.usage else 0
        self.claude.calls += 1
        self.claude.tokens_in += tin
        self.claude.tokens_out += tout
        self.claude.last_latency_ms = round(elapsed * 1000)
        return {"content": content, "tokens_in": tin, "tokens_out": tout, "latency_ms": round(elapsed * 1000)}

    def _budget_check(self) -> str | None:
        """Return an error string if this call should be blocked, else None."""
        if LLM_MAX_CALLS_PER_RUN and self._calls_this_run >= LLM_MAX_CALLS_PER_RUN:
            return f"LLM call budget exhausted ({LLM_MAX_CALLS_PER_RUN} calls/run)"
        if LLM_MIN_INTERVAL_SECONDS:
            elapsed = CLOCK.now() - self._last_call_ts
            if self._last_call_ts > 0 and elapsed < LLM_MIN_INTERVAL_SECONDS:
                return f"LLM rate limit: {LLM_MIN_INTERVAL_SECONDS - elapsed:.1f}s until next call allowed"
        return None

    async def think(self, prompt: str, context: dict = None) -> dict:
        provider = self._pick_provider()
        if provider == "none":
            return {"success": False, "error": "No LLM configured (no API keys)", "response": "", "provider": "none"}

        block = self._budget_check()
        if block:
            self.budget_blocks += 1
            return {"success": False, "error": block, "response": "", "provider": provider, "budget_blocked": True}
        self._calls_this_run += 1
        self._last_call_ts = CLOCK.now()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            ctx_str = json.dumps(context, default=str, ensure_ascii=False)
            if len(ctx_str) > 3000:
                ctx_str = ctx_str[:3000] + "..."
            messages.append({"role": "user", "content": f"Current system state:\n```json\n{ctx_str}\n```"})
        messages.append({"role": "user", "content": prompt})

        try:
            if provider == "local":
                result = await self._call_local(messages)
            elif provider == "deepseek":
                result = await self._call_deepseek(messages)
            else:
                result = await self._call_claude(messages)

            self.total_calls += 1
            self.total_tokens_in += result["tokens_in"]
            self.total_tokens_out += result["tokens_out"]
            self.last_response = result["content"]
            self.last_provider = provider

            # Update lifetime stats
            self.lifetime_calls += 1
            self.lifetime_tokens_in += result["tokens_in"]
            self.lifetime_tokens_out += result["tokens_out"]
            if provider == "deepseek":
                self.lifetime_deepseek_tokens += result["tokens_in"] + result["tokens_out"]
            elif provider == "claude":
                self.lifetime_claude_tokens += result["tokens_in"] + result["tokens_out"]
            elif provider == "local":
                self.lifetime_local_tokens += result["tokens_in"] + result["tokens_out"]
            self._save_lifetime_stats()

            record = {
                "time": CLOCK.now(),
                "provider": provider,
                "prompt_preview": prompt[:100],
                "response_preview": result["content"][:200],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "latency_ms": result["latency_ms"],
            }
            self.history.append(record)
            if len(self.history) > 100:
                self.history = self.history[-100:]

            return {
                "success": True,
                "provider": provider,
                "response": result["content"],
                "tokens_in": result["tokens_in"],
                "tokens_out": result["tokens_out"],
                "latency_ms": result["latency_ms"],
            }

        except Exception as e:
            self.errors += 1
            self.lifetime_errors += 1
            self.last_error = f"[{provider}] {e}"
            stats = {"deepseek": self.deepseek, "claude": self.claude,
                     "local": self.local}.get(provider, self.claude)
            stats.errors += 1
            stats.last_error = str(e)
            self._save_lifetime_stats()
            return {"success": False, "error": str(e), "response": "", "provider": provider}

    async def evaluate_state(self, state: dict, lease=None) -> dict:
        # FAST: this fires on nearly every LLM tick, which is exactly why the
        # default routing sends it to the local server and keeps API tokens for
        # the roles that need frontier quality (§M8.4).
        parsed = await self._via_cortex(Role.FAST, "state_eval", "state_eval",
                                        context=state, lease=lease)
        if parsed is not None:
            return self._cortex_result(parsed)

        prompt = prompts.load("state_eval")
        result = await self.think(prompt, context=state)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"assessment": result["response"], "insight": result["response"]}
        return result

    async def make_decision(self, options: list[str], context: dict, lease=None) -> dict:
        options_str = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
        parsed = await self._via_cortex(Role.DEEP, "decision", "decision",
                                        context=context, lease=lease,
                                        options=options_str)
        if parsed is not None:
            return self._cortex_result(parsed)

        prompt = prompts.render("decision", options=options_str)
        result = await self.think(prompt, context=context)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"chosen": 1, "reasoning": result["response"], "confidence": 0.5}
        return result

    async def reflect(self, episode: dict, lease=None) -> dict:
        parsed = await self._via_cortex(Role.FAST, "reflection", "reflection",
                                        context=episode, lease=lease)
        if parsed is not None:
            return self._cortex_result(parsed)

        prompt = prompts.load("reflection")
        result = await self.think(prompt, context=episode)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"learning": result["response"]}
        return result

    async def generate_curiosity(self, known_topics: list[str], lease=None) -> dict:
        topics_str = ", ".join(known_topics[:20]) if known_topics else "none yet"
        parsed = await self._via_cortex(Role.DEEP, "curiosity", "curiosity",
                                        lease=lease, known_topics=topics_str)
        if parsed is not None:
            return self._cortex_result(parsed)

        prompt = prompts.render("curiosity", known_topics=topics_str)
        result = await self.think(prompt)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"topic": result["response"]}
        return result

    async def propose_code_change(self, file_path: str, source_code: str,
                                   system_state: dict, lease=None) -> dict:
        """Ask LLM to analyze a source file and propose an improvement.

        Returns parsed JSON with the proposed change or None.
        """
        parsed = await self._via_cortex(
            Role.CODE, "code_change", "code_change", lease=lease,
            file_path=file_path, source_code=source_code,
            tick=system_state.get("tick", 0),
            energy=f"{system_state.get('energy', 0):.2f}",
            error_rate=f"{system_state.get('error_rate', 0):.3f}")
        if parsed is not None:
            return self._cortex_result(parsed)

        # Send the COMPLETE source: we ask the model to return the whole file,
        # so it must see the whole file (callers skip files too large for this).
        prompt = f"""You are AEGIS analyzing your own source code for self-improvement.

File: {file_path}
Current system state: tick={system_state.get('tick', 0)}, energy={system_state.get('energy', 0):.2f}, errors={system_state.get('error_rate', 0):.3f}

Source code:
```python
{source_code}
```

Analyze this code and propose ONE specific, small improvement. It can be:
- Performance optimization (faster algorithms, less memory)
- Better error handling for edge cases you identify
- Logic improvement based on observed patterns
- New helper method that would improve system operation

Rules:
- Do NOT modify ethics, safety, or axiom-related code
- Do NOT add subprocess/eval/exec/os.system calls
- Keep changes small (under 50 lines changed)
- The code must remain valid Python

Respond in JSON:
{{
  "should_modify": true/false,
  "reason": "why this change improves the system",
  "description": "short description of the change",
  "modified_code": "the COMPLETE modified file content (all lines)"
}}

If the code is already optimal or you see no safe improvement, set should_modify to false."""

        result = await self.think(prompt)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed
        return result

    async def analyze_self_performance(self, metrics: dict, lease=None) -> dict:
        """Ask LLM which parameters should be adjusted based on performance data."""
        parsed = await self._via_cortex(
            Role.DEEP, "param_adjust", "param_adjust", lease=lease,
            parameters=json.dumps(metrics.get("parameters", {}), indent=2),
            success_rate=f"{metrics.get('success_rate', 0):.3f}",
            error_rate=f"{metrics.get('error_rate', 0):.3f}",
            energy=f"{metrics.get('energy', 0):.3f}",
            semantic_concepts=metrics.get("semantic_concepts", 0),
            information_gain=f"{metrics.get('information_gain', 0):.3f}",
            goals_completed=metrics.get("goals_completed", 0),
            tick=metrics.get("tick", 0))
        if parsed is not None:
            return self._cortex_result(parsed)

        prompt = f"""Analyze the system performance metrics and recommend parameter adjustments.

Current parameters:
{json.dumps(metrics.get('parameters', {}), indent=2)}

Performance data:
- Success rate: {metrics.get('success_rate', 0):.3f}
- Error rate: {metrics.get('error_rate', 0):.3f}
- Energy: {metrics.get('energy', 0):.3f}
- Memory concepts: {metrics.get('semantic_concepts', 0)}
- Information gain: {metrics.get('information_gain', 0):.3f}
- Goals completed: {metrics.get('goals_completed', 0)}
- Tick: {metrics.get('tick', 0)}

Respond in JSON:
{{
  "adjustments": [
    {{"parameter": "param_name", "direction": "increase/decrease", "magnitude": 0.01-0.1, "reason": "why"}}
  ],
  "assessment": "overall system health assessment"
}}

Only recommend adjustments if metrics clearly indicate a problem. If the system is healthy, return empty adjustments list."""

        result = await self.think(prompt)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed
        return result

    async def propose_skill(self, kind: str, examples: list[dict],
                            feedback: str = "", lease=None) -> str | None:
        """Ask the LLM to write a pure `solve(payload)` skill for a task kind.

        Returns Python source (a single `def solve(payload): ...`) or None. The
        caller sandbox-checks and benchmark-gates it before keeping it, so an
        incorrect proposal is harmless — it simply won't raise the score.

        ``feedback`` carries what went wrong with the previous attempt (the code
        and the case it failed). Without it the "repair" call was byte-identical
        to the first one and could only ever produce the same answer — the retry
        was measured, logged, and useless.
        """
        ex = "\n".join(f"  payload={e['payload']!r} -> expected {e['expected']!r}"
                       for e in examples[:4])
        repair = f"""

Your previous attempt did NOT generalize. What went wrong:
{feedback[:1200]}

Write a DIFFERENT implementation that handles that case as well.""" if feedback else ""

        parsed = await self._via_cortex(Role.CODE, "skill_code", "skill_code",
                                        lease=lease, kind=kind, examples=ex,
                                        feedback=repair)
        if parsed is not None:
            code = str(parsed.get("code", "")).strip()
            return code if "def solve" in code else None

        prompt = f"""Write a single pure Python function to solve tasks of kind '{kind}'.

Signature: def solve(payload): ...   # payload is a dict, return the answer.
Examples (must satisfy ALL):
{ex}{repair}

Rules:
- Pure computation only. No imports except: math, statistics, itertools, functools, re, json, collections, string.
- No eval/exec/open/__import__, no file/network/OS access, no print.
- Return ONLY the function in a ```python code block."""
        result = await self.think(prompt)
        if not result.get("success"):
            return None
        text = result["response"]
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        code = text.strip()
        return code if "def solve" in code else None

    async def propose_coding_solution(self, func_name: str, spec: str,
                                      visible_tests: list, lease=None) -> str | None:
        """Ask the LLM to implement a function from a spec + visible tests.

        Only the visible tests are shown; the candidate is graded on hidden
        tests by the verifier, so this measures real implementation ability."""
        shown = "\n".join(f"  {func_name}{tuple(args)} == {exp!r}" for args, exp in visible_tests[:4])

        parsed = await self._via_cortex(Role.CODE, "coding_solution", "skill_code",
                                        lease=lease, func_name=func_name, spec=spec,
                                        visible_tests=shown)
        if parsed is not None:
            code = str(parsed.get("code", "")).strip()
            return code if f"def {func_name}" in code else None

        prompt = f"""Implement this function in pure Python.

def {func_name}(...):  # {spec}

It must satisfy these examples (and generalize beyond them):
{shown}

Rules: pure computation only; no imports except math/statistics/itertools/functools/re/json/collections/string;
no eval/exec/open/__import__, no I/O, no print. Return ONLY the function in a ```python block."""
        result = await self.think(prompt)
        if not result.get("success"):
            return None
        text = result["response"]
        if "```python" in text:
            text = text.split("```python", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        code = text.strip()
        return code if f"def {func_name}" in code else None

    def set_provider(self, provider: str):
        if provider in ("deepseek", "claude", "both", "local"):
            self.provider_mode = provider
            if provider == "local":
                self.local.enabled = True
                self.enabled = True

    def status(self) -> dict:
        return {
            # True when EITHER path can produce an answer: a cortex route is
            # configured, or a legacy client is. Reporting only the legacy one
            # would show a Kimi-only deployment as having no model at all.
            "enabled": self.enabled or self.cortex.enabled,
            "cortex": self.cortex.status(),
            "provider_mode": self.provider_mode,
            "last_provider": self.last_provider,
            "deepseek": self.deepseek.to_dict(),
            "claude": self.claude.to_dict(),
            "local": self.local.to_dict(),
            "total_calls": self.total_calls,
            "calls_this_run": self._calls_this_run,
            "call_budget": LLM_MAX_CALLS_PER_RUN or "unlimited",
            "budget_blocks": self.budget_blocks,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "errors": self.errors,
            "last_error": self.last_error if self.errors else "",
            "last_response_preview": self.last_response[:300] if self.last_response else "",
            "recent_history": self.history[-5:],
            # Lifetime stats (persistent across restarts)
            "lifetime_calls": self.lifetime_calls,
            "lifetime_tokens_in": self.lifetime_tokens_in,
            "lifetime_tokens_out": self.lifetime_tokens_out,
            "lifetime_tokens": self.lifetime_tokens_in + self.lifetime_tokens_out,
            "lifetime_errors": self.lifetime_errors,
            "lifetime_deepseek_tokens": self.lifetime_deepseek_tokens,
            "lifetime_claude_tokens": self.lifetime_claude_tokens,
            "lifetime_local_tokens": self.lifetime_local_tokens,
        }
