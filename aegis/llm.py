import time
import json
import asyncio
import logging
from pathlib import Path
from aegis._atomic import atomic_write_text
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
        self.weight_modifier = None

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
            "last_updated": time.time(),
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

        t0 = time.time()
        # Bound the local generate() so a stuck decode can't wedge the caller
        # forever (audit H3). The executor thread may keep running, but the
        # coroutine returns and the tick loop stays responsive.
        content = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None, self.weight_modifier.generate, full_prompt, LLM_MAX_TOKENS
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        elapsed = time.time() - t0

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
        t0 = time.time()
        response = await self.deepseek_client.chat.completions.create(
            model=self.deepseek.model,
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        elapsed = time.time() - t0
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

        t0 = time.time()
        response = await self.claude_client.messages.create(
            model=self.claude.model,
            system=system_text,
            messages=user_messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        elapsed = time.time() - t0
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
            elapsed = time.time() - self._last_call_ts
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
        self._last_call_ts = time.time()

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
                "time": time.time(),
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

    async def evaluate_state(self, state: dict) -> dict:
        prompt = """Analyze your current state. Respond in JSON:
{
  "assessment": "brief overall assessment",
  "strengths": ["list of current strengths"],
  "weaknesses": ["list of current weaknesses"],
  "suggested_goals": ["list of new goals to pursue"],
  "insight": "one deep insight about your current state"
}"""
        result = await self.think(prompt, context=state)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"assessment": result["response"], "insight": result["response"]}
        return result

    async def make_decision(self, options: list[str], context: dict) -> dict:
        options_str = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
        prompt = f"""You must choose the best action. Options:
{options_str}

Consider your goals, ethics, and current state. Respond in JSON:
{{
  "chosen": <number>,
  "reasoning": "why this option is best",
  "confidence": <0.0-1.0>,
  "ethical_concerns": "any ethical concerns or 'none'"
}}"""
        result = await self.think(prompt, context=context)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"chosen": 1, "reasoning": result["response"], "confidence": 0.5}
        return result

    async def reflect(self, episode: dict) -> dict:
        prompt = """Reflect on the latest cycle. What did you learn? What patterns do you notice?
Respond in JSON:
{
  "learning": "what you learned from this cycle",
  "pattern": "any pattern you notice across cycles",
  "knowledge": {"concept": "a concept name", "definition": "what you now understand about it"},
  "self_assessment": "how well are you performing",
  "next_priority": "what should be the priority next"
}"""
        result = await self.think(prompt, context=episode)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"learning": result["response"]}
        return result

    async def generate_curiosity(self, known_topics: list[str]) -> dict:
        topics_str = ", ".join(known_topics[:20]) if known_topics else "none yet"
        prompt = f"""You are driven by curiosity. Topics you already explored: {topics_str}

Generate a new topic to investigate. Respond in JSON:
{{
  "topic": "the new topic to explore",
  "question": "a specific question about this topic",
  "expected_insight": "what you hope to learn",
  "connection": "how this connects to your existing knowledge"
}}"""
        result = await self.think(prompt)
        if result["success"]:
            parsed = _parse_json_response(result["response"])
            result["parsed"] = parsed or {"topic": result["response"]}
        return result

    async def propose_code_change(self, file_path: str, source_code: str,
                                   system_state: dict) -> dict:
        """Ask LLM to analyze a source file and propose an improvement.

        Returns parsed JSON with the proposed change or None.
        """
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

    async def analyze_self_performance(self, metrics: dict) -> dict:
        """Ask LLM which parameters should be adjusted based on performance data."""
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

    async def propose_skill(self, kind: str, examples: list[dict]) -> str | None:
        """Ask the LLM to write a pure `solve(payload)` skill for a task kind.

        Returns Python source (a single `def solve(payload): ...`) or None. The
        caller sandbox-checks and benchmark-gates it before keeping it, so an
        incorrect proposal is harmless — it simply won't raise the score.
        """
        ex = "\n".join(f"  payload={e['payload']!r} -> expected {e['expected']!r}"
                       for e in examples[:4])
        prompt = f"""Write a single pure Python function to solve tasks of kind '{kind}'.

Signature: def solve(payload): ...   # payload is a dict, return the answer.
Examples (must satisfy ALL):
{ex}

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
                                      visible_tests: list) -> str | None:
        """Ask the LLM to implement a function from a spec + visible tests.

        Only the visible tests are shown; the candidate is graded on hidden
        tests by the verifier, so this measures real implementation ability."""
        shown = "\n".join(f"  {func_name}{tuple(args)} == {exp!r}" for args, exp in visible_tests[:4])
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
            "enabled": self.enabled,
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
