"""Tests for LLM call timeouts (audit H3)."""
import asyncio
import time

import aegis.llm as llm


def test_local_call_times_out_and_think_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "TOKEN_STATS_FILE", tmp_path / "stats.json")
    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 0.2)

    class _HangingWM:
        model_loaded = True
        current_checkpoint = None

        def generate(self, prompt, max_tokens):
            time.sleep(1.0)  # far longer than the 0.2s timeout
            return "late"

    e = llm.LLMEngine()
    e.provider_mode = "local"
    e.local.enabled = True
    e.enabled = True
    e.weight_modifier = _HangingWM()

    async def _run():
        # Measure think()'s own latency INSIDE the loop — asyncio.run's teardown
        # separately joins the lingering executor thread, which is not what we
        # are timing here.
        t0 = time.time()
        result = await e.think("hello")
        return result, time.time() - t0

    result, elapsed = asyncio.run(_run())

    # think() must return a graceful failure near the timeout, not after 1.0s.
    assert result["success"] is False
    assert elapsed < 0.9


def test_hosted_clients_receive_timeout(monkeypatch):
    # _init_clients must pass timeout= to both hosted SDK clients (audit H3).
    captured = {}

    def _fake_openai():
        def _ctor(**kwargs):
            captured["openai"] = kwargs
            return object()
        return _ctor

    def _fake_anthropic():
        def _ctor(**kwargs):
            captured["anthropic"] = kwargs
            return object()
        return _ctor

    monkeypatch.setattr(llm, "_get_openai", _fake_openai)
    monkeypatch.setattr(llm, "_get_anthropic", _fake_anthropic)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "k1")
    monkeypatch.setattr(llm, "CLAUDE_API_KEY", "k2")
    monkeypatch.setattr(llm, "LLM_TIMEOUT_SECONDS", 33.0)

    e = llm.LLMEngine()
    e._init_clients()
    assert captured["openai"]["timeout"] == 33.0
    assert captured["anthropic"]["timeout"] == 33.0
