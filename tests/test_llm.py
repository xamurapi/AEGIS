"""Unit tests for aegis/llm.py.

All external providers (openai.AsyncOpenAI, anthropic.AsyncAnthropic) and the
local model are mocked — no real network call or model load happens here. The
persistent token-stats file is redirected into tmp_path so the real data/ dir is
never touched.
"""
import json
import time
import asyncio

import pytest

import aegis.llm as llm
from aegis.llm import (
    _parse_json_response,
    ProviderStats,
    LLMEngine,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def stats_file(tmp_path, monkeypatch):
    """Redirect the persistent token-stats file into tmp_path."""
    p = tmp_path / "token_stats.json"
    monkeypatch.setattr(llm, "TOKEN_STATS_FILE", p)
    return p


@pytest.fixture
def engine(stats_file):
    """A fresh engine with no provider keys (offline)."""
    return LLMEngine()


# --------------------------------------------------------------------------- #
# Fake provider response objects                                              #
# --------------------------------------------------------------------------- #
class _Usage:
    def __init__(self, a, b, style="openai"):
        if style == "openai":
            self.prompt_tokens = a
            self.completion_tokens = b
        else:
            self.input_tokens = a
            self.output_tokens = b


class _OpenAIResp:
    def __init__(self, content, usage):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = usage


class _AnthResp:
    def __init__(self, text, usage):
        self.content = [type("B", (), {"text": text})()] if text is not None else []
        self.usage = usage


def _make_deepseek_client(content="hi", usage=_Usage(3, 5)):
    class _Completions:
        async def create(self, **kwargs):
            _make_deepseek_client.last_kwargs = kwargs
            return _OpenAIResp(content, usage)

    class _Chat:
        completions = _Completions()

    return type("Client", (), {"chat": _Chat()})()


def _make_claude_client(text="hi", usage=_Usage(4, 6, "anth")):
    captured = {}

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _AnthResp(text, usage)

    client = type("Client", (), {"messages": _Messages()})()
    client.captured = captured
    return client


# --------------------------------------------------------------------------- #
# _parse_json_response                                                        #
# --------------------------------------------------------------------------- #
def test_parse_plain_json():
    assert _parse_json_response('{"x": 1}') == {"x": 1}


def test_parse_json_fence():
    assert _parse_json_response('a\n```json\n{"y": 2}\n```') == {"y": 2}


def test_parse_generic_fence():
    assert _parse_json_response('```\n{"z": 3}\n```') == {"z": 3}


def test_parse_invalid_returns_none():
    assert _parse_json_response("nope") is None


def test_parse_fence_without_close_returns_none():
    # "```json" present but no closing fence -> IndexError -> None
    assert _parse_json_response("```json\n{not closed") is None


# --------------------------------------------------------------------------- #
# ProviderStats                                                               #
# --------------------------------------------------------------------------- #
def test_provider_stats_to_dict_no_error():
    s = ProviderStats("DeepSeek", "m")
    s.tokens_in, s.tokens_out = 10, 20
    s.last_error = "boom"  # hidden while errors == 0
    d = s.to_dict()
    assert d["tokens_total"] == 30
    assert d["last_error"] == ""


def test_provider_stats_to_dict_with_error():
    s = ProviderStats("Claude", "m")
    s.errors = 2
    s.last_error = "boom"
    assert s.to_dict()["last_error"] == "boom"


# --------------------------------------------------------------------------- #
# _load_lifetime_stats / _save_lifetime_stats                                 #
# --------------------------------------------------------------------------- #
def test_load_lifetime_stats_from_file(stats_file):
    stats_file.write_text(json.dumps({
        "lifetime_calls": 7,
        "lifetime_tokens_in": 11,
        "lifetime_tokens_out": 13,
        "lifetime_errors": 1,
        "lifetime_deepseek_tokens": 5,
        "lifetime_claude_tokens": 3,
        "lifetime_local_tokens": 2,
    }), encoding="utf-8")
    e = LLMEngine()
    assert e.lifetime_calls == 7
    assert e.lifetime_tokens_in == 11
    assert e.lifetime_local_tokens == 2


def test_load_lifetime_stats_corrupt_file(stats_file):
    stats_file.write_text("{ this is not json", encoding="utf-8")
    e = LLMEngine()  # must not raise; falls back to zeros
    assert e.lifetime_calls == 0


def test_save_lifetime_stats_roundtrip(engine, stats_file):
    engine.lifetime_calls = 42
    engine._save_lifetime_stats()
    data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert data["lifetime_calls"] == 42
    assert "last_updated" in data


def test_save_lifetime_stats_swallows_errors(engine, monkeypatch):
    class Boom:
        def write_text(self, *a, **k):
            raise OSError("disk full")

    monkeypatch.setattr(llm, "TOKEN_STATS_FILE", Boom())
    engine._save_lifetime_stats()  # must not raise


# --------------------------------------------------------------------------- #
# _init_clients                                                               #
# --------------------------------------------------------------------------- #
def test_init_clients_creates_deepseek(stats_file, monkeypatch):
    created = {}

    def fake_oai(**kwargs):
        created.update(kwargs)
        return "deepseek-client"

    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "dkey")
    monkeypatch.setattr(llm, "_get_openai", lambda: fake_oai)
    monkeypatch.setattr(llm, "CLAUDE_API_KEY", "")
    e = LLMEngine()
    assert e.deepseek_client == "deepseek-client"
    assert e.deepseek.enabled is True
    assert e.enabled is True
    assert created["api_key"] == "dkey"


def test_init_clients_creates_claude(stats_file, monkeypatch):
    def fake_anth(**kwargs):
        return "claude-client"

    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "CLAUDE_API_KEY", "ckey")
    monkeypatch.setattr(llm, "_get_anthropic", lambda: fake_anth)
    e = LLMEngine()
    assert e.claude_client == "claude-client"
    assert e.claude.enabled is True


def test_init_clients_local_mode(stats_file, monkeypatch):
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "CLAUDE_API_KEY", "")
    monkeypatch.setattr(llm, "LLM_PROVIDER", "local")
    e = LLMEngine()
    assert e.local.enabled is True
    assert e.enabled is True


def test_init_clients_none(engine):
    assert engine.enabled is False


# --------------------------------------------------------------------------- #
# _pick_provider                                                              #
# --------------------------------------------------------------------------- #
def test_pick_local_mode(engine):
    engine.provider_mode = "local"
    engine.local.enabled = True
    assert engine._pick_provider() == "local"


def test_pick_deepseek_mode(engine):
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True
    assert engine._pick_provider() == "deepseek"


def test_pick_claude_mode(engine):
    engine.provider_mode = "claude"
    engine.claude.enabled = True
    assert engine._pick_provider() == "claude"


def test_pick_both_alternates(engine):
    engine.provider_mode = "both"
    engine.deepseek.enabled = True
    engine.claude.enabled = True
    first = engine._pick_provider()   # counter -> 1 (odd) -> deepseek
    second = engine._pick_provider()  # counter -> 2 (even) -> claude
    assert first == "deepseek"
    assert second == "claude"


def test_pick_both_only_deepseek(engine):
    engine.provider_mode = "both"
    engine.deepseek.enabled = True
    engine.claude.enabled = False
    assert engine._pick_provider() == "deepseek"


def test_pick_both_only_claude(engine):
    engine.provider_mode = "both"
    engine.deepseek.enabled = False
    engine.claude.enabled = True
    assert engine._pick_provider() == "claude"


def test_pick_fallback_local(engine):
    engine.provider_mode = "deepseek"       # but deepseek disabled
    engine.deepseek.enabled = False
    engine.local.enabled = True
    assert engine._pick_provider() == "local"


def test_pick_fallback_deepseek(engine):
    engine.provider_mode = "claude"
    engine.claude.enabled = False
    engine.deepseek.enabled = True
    assert engine._pick_provider() == "deepseek"


def test_pick_fallback_claude(engine):
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = False
    engine.claude.enabled = True
    assert engine._pick_provider() == "claude"


def test_pick_none(engine):
    engine.deepseek.enabled = engine.claude.enabled = engine.local.enabled = False
    assert engine._pick_provider() == "none"


# --------------------------------------------------------------------------- #
# _budget_check                                                               #
# --------------------------------------------------------------------------- #
def test_budget_exhausted(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 3)
    engine._calls_this_run = 3
    assert "budget" in engine._budget_check()


def test_budget_rate_limited(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 100.0)
    engine._last_call_ts = time.time()
    msg = engine._budget_check()
    assert "rate limit" in msg


def test_budget_ok(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    assert engine._budget_check() is None


# --------------------------------------------------------------------------- #
# _call_deepseek / _call_claude / _call_local                                 #
# --------------------------------------------------------------------------- #
def test_call_deepseek(engine):
    engine.deepseek_client = _make_deepseek_client("answer", _Usage(3, 7))
    res = asyncio.run(engine._call_deepseek([{"role": "user", "content": "q"}]))
    assert res["content"] == "answer"
    assert res["tokens_in"] == 3 and res["tokens_out"] == 7
    assert engine.deepseek.calls == 1


def test_call_deepseek_no_usage_and_null_content(engine):
    engine.deepseek_client = _make_deepseek_client(None, None)
    res = asyncio.run(engine._call_deepseek([{"role": "user", "content": "q"}]))
    assert res["content"] == ""
    assert res["tokens_in"] == 0 and res["tokens_out"] == 0


def test_call_claude(engine):
    engine.claude_client = _make_claude_client("cool", _Usage(2, 9, "anth"))
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"}]
    res = asyncio.run(engine._call_claude(msgs))
    assert res["content"] == "cool"
    assert res["tokens_in"] == 2 and res["tokens_out"] == 9
    assert engine.claude_client.captured["system"] == "sys"
    assert engine.claude.calls == 1


def test_call_claude_injects_default_user_message(engine):
    engine.claude_client = _make_claude_client("x")
    # only a system message -> user_messages empty -> "Think." injected
    asyncio.run(engine._call_claude([{"role": "system", "content": "s"}]))
    assert engine.claude_client.captured["messages"] == [{"role": "user", "content": "Think."}]


def test_call_claude_empty_content_and_no_usage(engine):
    engine.claude_client = _make_claude_client(None, None)
    res = asyncio.run(engine._call_claude([{"role": "user", "content": "hi"}]))
    assert res["content"] == ""
    assert res["tokens_in"] == 0 and res["tokens_out"] == 0


def test_call_local_not_loaded_raises(engine):
    engine.weight_modifier = type("W", (), {"model_loaded": False})()
    with pytest.raises(RuntimeError):
        asyncio.run(engine._call_local([{"role": "user", "content": "hi"}]))


def test_call_local_success(engine):
    class FakeWM:
        model_loaded = True
        current_checkpoint = "ckpt-1"

        def generate(self, prompt, max_tokens):
            FakeWM.seen_prompt = prompt
            return "local-out"

    engine.weight_modifier = FakeWM()
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    res = asyncio.run(engine._call_local(msgs))
    assert res["content"] == "local-out"
    assert engine.local.calls == 1
    assert engine.local.model == "ckpt-1"
    assert "System: S" in FakeWM.seen_prompt
    assert FakeWM.seen_prompt.endswith("Assistant:")


def test_call_local_truncates_long_prompt(engine):
    long_content = "z" * 6000

    class FakeWM:
        model_loaded = True
        current_checkpoint = None

        def generate(self, prompt, max_tokens):
            FakeWM.seen_len = len(prompt)
            return "ok"

    engine.weight_modifier = FakeWM()
    asyncio.run(engine._call_local([{"role": "user", "content": long_content}]))
    assert FakeWM.seen_len == 4000
    assert engine.local.model == "base"  # current_checkpoint None -> "base"


# --------------------------------------------------------------------------- #
# think()                                                                     #
# --------------------------------------------------------------------------- #
def test_think_no_provider(engine):
    res = asyncio.run(engine.think("hi"))
    assert res["success"] is False
    assert res["provider"] == "none"


def test_think_budget_blocked(engine, monkeypatch):
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 1)
    engine._calls_this_run = 1
    res = asyncio.run(engine.think("hi"))
    assert res["success"] is False
    assert res["budget_blocked"] is True
    assert engine.budget_blocks == 1


def test_think_success_deepseek_updates_stats(engine, monkeypatch, stats_file):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True
    engine.deepseek_client = _make_deepseek_client("resp", _Usage(4, 6))
    res = asyncio.run(engine.think("prompt", context={"k": "v"}))
    assert res["success"] is True
    assert res["response"] == "resp"
    assert engine.total_calls == 1
    assert engine.total_tokens_in == 4 and engine.total_tokens_out == 6
    assert engine.lifetime_deepseek_tokens == 10
    assert len(engine.history) == 1
    # lifetime stats persisted
    data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert data["lifetime_calls"] == 1


def test_think_context_truncated(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True

    captured = {}

    class C:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    captured.update(kwargs)
                    return _OpenAIResp("ok", _Usage(1, 1))

    engine.deepseek_client = C()
    big_ctx = {"data": "x" * 5000}
    asyncio.run(engine.think("p", context=big_ctx))
    user_msg = captured["messages"][1]["content"]
    assert "..." in user_msg  # context was truncated


def test_think_claude_lifetime_tokens(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "claude"
    engine.claude.enabled = True
    engine.claude_client = _make_claude_client("hey", _Usage(5, 5, "anth"))
    asyncio.run(engine.think("p"))
    assert engine.lifetime_claude_tokens == 10


def test_think_local_lifetime_tokens(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "local"
    engine.local.enabled = True
    engine.weight_modifier = type("W", (), {
        "model_loaded": True,
        "current_checkpoint": None,
        "generate": lambda self, p, m: "abcd",
    })()
    asyncio.run(engine.think("p"))
    assert engine.lifetime_local_tokens >= 1


def test_think_history_trimmed(engine, monkeypatch):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True
    engine.deepseek_client = _make_deepseek_client("r", _Usage(1, 1))
    engine.history = [{"i": i} for i in range(100)]
    asyncio.run(engine.think("p"))
    assert len(engine.history) == 100  # trimmed back to last 100


def test_think_exception_path(engine, monkeypatch, stats_file):
    monkeypatch.setattr(llm, "LLM_MAX_CALLS_PER_RUN", 0)
    monkeypatch.setattr(llm, "LLM_MIN_INTERVAL_SECONDS", 0)
    engine.provider_mode = "deepseek"
    engine.deepseek.enabled = True

    class Boom:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("api down")

    engine.deepseek_client = Boom()
    res = asyncio.run(engine.think("p"))
    assert res["success"] is False
    assert "api down" in res["error"]
    assert engine.errors == 1
    assert engine.lifetime_errors == 1
    assert engine.deepseek.errors == 1


# --------------------------------------------------------------------------- #
# Task wrappers (with think() mocked)                                         #
# --------------------------------------------------------------------------- #
def _stub_think(engine, response, success=True):
    async def fake_think(prompt, context=None):
        fake_think.prompt = prompt
        fake_think.context = context
        return {"success": success, "response": response, "provider": "x"}
    engine.think = fake_think
    return fake_think


def test_evaluate_state_parsed(engine):
    _stub_think(engine, '{"assessment": "ok", "insight": "deep"}')
    res = asyncio.run(engine.evaluate_state({"tick": 1}))
    assert res["parsed"]["assessment"] == "ok"


def test_evaluate_state_fallback(engine):
    _stub_think(engine, "unparseable text")
    res = asyncio.run(engine.evaluate_state({}))
    assert res["parsed"]["assessment"] == "unparseable text"


def test_evaluate_state_failure(engine):
    _stub_think(engine, "", success=False)
    res = asyncio.run(engine.evaluate_state({}))
    assert "parsed" not in res


def test_make_decision_parsed(engine):
    stub = _stub_think(engine, '{"chosen": 2, "confidence": 0.9}')
    res = asyncio.run(engine.make_decision(["a", "b"], {"ctx": 1}))
    assert res["parsed"]["chosen"] == 2
    assert "1. a" in stub.prompt and "2. b" in stub.prompt


def test_make_decision_fallback(engine):
    _stub_think(engine, "nope")
    res = asyncio.run(engine.make_decision(["a"], {}))
    assert res["parsed"]["chosen"] == 1


def test_reflect_parsed_and_fallback(engine):
    _stub_think(engine, '{"learning": "L"}')
    assert asyncio.run(engine.reflect({}))["parsed"]["learning"] == "L"
    _stub_think(engine, "plain")
    assert asyncio.run(engine.reflect({}))["parsed"]["learning"] == "plain"


def test_generate_curiosity_with_topics(engine):
    stub = _stub_think(engine, '{"topic": "T"}')
    res = asyncio.run(engine.generate_curiosity(["math", "logic"]))
    assert res["parsed"]["topic"] == "T"
    assert "math" in stub.prompt


def test_generate_curiosity_empty_topics_fallback(engine):
    stub = _stub_think(engine, "raw topic")
    res = asyncio.run(engine.generate_curiosity([]))
    assert res["parsed"]["topic"] == "raw topic"
    assert "none yet" in stub.prompt


def test_propose_code_change(engine):
    _stub_think(engine, '{"should_modify": false}')
    res = asyncio.run(engine.propose_code_change("f.py", "code", {"tick": 1}))
    assert res["parsed"] == {"should_modify": False}


def test_propose_code_change_failure(engine):
    _stub_think(engine, "", success=False)
    res = asyncio.run(engine.propose_code_change("f.py", "code", {}))
    assert "parsed" not in res


def test_analyze_self_performance(engine):
    _stub_think(engine, '{"adjustments": [], "assessment": "healthy"}')
    res = asyncio.run(engine.analyze_self_performance({"parameters": {"a": 1}}))
    assert res["parsed"]["assessment"] == "healthy"


def test_propose_skill_python_block(engine):
    _stub_think(engine, "```python\ndef solve(payload):\n    return 1\n```")
    code = asyncio.run(engine.propose_skill("k", [{"payload": {}, "expected": 1}]))
    assert "def solve" in code


def test_propose_skill_generic_block(engine):
    _stub_think(engine, "```\ndef solve(payload):\n    return 2\n```")
    code = asyncio.run(engine.propose_skill("k", [{"payload": {}, "expected": 2}]))
    assert "def solve" in code


def test_propose_skill_no_solve_returns_none(engine):
    _stub_think(engine, "```python\nx = 1\n```")
    assert asyncio.run(engine.propose_skill("k", [])) is None


def test_propose_skill_think_fails(engine):
    _stub_think(engine, "", success=False)
    assert asyncio.run(engine.propose_skill("k", [])) is None


def test_propose_coding_solution(engine):
    _stub_think(engine, "```python\ndef foo(a):\n    return a\n```")
    code = asyncio.run(engine.propose_coding_solution("foo", "id", [((1,), 1)]))
    assert "def foo" in code


def test_propose_coding_solution_generic_block(engine):
    _stub_think(engine, "```\ndef foo(a):\n    return a\n```")
    code = asyncio.run(engine.propose_coding_solution("foo", "id", [((1,), 1)]))
    assert "def foo" in code


def test_propose_coding_solution_wrong_name_returns_none(engine):
    _stub_think(engine, "```python\ndef bar(a):\n    return a\n```")
    assert asyncio.run(engine.propose_coding_solution("foo", "s", [])) is None


def test_propose_coding_solution_think_fails(engine):
    _stub_think(engine, "", success=False)
    assert asyncio.run(engine.propose_coding_solution("foo", "s", [])) is None


# --------------------------------------------------------------------------- #
# set_provider / status                                                       #
# --------------------------------------------------------------------------- #
def test_set_provider_valid(engine):
    engine.set_provider("claude")
    assert engine.provider_mode == "claude"


def test_set_provider_local_enables(engine):
    engine.set_provider("local")
    assert engine.provider_mode == "local"
    assert engine.local.enabled is True
    assert engine.enabled is True


def test_set_provider_invalid_ignored(engine):
    engine.provider_mode = "both"
    engine.set_provider("bogus")
    assert engine.provider_mode == "both"


def test_status_shape(engine):
    engine.errors = 1
    engine.last_error = "err"
    engine.last_response = "resp"
    st = engine.status()
    assert st["errors"] == 1
    assert st["last_error"] == "err"
    assert st["last_response_preview"] == "resp"
    assert "deepseek" in st and "claude" in st and "local" in st
    assert "lifetime_calls" in st
