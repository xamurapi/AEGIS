"""Tests for LLM helpers: JSON parsing and call budget."""
import aegis.config as cfg
from aegis.llm import _parse_json_response, LLMEngine


def test_parse_plain_json():
    assert _parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_fenced_json():
    text = 'here you go:\n```json\n{"chosen": 2}\n```\n'
    assert _parse_json_response(text) == {"chosen": 2}


def test_parse_invalid_returns_none():
    assert _parse_json_response("not json at all") is None


def test_budget_check_blocks_when_exhausted(monkeypatch):
    monkeypatch.setattr("aegis.llm.LLM_MAX_CALLS_PER_RUN", 2)
    engine = LLMEngine()
    engine._calls_this_run = 2
    assert engine._budget_check() is not None


def test_budget_check_allows_under_limit(monkeypatch):
    monkeypatch.setattr("aegis.llm.LLM_MAX_CALLS_PER_RUN", 5)
    monkeypatch.setattr("aegis.llm.LLM_MIN_INTERVAL_SECONDS", 0)
    engine = LLMEngine()
    engine._calls_this_run = 1
    assert engine._budget_check() is None


def test_provider_none_when_no_keys():
    engine = LLMEngine()
    engine.deepseek.enabled = False
    engine.claude.enabled = False
    engine.local.enabled = False
    assert engine._pick_provider() == "none"
