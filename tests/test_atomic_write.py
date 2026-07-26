"""Tests for the atomic-write helper and its use in stat persistence (audit A3)."""
import json

from aegis._atomic import atomic_write_text


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, '{"a": 1}')
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}


def test_atomic_write_leaves_no_temp(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "hello")
    assert not (tmp_path / "x.json.tmp").exists()


def test_atomic_write_overwrites_completely(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_text(p, "aaaaaaaaaa")   # 10 chars
    atomic_write_text(p, "bb")            # shorter — must fully replace
    assert p.read_text(encoding="utf-8") == "bb"


def test_atomic_write_preserves_lf_newlines(tmp_path):
    p = tmp_path / "x.txt"
    atomic_write_text(p, "a\nb\nc\n")
    assert p.read_bytes() == b"a\nb\nc\n"  # no CRLF translation


def test_llm_stats_save_is_atomic(tmp_path, monkeypatch):
    # LLMEngine._save_lifetime_stats routes through atomic_write_text now.
    import aegis.llm as llm
    monkeypatch.setattr(llm, "TOKEN_STATS_FILE", tmp_path / "token_stats.json")
    e = llm.LLMEngine()
    e.lifetime_calls = 42
    e._save_lifetime_stats()
    data = json.loads((tmp_path / "token_stats.json").read_text(encoding="utf-8"))
    assert data["lifetime_calls"] == 42
    assert not (tmp_path / "token_stats.json.tmp").exists()
