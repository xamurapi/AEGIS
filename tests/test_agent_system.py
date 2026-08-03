"""Unit tests for aegis.layers.agent_system.

All HTTP is mocked — no test performs a real network request. httpx.AsyncClient
is replaced with a fake async-context-manager client whose .get() is driven by a
per-test URL handler.
"""
import asyncio
import time

import pytest

import aegis.layers.agent_system as ags
from aegis.layers.agent_system import AgentSystem, SpiderAgent, AGENT_BLUEPRINTS


# ── Fake httpx plumbing ───────────────────────────────────────────────────

class FakeResp:
    def __init__(self, status_code=200, text="", json_data=None, json_exc=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_exc = json_exc

    def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data


def make_client(handler):
    """handler(url) -> FakeResp (or raises)."""
    class FakeClient:
        def __init__(self, *a, **k):
            self.args = a
            self.kwargs = k

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, *a, **k):
            return handler(url)

    return FakeClient


def install(monkeypatch, handler):
    monkeypatch.setattr(ags.httpx, "AsyncClient", make_client(handler))


# Reuse a single event loop across tests. Creating a fresh loop per asyncio.run
# call exhausts Windows socket buffers (WinError 10055) under many async tests.
_LOOP = asyncio.new_event_loop()


def run(coro):
    return _LOOP.run_until_complete(coro)


# ── SpiderAgent dataclass ─────────────────────────────────────────────────

def test_spider_agent_success_rate_and_is_due():
    a = SpiderAgent(agent_id="x", name="n", source_type="arxiv", task_description="t")
    assert a.success_rate() == 0.0  # runs==0 -> max(runs,1)==1
    a.runs, a.successes = 4, 2
    assert a.success_rate() == 0.5
    # is_due depends on status + next_run
    a.status = "active"
    a.next_run = time.time() - 1
    assert a.is_due() is True
    a.status = "created"
    assert a.is_due() is False


def test_spider_agent_to_dict_truncates_and_previews():
    a = SpiderAgent(agent_id="x", name="n", source_type="wikipedia", task_description="t")
    a.last_error = "E" * 200
    a.last_data = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]
    d = a.to_dict()
    assert len(d["last_error"]) == 80
    assert len(d["last_data_preview"]) == 3  # only first 3 previewed
    # empty last_error branch
    a.last_error = ""
    assert a.to_dict()["last_error"] == ""


# ── Construction / lifecycle ──────────────────────────────────────────────

def test_auto_initialize_creates_blueprint_agents_once():
    s = AgentSystem()
    s.auto_initialize()
    assert len(s.agents) == len(AGENT_BLUEPRINTS)
    assert all(a.status == "active" for a in s.agents)
    n = len(s.agents)
    s.auto_initialize()  # idempotent
    assert len(s.agents) == n


def test_create_agent_public_api():
    s = AgentSystem()
    a = s.create_agent("custom", "arxiv", "do stuff", topic="ai")
    assert a.status == "active"
    assert a.next_run > time.time()
    assert s.total_generated == 1
    assert a in s.agents


# ── run_due_agents orchestration (fetchers mocked at instance level) ───────

def test_run_due_agents_no_http(monkeypatch):
    monkeypatch.setattr(ags, "HAS_HTTP", False)
    s = AgentSystem()
    assert run(s.run_due_agents()) == []


def _due_agent(s, source="arxiv"):
    s._initialized = True  # skip auto_initialize default creation
    a = s._create("ag", source, "task", topic="t")
    a.status = "active"
    a.next_run = time.time() - 1
    return a


def test_run_due_agents_success_path(monkeypatch):
    s = AgentSystem()
    a = _due_agent(s)

    async def fake_exec(agent):
        return [{"title": "T1", "summary": "S1"}, {"title": "T2", "summary": "S2"}]

    monkeypatch.setattr(s, "_execute_agent", fake_exec)
    results = run(s.run_due_agents())
    assert results and results[0]["success"] is True
    assert results[0]["items"] == 2
    assert a.successes == 1
    assert a.data_collected == 2
    assert s.total_data_items == 2
    assert len(s.collected_knowledge) == 2
    # a "fetched" event was logged
    assert any(e["event"] == "fetched" for e in s.generation_log)


def test_run_due_agents_empty_result_marks_failure(monkeypatch):
    s = AgentSystem()
    a = _due_agent(s)

    async def fake_exec(agent):
        return []

    monkeypatch.setattr(s, "_execute_agent", fake_exec)
    run(s.run_due_agents())
    assert a.failures == 1
    assert a.last_error == "Empty result"


def test_run_due_agents_skips_not_due(monkeypatch):
    s = AgentSystem()
    s._initialized = True
    a = s._create("ag", "arxiv", "task")
    a.status = "active"
    a.next_run = time.time() + 999  # not due

    async def fake_exec(agent):  # pragma: no cover - must not be called
        raise AssertionError("should not run")

    monkeypatch.setattr(s, "_execute_agent", fake_exec)
    assert run(s.run_due_agents()) == []


def test_run_due_agents_exception_marks_failed_when_persistent(monkeypatch):
    s = AgentSystem()
    a = _due_agent(s)
    # Pre-load history so a single failure crosses the failed threshold.
    a.failures = 9
    a.runs = 10
    a.successes = 0

    async def boom(agent):
        raise RuntimeError("network down")

    monkeypatch.setattr(s, "_execute_agent", boom)
    run(s.run_due_agents())
    assert a.failures == 10
    assert a.success_rate() < 0.2
    assert a.status == "failed"
    assert "network down" in a.last_error


# ── _execute_agent dispatch ───────────────────────────────────────────────

def test_execute_agent_dispatches_all_sources(monkeypatch):
    s = AgentSystem()

    calls = {}

    async def mk(name):
        async def f(*a, **k):
            calls[name] = True
            return [{"ok": name}]
        return f

    monkeypatch.setattr(s, "_fetch_arxiv", run(mk("arxiv")))
    monkeypatch.setattr(s, "_fetch_wikipedia", run(mk("wiki")))
    monkeypatch.setattr(s, "_fetch_quotes", run(mk("quotes")))
    monkeypatch.setattr(s, "_fetch_github", run(mk("github")))
    monkeypatch.setattr(s, "_fetch_news", run(mk("news")))

    for src, key in [("arxiv", "arxiv"), ("wikipedia", "wiki"),
                     ("quotes", "quotes"), ("github", "github"), ("news", "news")]:
        ag = SpiderAgent(agent_id="i", name="custom", source_type=src, task_description="t", topic="x")
        out = run(s._execute_agent(ag))
        assert out == [{"ok": key}]

    # unknown source returns []
    ag = SpiderAgent(agent_id="i", name="custom", source_type="mystery", task_description="t")
    assert run(s._execute_agent(ag)) == []


def test_execute_agent_reassigns_topic_from_blueprint(monkeypatch):
    s = AgentSystem()

    async def f(topic):
        return [{"topic": topic}]

    monkeypatch.setattr(s, "_fetch_arxiv", f)
    # name matches a blueprint that has topics -> topic gets reassigned
    ag = SpiderAgent(agent_id="i", name="arxiv_scout", source_type="arxiv",
                     task_description="t", topic="")
    run(s._execute_agent(ag))
    assert ag.topic != ""  # picked from blueprint topics


def test_execute_agent_rotates_topic_for_evolved_agents(monkeypatch):
    """Evolved replacements are named "{blueprint}_vN" (see evolve()), so an
    exact-name match against the blueprint never hit and an evolved agent
    re-fetched its initial topic forever. The blueprint is matched by
    source_type, exactly as evolve() itself matches it."""
    s = AgentSystem()
    fetched = []

    async def f(topic):
        fetched.append(topic)
        return [{"topic": topic}]

    monkeypatch.setattr(s, "_fetch_wikipedia", f)
    ag = SpiderAgent(agent_id="i", name="wiki_explorer_v3", source_type="wikipedia",
                     task_description="t", topic="stale_initial_topic")
    run(s._execute_agent(ag))
    run(s._execute_agent(ag))

    bp = next(b for b in AGENT_BLUEPRINTS if b["source_type"] == "wikipedia")
    assert ag.topic in bp["topics"]        # rotated onto the blueprint's list
    assert fetched[0] != "stale_initial_topic"
    assert fetched[0] != fetched[1]        # and it keeps rotating


# ── Fetchers with mocked httpx ────────────────────────────────────────────

def test_fetch_arxiv_parses_entries(monkeypatch):
    text = (
        "<feed>"
        "<entry><title>Deep\nLearning</title>"
        "<summary>  A study of deep nets. </summary>"
        '<link href="http://arxiv.org/abs/1234"/></entry>'
        "<entry><title></title><summary>skipme</summary></entry>"
        "</feed>"
    )
    install(monkeypatch, lambda url: FakeResp(200, text=text))
    s = AgentSystem()
    out = run(s._fetch_arxiv("neural nets"))
    assert len(out) == 1  # empty-title entry skipped
    assert out[0]["title"] == "Deep Learning"
    assert out[0]["url"] == "http://arxiv.org/abs/1234"
    assert out[0]["source"] == "arxiv"


def test_fetch_arxiv_non_200(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(503, text="err"))
    s = AgentSystem()
    assert run(s._fetch_arxiv("x")) == []


def test_fetch_wikipedia_success_and_empty(monkeypatch):
    good = {"title": "AI", "extract": "Artificial intelligence is a field.",
            "content_urls": {"desktop": {"page": "http://wiki/AI"}}}
    install(monkeypatch, lambda url: FakeResp(200, json_data=good))
    s = AgentSystem()
    out = run(s._fetch_wikipedia("artificial intelligence"))
    assert out[0]["title"] == "AI"
    assert out[0]["url"] == "http://wiki/AI"

    install(monkeypatch, lambda url: FakeResp(200, json_data={"extract": ""}))
    assert run(s._fetch_wikipedia("empty")) == []


def test_fetch_wikipedia_non_200(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(404))
    s = AgentSystem()
    assert run(s._fetch_wikipedia("x")) == []


def test_fetch_quotes_success(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(200, json_data=[{"q": "Know thyself", "a": "Socrates"}]))
    s = AgentSystem()
    out = run(s._fetch_quotes())
    assert len(out) == 3  # loops 3 times
    assert all(o["source"] == "zenquotes" for o in out)


def test_fetch_quotes_bad_status_uses_fallback(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(500))
    s = AgentSystem()
    out = run(s._fetch_quotes())
    assert len(out) == 1
    assert out[0]["source"] == "fallback"


def test_fetch_quotes_exception_uses_fallback(monkeypatch):
    def boom(url):
        raise RuntimeError("down")
    install(monkeypatch, boom)
    s = AgentSystem()
    out = run(s._fetch_quotes())
    assert out[0]["source"] == "fallback"


def test_fetch_github_parses_items(monkeypatch):
    data = {"items": [
        {"full_name": "a/b", "description": "great repo", "html_url": "http://gh/ab",
         "stargazers_count": 42},
        {"full_name": "c/d", "description": None, "html_url": "http://gh/cd",
         "stargazers_count": 1},
    ]}
    install(monkeypatch, lambda url: FakeResp(200, json_data=data))
    s = AgentSystem()
    out = run(s._fetch_github("agents"))
    assert len(out) == 2
    assert out[0]["stars"] == 42
    assert out[1]["summary"] == ""  # None description coerced to ""


def test_fetch_github_non_200(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(403))
    s = AgentSystem()
    assert run(s._fetch_github("x")) == []


def test_fetch_news_parses_items(monkeypatch):
    text = (
        "<rss><channel>"
        "<item><title>Big AI News</title><link>http://n/1</link>"
        "<pubDate>Mon, 01</pubDate></item>"
        "<item><title></title><link>http://n/2</link><pubDate>Tue</pubDate></item>"
        "</channel></rss>"
    )
    install(monkeypatch, lambda url: FakeResp(200, text=text))
    s = AgentSystem()
    out = run(s._fetch_news("ai"))
    assert len(out) == 1  # empty-title item skipped
    assert out[0]["title"] == "Big AI News"
    assert out[0]["source"] == "google_news"


def test_fetch_news_non_200(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(500))
    s = AgentSystem()
    assert run(s._fetch_news("x")) == []


def test_xml_tag_found_and_missing():
    assert AgentSystem._xml_tag("<title>hi</title>", "title") == "hi"
    assert AgentSystem._xml_tag("no tags here", "title") == ""


# ── Evolution ─────────────────────────────────────────────────────────────

def test_evolve_retires_failed_and_replaces():
    s = AgentSystem()
    s._initialized = True
    a = s._create("arxiv_scout", "arxiv", "task")
    a.status = "failed"
    res = s.evolve()
    assert "arxiv_scout" in res["retired"]
    assert res["created"]  # a replacement was spawned
    assert res["cycle"] == 1
    # retired agent kept (within last 5) and a new active agent exists
    assert any(x.status == "active" for x in s.agents)


def test_evolve_retires_low_success_rate():
    s = AgentSystem()
    s._initialized = True
    a = s._create("wiki_explorer", "wikipedia", "task")
    a.status = "active"
    a.runs = 20
    a.successes = 1  # rate 0.05 < 0.15
    res = s.evolve()
    assert "wiki_explorer" in res["retired"]


def test_evolve_trims_retired_history():
    s = AgentSystem()
    s._initialized = True
    # create 8 already-retired agents; evolve keeps only last 5
    for i in range(8):
        ag = s._create(f"old{i}", "quotes", "task")
        ag.status = "retired"
    s.evolve()
    retired = [a for a in s.agents if a.status == "retired"]
    assert len(retired) <= 5


def test_evolve_no_matching_blueprint_source():
    s = AgentSystem()
    s._initialized = True
    a = s._create("weird", "unknown_source", "task")
    a.status = "failed"
    res = s.evolve()
    assert "weird" in res["retired"]
    assert res["created"] == []  # no blueprint matched -> no replacement


def test_evolve_multiple_failed_snapshot_iteration_safe():
    # _create() appends replacements to self.agents; evolve() must iterate over a
    # snapshot so the mutation during iteration cannot corrupt the loop.
    s = AgentSystem()
    s._initialized = True
    for name, src in [("arxiv_scout", "arxiv"),
                      ("wiki_explorer", "wikipedia"),
                      ("news_scanner", "news")]:
        a = s._create(name, src, "task")
        a.status = "failed"
    res = s.evolve()  # must not raise / loop over freshly-appended replacements
    assert len(res["retired"]) == 3
    assert len(res["created"]) == 3   # exactly one replacement per failed agent
    # replacements are active and were not themselves retired this cycle
    assert res["active_agents"] == 3


# ── Reporting ─────────────────────────────────────────────────────────────

def test_get_recent_knowledge():
    s = AgentSystem()
    for i in range(15):
        s.collected_knowledge.append({"i": i})
    out = s.get_recent_knowledge(limit=5)
    assert len(out) == 5
    assert out[-1]["i"] == 14


def test_status_report():
    s = AgentSystem()
    s._initialized = True
    a = s._create("arxiv_scout", "arxiv", "task")
    a.status = "active"
    s.generation_log.append({"event": "created", "agent_id": a.agent_id, "name": a.name})
    st = s.status()
    assert st["total_agents"] == 1
    assert st["by_status"]["active"] == 1
    assert st["recent_events"]
    assert len(st["agents"]) == 1
