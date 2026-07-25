"""Unit tests for aegis.layers.external_learning.

All HTTP is mocked via a fake httpx.AsyncClient — no real network access.
"""
import asyncio

import aegis.layers.external_learning as el
from aegis.layers.external_learning import ExternalLearning


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
    monkeypatch.setattr(el.httpx, "AsyncClient", make_client(handler))


# Reuse a single event loop across tests. Creating a fresh loop per asyncio.run
# call exhausts Windows socket buffers (WinError 10055) under many async tests.
_LOOP = asyncio.new_event_loop()


def run(coro):
    return _LOOP.run_until_complete(coro)


# ── learn_from_source dispatch / bookkeeping ──────────────────────────────

def test_learn_no_http(monkeypatch):
    monkeypatch.setattr(el, "HAS_HTTP", False)
    e = ExternalLearning()
    res = run(e.learn_from_source("wikipedia"))
    assert res["success"] is False
    assert "not installed" in res["error"]


def test_learn_unknown_source():
    e = ExternalLearning()
    res = run(e.learn_from_source("mystery"))
    assert res["success"] is False
    assert "not available" in res["error"]


def test_learn_disabled_source():
    e = ExternalLearning()
    e.SOURCES = {**e.SOURCES, "wikipedia": {**e.SOURCES["wikipedia"], "enabled": False}}
    res = run(e.learn_from_source("wikipedia"))
    assert res["success"] is False


def test_learn_wikipedia_success_updates_stats(monkeypatch):
    good = {"title": "Emergence",
            "extract": "Emergence is a phenomenon. It has many facets across science. "
                       "Complex behavior arises from simple rules over time here."}
    install(monkeypatch, lambda url: FakeResp(200, json_data=good))
    e = ExternalLearning()
    res = run(e.learn_from_source("wikipedia", "emergence"))
    assert res["success"] is True
    assert e.source_stats["wikipedia"] == 1
    assert e.total_concepts > 0
    assert e.learning_sessions == 1
    assert e.SOURCES["wikipedia"]["last_fetch"] > 0
    assert len(e.learned_items) == 1


def test_learn_failure_increments_failed(monkeypatch):
    # wikipedia returns non-200 -> _fetch returns success False
    install(monkeypatch, lambda url: FakeResp(404))
    e = ExternalLearning()
    res = run(e.learn_from_source("wikipedia", "topic"))
    assert res["success"] is False
    assert e.failed_fetches == 1
    assert e.learning_sessions == 1


def test_learn_dispatch_all_sources(monkeypatch):
    e = ExternalLearning()

    async def stub(*a, **k):
        return {"success": True, "concepts": ["c"]}

    for src in ("wikipedia", "arxiv", "news", "quotes"):
        # patch the matching private fetcher so dispatch branch is taken
        name = {"wikipedia": "_fetch_wikipedia", "arxiv": "_fetch_arxiv",
                "news": "_fetch_news", "quotes": "_fetch_quotes"}[src]
        monkeypatch.setattr(e, name, stub)
        res = run(e.learn_from_source(src, "t"))
        assert res["success"] is True


def test_learn_exception_path(monkeypatch):
    e = ExternalLearning()

    async def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(e, "_fetch_wikipedia", boom)
    res = run(e.learn_from_source("wikipedia", "t"))
    assert res["success"] is False
    assert "kaboom" in res["error"]
    assert e.failed_fetches == 1


# ── _fetch_wikipedia ──────────────────────────────────────────────────────

def test_fetch_wikipedia_default_topic_english(monkeypatch):
    captured = {}

    def handler(url):
        captured["url"] = url
        return FakeResp(200, json_data={"title": "T", "extract": "x" * 60})

    install(monkeypatch, handler)
    e = ExternalLearning()
    res = run(e._fetch_wikipedia(""))  # empty -> random default topic
    assert res["success"] is True
    assert "en.wikipedia.org" in captured["url"]


def test_fetch_wikipedia_cyrillic_uses_ru(monkeypatch):
    captured = {}

    def handler(url):
        captured["url"] = url
        return FakeResp(200, json_data={"title": "Сознание", "extract": "y" * 40})

    install(monkeypatch, handler)
    e = ExternalLearning()
    run(e._fetch_wikipedia("сознание"))
    assert "ru.wikipedia.org" in captured["url"]


def test_fetch_wikipedia_non_200_returns_failure(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(500))
    e = ExternalLearning()
    res = run(e._fetch_wikipedia("topic"))
    assert res["success"] is False
    assert res["concepts"] == []


def test_fetch_wikipedia_exception_returns_failure(monkeypatch):
    def boom(url):
        raise RuntimeError("net")
    install(monkeypatch, boom)
    e = ExternalLearning()
    res = run(e._fetch_wikipedia("topic"))
    assert res["success"] is False


# ── _fetch_arxiv ──────────────────────────────────────────────────────────

def test_fetch_arxiv_parses_titles(monkeypatch):
    text = ("<feed><title>ArXiv Query: all</title>"
            "<entry><title>Paper One</title></entry>"
            "<entry><title>Paper Two</title></entry></feed>")
    install(monkeypatch, lambda url: FakeResp(200, text=text))
    e = ExternalLearning()
    res = run(e._fetch_arxiv("machine learning"))
    assert res["success"] is True
    assert "Paper One" in res["concepts"]
    assert "ArXiv Query:" not in res["concepts"]


def test_fetch_arxiv_default_topic(monkeypatch):
    captured = {}

    def handler(url):
        captured["url"] = url
        return FakeResp(200, text="<feed><entry><title>P</title></entry></feed>")

    install(monkeypatch, handler)
    e = ExternalLearning()
    res = run(e._fetch_arxiv(""))
    assert res["success"] is True
    assert "export.arxiv.org" in captured["url"]


def test_fetch_arxiv_non_200(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(503))
    e = ExternalLearning()
    res = run(e._fetch_arxiv("x"))
    assert res["success"] is False


def test_fetch_arxiv_exception(monkeypatch):
    def boom(url):
        raise RuntimeError("down")
    install(monkeypatch, boom)
    e = ExternalLearning()
    res = run(e._fetch_arxiv("x"))
    assert res["success"] is False


# ── _fetch_news ───────────────────────────────────────────────────────────

def test_fetch_news_parses_items(monkeypatch):
    text = ("<rss><channel>"
            "<item><title>News A</title></item>"
            "<item><nope>no title here</nope></item>"  # triggers ValueError branch
            "<item><title>News B</title></item>"
            "</channel></rss>")
    install(monkeypatch, lambda url: FakeResp(200, text=text))
    e = ExternalLearning()
    res = run(e._fetch_news("ai"))
    assert res["success"] is True
    assert "News A" in res["concepts"]


def test_fetch_news_default_topic_and_fallback(monkeypatch):
    # non-200 -> falls through to fallback sample (still success True)
    install(monkeypatch, lambda url: FakeResp(500))
    e = ExternalLearning()
    res = run(e._fetch_news(""))
    assert res["success"] is True
    assert res["concepts"]  # fallback topics


def test_fetch_news_exception_fallback(monkeypatch):
    def boom(url):
        raise RuntimeError("x")
    install(monkeypatch, boom)
    e = ExternalLearning()
    res = run(e._fetch_news("topic"))
    assert res["success"] is True  # fallback


def test_fetch_news_no_titles_fallback(monkeypatch):
    # 200 but no parseable items -> empty titles -> fallback path
    install(monkeypatch, lambda url: FakeResp(200, text="<rss></rss>"))
    e = ExternalLearning()
    res = run(e._fetch_news("topic"))
    assert res["success"] is True


# ── _fetch_quotes ─────────────────────────────────────────────────────────

def test_fetch_quotes_success(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(200, json_data=[{"q": "Be wise", "a": "Sage"}]))
    e = ExternalLearning()
    res = run(e._fetch_quotes())
    assert res["success"] is True
    assert "Be wise" in res["summary"]


def test_fetch_quotes_bad_status_fallback(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(500))
    e = ExternalLearning()
    res = run(e._fetch_quotes())
    assert res["success"] is True
    assert res["concepts"]


def test_fetch_quotes_empty_content_fallback(monkeypatch):
    install(monkeypatch, lambda url: FakeResp(200, json_data=[{"q": "", "a": "x"}]))
    e = ExternalLearning()
    res = run(e._fetch_quotes())
    assert res["success"] is True  # falls to hardcoded fallback


def test_fetch_quotes_exception_fallback(monkeypatch):
    def boom(url):
        raise RuntimeError("x")
    install(monkeypatch, boom)
    e = ExternalLearning()
    res = run(e._fetch_quotes())
    assert res["success"] is True


# ── helpers / status ──────────────────────────────────────────────────────

def test_extract_concepts_filters_short():
    text = "short. This is a sufficiently long sentence to be kept as a concept. tiny."
    concepts = ExternalLearning._extract_concepts(text)
    assert any("sufficiently long" in c for c in concepts)
    assert all(len(c) > 20 for c in concepts)


def test_status_report(monkeypatch):
    good = {"title": "T", "extract": "x" * 80}
    install(monkeypatch, lambda url: FakeResp(200, json_data=good))
    e = ExternalLearning()
    run(e.learn_from_source("wikipedia", "t"))
    st = e.status()
    assert st["learning_sessions"] == 1
    assert "wikipedia" in st["sources"]
    assert st["recent_learning"]
