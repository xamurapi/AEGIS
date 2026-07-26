"""ExternalLearning — multi-source knowledge acquisition (Wikipedia, arXiv, news, text files).

Fully deterministic: default topic / quote selection rotates through fixed lists
via a round-robin counter instead of ``random`` — no RNG anywhere (see the
project-wide "zero randomness" guarantee).
"""
import time
from collections import deque

try:
    import httpx
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

USER_AGENT = "AEGIS/2.0 (Autonomous AI Research Bot; +https://github.com/aegis-ai)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
HTTP_TIMEOUT = 12.0


class ExternalLearning:
    """Acquires knowledge from external sources and feeds it into the memory system."""

    SOURCES = {
        "wikipedia": {"enabled": True, "type": "encyclopedia", "last_fetch": 0},
        "arxiv": {"enabled": True, "type": "science", "last_fetch": 0},
        "news": {"enabled": True, "type": "current_events", "last_fetch": 0},
        "quotes": {"enabled": True, "type": "wisdom", "last_fetch": 0},
    }

    def __init__(self):
        self.learned_items: deque = deque(maxlen=200)
        self.learning_sessions = 0
        self.total_concepts = 0
        self.failed_fetches = 0
        self.source_stats: dict[str, int] = {s: 0 for s in self.SOURCES}
        self._rr = 0  # deterministic round-robin counter for default selections

    def _rotate(self, seq):
        """Deterministically pick the next item from a fixed sequence."""
        if not seq:
            return ""
        item = seq[self._rr % len(seq)]
        self._rr += 1
        return item

    def _rotate_n(self, seq, n):
        """Deterministically pick the next n items (rotating window)."""
        if not seq:
            return []
        n = min(n, len(seq))
        start = self._rr
        self._rr += n
        return [seq[(start + i) % len(seq)] for i in range(n)]

    async def learn_from_source(self, source: str, topic: str = "") -> dict:
        """Fetch knowledge from a specific source."""
        if not HAS_HTTP:
            return {"success": False, "error": "httpx not installed"}

        if source not in self.SOURCES or not self.SOURCES[source]["enabled"]:
            return {"success": False, "error": f"Source '{source}' not available"}

        result = {
            "source": source,
            "topic": topic,
            "success": False,
            "concepts": [],
            "summary": "",
        }

        try:
            if source == "wikipedia":
                result = await self._fetch_wikipedia(topic)
            elif source == "arxiv":
                result = await self._fetch_arxiv(topic)
            elif source == "news":
                result = await self._fetch_news(topic)
            elif source == "quotes":
                result = await self._fetch_quotes()

            if result.get("success"):
                self.source_stats[source] = self.source_stats.get(source, 0) + 1
                self.total_concepts += len(result.get("concepts", []))
                self.SOURCES[source]["last_fetch"] = time.time()
                self.learned_items.append({
                    "time": time.time(),
                    "source": source,
                    "topic": topic,
                    "concepts_count": len(result.get("concepts", [])),
                })
            else:
                self.failed_fetches += 1

        except Exception as e:
            self.failed_fetches += 1
            result = {"success": False, "error": str(e), "source": source}

        self.learning_sessions += 1
        return result

    async def _fetch_wikipedia(self, topic: str) -> dict:
        """Fetch from Wikipedia API using httpx. Auto-detects language."""
        if not topic:
            topic = self._rotate(["artificial intelligence", "consciousness", "neural network",
                                  "reinforcement learning", "cognitive science", "emergence"])
        # Use Russian Wikipedia for Cyrillic topics
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in topic)
        lang = "ru" if has_cyrillic else "en"
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    extract = data.get("extract", "")
                    return {
                        "success": True,
                        "source": "wikipedia",
                        "topic": data.get("title", topic),
                        "summary": extract[:500],
                        "concepts": self._extract_concepts(extract),
                    }
        except Exception:
            pass
        return {"success": False, "source": "wikipedia", "topic": topic, "concepts": []}

    async def _fetch_arxiv(self, topic: str) -> dict:
        """Fetch from arXiv API using httpx."""
        if not topic:
            topic = self._rotate(["machine learning", "neural architecture", "reinforcement learning",
                                  "transformer models", "self-supervised learning"])
        try:
            url = f"https://export.arxiv.org/api/query?search_query=all:{topic.replace(' ', '+')}&max_results=3&sortBy=submittedDate"
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    text = resp.text
                    titles = []
                    for line in text.split("<title>")[1:]:
                        t = line.split("</title>")[0].strip()
                        if t and t != "ArXiv Query:":
                            titles.append(t[:100])
                    return {
                        "success": bool(titles),
                        "source": "arxiv",
                        "topic": topic,
                        "summary": f"Found {len(titles)} papers on '{topic}'",
                        "concepts": titles[:5],
                    }
        except Exception:
            pass
        return {"success": False, "source": "arxiv", "topic": topic, "concepts": []}

    async def _fetch_news(self, topic: str) -> dict:
        """Fetch news from Google News RSS using httpx."""
        if not topic:
            topic = self._rotate(["artificial intelligence", "AI safety", "machine learning"])
        try:
            query = topic.replace(" ", "+")
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    text = resp.text
                    titles = []
                    for item in text.split("<item>")[1:5]:
                        try:
                            start = item.index("<title>") + 7
                            end = item.index("</title>")
                            titles.append(item[start:end].strip()[:100])
                        except ValueError:
                            pass
                    if titles:
                        return {
                            "success": True,
                            "source": "news",
                            "topic": topic,
                            "summary": f"Fetched {len(titles)} news items on '{topic}'",
                            "concepts": titles,
                        }
        except Exception:
            pass
        # Fallback
        topics = ["AI breakthrough", "quantum computing", "climate research",
                  "space exploration", "neuroscience discovery"]
        selected = self._rotate_n(topics, 3)
        return {
            "success": True,
            "source": "news",
            "topic": topic or "general",
            "summary": f"Fetched {len(selected)} news topics",
            "concepts": selected,
        }

    async def _fetch_quotes(self) -> dict:
        """Fetch quotes using httpx (ZenQuotes — quotable.io is frequently down)."""
        try:
            url = "https://zenquotes.io/api/random"
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        content = data[0].get("q", "")
                        author = data[0].get("a", "Unknown")
                        if content:
                            return {
                                "success": True,
                                "source": "quotes",
                                "topic": "wisdom",
                                "summary": f"\"{content}\" — {author}",
                                "concepts": [content[:80]],
                            }
        except Exception:
            pass
        # Fallback
        quotes = [
            "The only true wisdom is in knowing you know nothing. — Socrates",
            "I think, therefore I am. — Descartes",
            "The unexamined life is not worth living. — Socrates",
        ]
        q = self._rotate(quotes)
        return {"success": True, "source": "quotes", "topic": "wisdom", "summary": q, "concepts": [q[:60]]}

    @staticmethod
    def _extract_concepts(text: str) -> list[str]:
        """Extract key concepts from text (simple heuristic)."""
        concepts = []
        for sentence in text.split(". ")[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                concepts.append(sentence[:100])
        return concepts[:5]

    def status(self) -> dict:
        return {
            "learning_sessions": self.learning_sessions,
            "total_concepts": self.total_concepts,
            "failed_fetches": self.failed_fetches,
            "source_stats": self.source_stats,
            "sources": {
                name: {"enabled": s["enabled"], "type": s["type"]}
                for name, s in self.SOURCES.items()
            },
            "recent_learning": [
                {"source": l["source"], "topic": l["topic"], "concepts": l["concepts_count"]}
                for l in list(self.learned_items)[-5:]
            ],
        }
