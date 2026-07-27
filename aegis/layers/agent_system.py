"""AgentSystem — autonomous spider-bot creation, execution, monitoring and evolution.

Deterministic: agent ids are a monotonic counter, topic selection rotates
through each blueprint's fixed list, and next-run staggering is a fixed spread —
no ``random`` anywhere (project-wide "zero randomness" guarantee).
"""
from aegis.clock import CLOCK
import asyncio
import itertools
from collections import deque
from dataclasses import dataclass, field

try:
    import httpx
    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

USER_AGENT = "AEGIS/2.0 (Autonomous AI Research Bot; +https://github.com/aegis-ai)"
DEFAULT_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
HTTP_TIMEOUT = 12.0


@dataclass
class SpiderAgent:
    """An autonomous data-collecting agent."""
    agent_id: str
    name: str
    source_type: str
    task_description: str
    topic: str = ""
    status: str = "created"
    # Injectable clock (spec §3.6) — agent age drives retirement,
    # so it has to be movable in a test rather than only by waiting.
    created_at: float = field(default_factory=CLOCK.now)
    last_run: float = 0.0
    next_run: float = 0.0
    run_interval: float = 120.0
    runs: int = 0
    successes: int = 0
    failures: int = 0
    data_collected: int = 0
    last_data: list = field(default_factory=list)
    last_error: str = ""

    def success_rate(self) -> float:
        return self.successes / max(self.runs, 1)

    def is_due(self) -> bool:
        return CLOCK.now() >= self.next_run and self.status == "active"

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "source_type": self.source_type,
            "topic": self.topic,
            "status": self.status,
            "runs": self.runs,
            "success_rate": round(self.success_rate(), 3),
            "data_collected": self.data_collected,
            "last_error": self.last_error[:80] if self.last_error else "",
            "last_data_preview": [str(d)[:100] for d in self.last_data[:3]],
        }


AGENT_BLUEPRINTS = [
    {
        "name": "arxiv_scout",
        "source_type": "arxiv",
        "task": "Search arXiv for recent AI/ML papers",
        "topics": ["artificial intelligence", "reinforcement learning", "neural networks",
                   "transformer models", "self-supervised learning", "cognitive architecture",
                   "multi-agent systems", "consciousness AI"],
        "interval": 180,
    },
    {
        "name": "wiki_explorer",
        "source_type": "wikipedia",
        "task": "Fetch Wikipedia summaries on knowledge topics",
        "topics": ["emergence", "consciousness", "neural plasticity", "self-organization",
                   "autopoiesis", "cybernetics", "cognitive science", "artificial general intelligence",
                   "swarm intelligence", "epistemology", "philosophy of mind", "qualia"],
        "interval": 150,
    },
    {
        "name": "quote_gatherer",
        "source_type": "quotes",
        "task": "Collect philosophical and scientific quotes",
        "topics": [],
        "interval": 200,
    },
    {
        "name": "github_watcher",
        "source_type": "github",
        "task": "Search GitHub for trending AI repositories",
        "topics": ["autonomous agent", "self-modifying AI", "cognitive architecture",
                   "neural network framework", "AI consciousness", "LLM agent"],
        "interval": 300,
    },
    {
        "name": "news_scanner",
        "source_type": "news",
        "task": "Scan for AI and technology news",
        "topics": ["artificial intelligence", "AI safety", "AGI research",
                   "neural network", "machine learning"],
        "interval": 240,
    },
]


class AgentSystem:
    """Manages autonomous data-collecting spider agents with real execution."""

    def __init__(self, max_agents: int = 20):
        self.agents: list[SpiderAgent] = []
        self.max_agents = max_agents
        self.generation_log: deque = deque(maxlen=50)
        self.collected_knowledge: deque = deque(maxlen=500)
        self.total_generated = 0
        self.total_retired = 0
        self.total_data_items = 0
        self.evolution_cycles = 0
        self._initialized = False
        self._rr = 0                        # round-robin topic selector
        self._id_seq = itertools.count(1)   # monotonic, collision-free agent ids
        self._stagger = 0                   # deterministic next-run spread

    def _rotate(self, seq):
        """Deterministically pick the next item from a fixed sequence."""
        if not seq:
            return ""
        item = seq[self._rr % len(seq)]
        self._rr += 1
        return item

    def auto_initialize(self):
        """Create default agents on first call."""
        if self._initialized:
            return
        self._initialized = True
        for bp in AGENT_BLUEPRINTS:
            topic = self._rotate(bp["topics"]) if bp["topics"] else ""
            agent = self._create(bp["name"], bp["source_type"], bp["task"], topic, bp["interval"])
            agent.status = "active"
            # Stagger start times deterministically across a 5..30s window.
            agent.next_run = CLOCK.now() + 5 + (self._stagger % 26)
            self._stagger += 5

    def _create(self, name: str, source_type: str, task: str,
                topic: str = "", interval: float = 120) -> SpiderAgent:
        agent_id = f"agent_{int(CLOCK.now())}_{next(self._id_seq):04d}"
        agent = SpiderAgent(
            agent_id=agent_id, name=name, source_type=source_type,
            task_description=task, topic=topic, run_interval=interval,
        )
        self.agents.append(agent)
        self.total_generated += 1
        self.generation_log.append({
            "time": CLOCK.now(), "event": "created",
            "agent_id": agent_id, "name": name, "source_type": source_type,
        })
        return agent

    def create_agent(self, name: str, source_type: str, task: str,
                     topic: str = "") -> SpiderAgent:
        """Public API to create and activate a new agent."""
        agent = self._create(name, source_type, task, topic)
        agent.status = "active"
        agent.next_run = CLOCK.now() + 5
        return agent

    # ── Execution ─────────────────────────────────────────────

    async def run_due_agents(self) -> list[dict]:
        """Run all agents that are due. Called from substrate tick."""
        if not HAS_HTTP:
            return []

        self.auto_initialize()
        results = []

        for agent in self.agents:
            if not agent.is_due():
                continue

            try:
                data = await self._execute_agent(agent)
                agent.runs += 1
                agent.last_run = CLOCK.now()
                agent.next_run = CLOCK.now() + agent.run_interval

                if data:
                    agent.successes += 1
                    agent.data_collected += len(data)
                    agent.last_data = data[:5]
                    agent.last_error = ""
                    self.total_data_items += len(data)
                    for item in data:
                        self.collected_knowledge.append({
                            "time": CLOCK.now(),
                            "agent": agent.name,
                            "source": agent.source_type,
                            "data": item,
                        })
                    results.append({
                        "agent": agent.name, "source": agent.source_type,
                        "items": len(data), "success": True,
                    })
                    self.generation_log.append({
                        "time": CLOCK.now(), "event": "fetched",
                        "agent_id": agent.agent_id, "name": agent.name,
                        "items": len(data),
                    })
                else:
                    agent.failures += 1
                    agent.last_error = "Empty result"

            except Exception as e:
                agent.runs += 1
                agent.failures += 1
                agent.last_run = CLOCK.now()
                agent.next_run = CLOCK.now() + agent.run_interval * 2
                agent.last_error = str(e)[:200]
                if agent.failures >= 10 and agent.success_rate() < 0.2:
                    agent.status = "failed"

        return results

    async def _execute_agent(self, agent: SpiderAgent) -> list[dict]:
        """Execute an agent's data collection task."""
        for bp in AGENT_BLUEPRINTS:
            if bp["name"] == agent.name and bp["topics"]:
                agent.topic = self._rotate(bp["topics"])
                break

        if agent.source_type == "arxiv":
            return await self._fetch_arxiv(agent.topic)
        elif agent.source_type == "wikipedia":
            return await self._fetch_wikipedia(agent.topic)
        elif agent.source_type == "quotes":
            return await self._fetch_quotes()
        elif agent.source_type == "github":
            return await self._fetch_github(agent.topic)
        elif agent.source_type == "news":
            return await self._fetch_news(agent.topic)
        return []

    # ── Fetchers (httpx) ──────────────────────────────────────

    async def _fetch_arxiv(self, topic: str) -> list[dict]:
        query = topic.replace(" ", "+")
        url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results=5&sortBy=submittedDate"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            text = resp.text
            results = []
            entries = text.split("<entry>")[1:]
            for entry in entries[:5]:
                title = self._xml_tag(entry, "title").strip().replace("\n", " ")
                summary = self._xml_tag(entry, "summary").strip()[:200]
                link = ""
                if 'href="' in entry:
                    link = entry.split('href="')[1].split('"')[0]
                if title:
                    results.append({
                        "type": "paper", "title": title,
                        "summary": summary, "url": link, "source": "arxiv",
                    })
            return results

    async def _fetch_wikipedia(self, topic: str) -> list[dict]:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            data = resp.json()
            extract = data.get("extract", "")
            if extract:
                return [{
                    "type": "wiki", "title": data.get("title", topic),
                    "summary": extract[:300],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "source": "wikipedia",
                }]
            return []

    async def _fetch_quotes(self) -> list[dict]:
        results = []
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
            for _ in range(3):
                try:
                    resp = await client.get("https://zenquotes.io/api/random")
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list) and data and data[0].get("q"):
                            results.append({
                                "type": "quote",
                                "title": data[0].get("a", "Unknown"),
                                "summary": data[0].get("q", ""),
                                "source": "zenquotes",
                            })
                except Exception:
                    pass
        if not results:
            fallback = [
                ("Socrates", "The only true wisdom is in knowing you know nothing."),
                ("Descartes", "I think, therefore I am."),
                ("Aristotle", "Knowing yourself is the beginning of all wisdom."),
                ("Lao Tzu", "A journey of a thousand miles begins with a single step."),
                ("Einstein", "Imagination is more important than knowledge."),
            ]
            a, q = self._rotate(fallback)
            results.append({"type": "quote", "title": a, "summary": q, "source": "fallback"})
        return results

    async def _fetch_github(self, topic: str) -> list[dict]:
        url = f"https://api.github.com/search/repositories?q={topic.replace(' ', '+')}&sort=stars&per_page=5"
        headers = {**DEFAULT_HEADERS, "Accept": "application/vnd.github.v3+json"}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for repo in data.get("items", [])[:5]:
                results.append({
                    "type": "repo", "title": repo.get("full_name", ""),
                    "summary": (repo.get("description", "") or "")[:200],
                    "url": repo.get("html_url", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "source": "github",
                })
            return results

    async def _fetch_news(self, topic: str) -> list[dict]:
        query = topic.replace(" ", "+")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=DEFAULT_HEADERS) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            text = resp.text
            results = []
            items = text.split("<item>")[1:]
            for item in items[:5]:
                title = self._xml_tag(item, "title")
                link = self._xml_tag(item, "link")
                pub = self._xml_tag(item, "pubDate")
                if title:
                    results.append({
                        "type": "news", "title": title[:150],
                        "summary": pub, "url": link, "source": "google_news",
                    })
            return results

    @staticmethod
    def _xml_tag(text: str, tag: str) -> str:
        try:
            start = text.index(f"<{tag}>") + len(tag) + 2
            end = text.index(f"</{tag}>")
            return text[start:end].strip()
        except ValueError:
            return ""

    # ── Evolution ─────────────────────────────────────────────

    def evolve(self) -> dict:
        """Retire failed agents, auto-replace them."""
        self.evolution_cycles += 1
        retired = []
        created = []

        # Iterate over a snapshot: _create() appends replacements to
        # self.agents, which would mutate the list during iteration.
        for agent in list(self.agents):
            if agent.status == "failed" or (agent.runs > 10 and agent.success_rate() < 0.15):
                agent.status = "retired"
                self.total_retired += 1
                retired.append(agent.name)

                for bp in AGENT_BLUEPRINTS:
                    if bp["source_type"] == agent.source_type:
                        topic = self._rotate(bp["topics"]) if bp["topics"] else ""
                        new = self._create(
                            f"{bp['name']}_v{self.evolution_cycles}",
                            bp["source_type"], bp["task"], topic, bp["interval"],
                        )
                        new.status = "active"
                        new.next_run = CLOCK.now() + 10
                        created.append(new.name)
                        break

        active = [a for a in self.agents if a.status != "retired"]
        retired_kept = [a for a in self.agents if a.status == "retired"][-5:]
        self.agents = active + retired_kept

        return {
            "cycle": self.evolution_cycles,
            "retired": retired,
            "created": created,
            "active_agents": sum(1 for a in self.agents if a.status == "active"),
        }

    def get_recent_knowledge(self, limit: int = 10) -> list[dict]:
        """Get recently collected knowledge for memory integration."""
        return list(self.collected_knowledge)[-limit:]

    def status(self) -> dict:
        by_status = {}
        for a in self.agents:
            by_status[a.status] = by_status.get(a.status, 0) + 1

        return {
            "total_agents": len(self.agents),
            "total_generated": self.total_generated,
            "total_retired": self.total_retired,
            "total_data_items": self.total_data_items,
            "evolution_cycles": self.evolution_cycles,
            "knowledge_buffer": len(self.collected_knowledge),
            "by_status": by_status,
            "agents": [a.to_dict() for a in self.agents if a.status != "retired"][:10],
            "recent_events": [
                {"event": e["event"], "agent_id": e.get("agent_id", ""),
                 "name": e.get("name", ""), "items": e.get("items", "")}
                for e in list(self.generation_log)[-8:]
            ],
        }
