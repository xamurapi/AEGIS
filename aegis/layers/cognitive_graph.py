"""System 2: Cognitive Graph — typed graph of knowledge and experience (CG-001..CG-005).

Nodes: concept | event | skill | goal | outcome.
Edges: relates_to | causes | requires | learned_from | led_to.

Built incrementally from the memory system and other layers; supports path
finding, relevance queries (token overlap) and centrality, so reasoning can
use CONNECTED knowledge instead of flat recency lists. Deterministic and
dependency-free.
"""
import json
import time
import logging
from collections import deque
from pathlib import Path

from aegis.config import COGNITIVE_GRAPH_DIR

logger = logging.getLogger("aegis.cognitive_graph")

NODE_TYPES = ("concept", "event", "skill", "goal", "outcome")
EDGE_RELATIONS = ("relates_to", "causes", "requires", "learned_from", "led_to")
MAX_NODES = 5000


def _tokenize(text: str) -> set[str]:
    out, word = set(), []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        elif word:
            tok = "".join(word)
            if len(tok) > 2:
                out.add(tok)
            word = []
    if word:
        tok = "".join(word)
        if len(tok) > 2:
            out.add(tok)
    return out


class CognitiveGraph:
    def __init__(self, store_path: Path | None = None):
        # nodes[id] = {type, meta, created, updated}
        self.nodes: dict[str, dict] = {}
        # edges[src][dst] = {relation, weight, updated}
        self.edges: dict[str, dict[str, dict]] = {}
        # Derived index: number of INCOMING edges per node. Maintained
        # incrementally because computing it by scanning every adjacency list
        # made _degree() O(E), and _degree() is a sort key in _prune(),
        # central_nodes() and related() — i.e. O(N·E) per status() call on a
        # graph capped at 5000 nodes (audit R3-7). Not persisted: rebuilt from
        # self.edges on load, so the on-disk format is unchanged.
        self._in_degree: dict[str, int] = {}
        self._ingested_episodic = 0  # high-water mark into memory.episodic
        self._store_path = store_path or (COGNITIVE_GRAPH_DIR / "graph.json")
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self):
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", {})
                self._ingested_episodic = data.get("ingested_episodic", 0)
            except Exception:
                logger.warning("Failed to load cognitive graph from %s — starting empty",
                               self._store_path, exc_info=True)
        self._rebuild_in_degree()

    def _rebuild_in_degree(self):
        self._in_degree = {}
        for dsts in self.edges.values():
            for dst in dsts:
                self._in_degree[dst] = self._in_degree.get(dst, 0) + 1

    def save(self):
        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "ingested_episodic": self._ingested_episodic,
        }
        try:
            payload = json.dumps(data, ensure_ascii=False)
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._store_path)
        except Exception:
            logger.warning("Failed to save cognitive graph to %s", self._store_path, exc_info=True)

    # ── construction ─────────────────────────────────────────────────

    def add_node(self, node_id: str, node_type: str, meta: dict | None = None) -> str:
        node_id = node_id[:120]
        if node_type not in NODE_TYPES:
            node_type = "concept"
        now = time.time()
        if node_id in self.nodes:
            self.nodes[node_id]["updated"] = now
            if meta:
                self.nodes[node_id]["meta"].update(meta)
        else:
            self.nodes[node_id] = {
                "type": node_type, "meta": meta or {},
                "created": now, "updated": now,
            }
        self._prune()
        return node_id

    def add_edge(self, src: str, dst: str, relation: str = "relates_to", weight: float = 0.5):
        if src not in self.nodes or dst not in self.nodes or src == dst:
            return
        if relation not in EDGE_RELATIONS:
            relation = "relates_to"
        existing = self.edges.setdefault(src, {}).get(dst)
        if existing:
            # Reinforce: repeated observations strengthen the connection.
            existing["weight"] = min(1.0, existing["weight"] + 0.1)
            existing["updated"] = time.time()
        else:
            self.edges[src][dst] = {
                "relation": relation,
                "weight": max(0.0, min(1.0, weight)),
                "updated": time.time(),
            }
            self._in_degree[dst] = self._in_degree.get(dst, 0) + 1

    def _degree(self, node_id: str) -> int:
        return len(self.edges.get(node_id, {})) + self._in_degree.get(node_id, 0)

    def _prune(self):
        if len(self.nodes) <= MAX_NODES:
            return
        # Drop lowest-degree, oldest nodes and their edges.
        candidates = sorted(self.nodes.items(),
                            key=lambda kv: (self._degree(kv[0]), kv[1]["updated"]))
        for node_id, _ in candidates[:len(self.nodes) - MAX_NODES]:
            self.nodes.pop(node_id, None)
            # Outgoing edges disappear — every target loses one incoming edge.
            for dst in self.edges.pop(node_id, {}):
                if dst in self._in_degree:
                    self._in_degree[dst] -= 1
                    if self._in_degree[dst] <= 0:
                        del self._in_degree[dst]
            # Incoming edges disappear with the node itself.
            for dsts in self.edges.values():
                dsts.pop(node_id, None)
            self._in_degree.pop(node_id, None)

    def ingest_memory(self, memory, concept_limit: int = 20, event_window: int = 40):
        """Pull recent semantic concepts and recent episodic events into the
        graph, linking events to the concepts whose tokens they mention.

        Events are taken from the last `event_window` of episodic memory rather
        than tracked by an absolute high-water index. An index breaks silently:
        `apply_forgetting` and the on-disk `episodic[-1000:]` truncation remove
        items from anywhere in the list, so a saved index can exceed the (now
        shorter) list and skip every new event forever. add_node/add_edge are
        idempotent (dedup by id, edges reinforce), so re-scanning a bounded
        recent window is safe and cheap, and never drops a fresh event.
        """
        # Concepts (most recent slice).
        for concept in list(memory.semantic.keys())[-concept_limit:]:
            rel = memory.semantic[concept].get("relations", {})
            self.add_node(concept, "concept", {
                "source": rel.get("type", "semantic"),
            })

        concept_tokens = {c: _tokenize(c) for c in self.nodes
                          if self.nodes[c]["type"] == "concept"}
        for ep in memory.episodic[-event_window:]:
            event_id = f"ev:{ep.get('event', '')[:100]}"
            self.add_node(event_id, "event", {"importance": ep.get("importance", 0.5)})
            ev_tokens = _tokenize(ep.get("event", ""))
            for concept, ctoks in concept_tokens.items():
                if ctoks and ctoks & ev_tokens:
                    self.add_edge(event_id, concept, "relates_to",
                                  0.3 + 0.4 * ep.get("importance", 0.5))
        self._ingested_episodic = len(memory.episodic)

    # ── queries ──────────────────────────────────────────────────────

    def neighbors(self, node_id: str) -> list[dict]:
        result = [{"node": dst, "relation": e["relation"], "weight": e["weight"], "direction": "out"}
                  for dst, e in self.edges.get(node_id, {}).items()]
        for src, dsts in self.edges.items():
            if node_id in dsts:
                e = dsts[node_id]
                result.append({"node": src, "relation": e["relation"],
                               "weight": e["weight"], "direction": "in"})
        return result

    def find_path(self, start: str, goal: str, max_depth: int = 6) -> list[str] | None:
        """Shortest undirected path between two nodes (BFS)."""
        if start not in self.nodes or goal not in self.nodes:
            return None
        if start == goal:
            return [start]
        # Undirected adjacency.
        adj: dict[str, set[str]] = {}
        for src, dsts in self.edges.items():
            for dst in dsts:
                adj.setdefault(src, set()).add(dst)
                adj.setdefault(dst, set()).add(src)
        visited = {start}
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            if len(path) > max_depth:
                return None
            for nxt in adj.get(path[-1], ()):
                if nxt == goal:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def related(self, query: str, k: int = 5) -> list[dict]:
        """Nodes most relevant to a query (Jaccard token overlap), boosted by degree."""
        q = _tokenize(query)
        if not q:
            return []
        scored = []
        for node_id, node in self.nodes.items():
            doc = _tokenize(node_id)
            if not doc:
                continue
            overlap = len(q & doc)
            if overlap == 0:
                continue
            score = overlap / len(q | doc)
            score *= 1 + 0.05 * min(10, self._degree(node_id))
            scored.append({"node": node_id, "type": node["type"], "score": round(score, 3)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def central_nodes(self, k: int = 5) -> list[dict]:
        ranked = sorted(self.nodes, key=self._degree, reverse=True)
        return [{"node": n, "type": self.nodes[n]["type"], "degree": self._degree(n)}
                for n in ranked[:k]]

    # ── status ───────────────────────────────────────────────────────

    def status(self) -> dict:
        by_type: dict[str, int] = {}
        for node in self.nodes.values():
            by_type[node["type"]] = by_type.get(node["type"], 0) + 1
        return {
            "nodes": len(self.nodes),
            "edges": sum(len(v) for v in self.edges.values()),
            "by_type": by_type,
            "central": self.central_nodes(5),
        }
