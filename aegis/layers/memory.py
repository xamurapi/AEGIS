"""Layer 1: Memory — multi-level memory system (M-001..M-007).

Forgetting is deterministic: memories survive based on retention score
computed from age, importance, and access count — no random threshold.
"""
import json
import math
import logging
from aegis.config import MEMORY_DIR, MAX_WORKING_MEMORY, MEMORY_DECAY_RATE
from aegis.clock import CLOCK

logger = logging.getLogger("aegis.memory")

# Soft RAM caps so a long-running process cannot grow memory without bound.
# These are far above normal working sizes — they only trip on runaway growth.
MAX_EPISODIC_RAM = 5000       # episodic entries kept in RAM (oldest dropped)
MAX_SEMANTIC_CONCEPTS = 2000  # semantic concepts kept (least-recently-updated pruned)


class MemorySystem:
    def __init__(self):
        self.working: list[dict] = []
        self.episodic: list[dict] = []
        self.semantic: dict[str, dict] = {}  # knowledge graph (simplified)
        self.procedural: list[dict] = []
        self.meta: dict[str, dict] = {}  # what the system knows/doesn't know
        self.forgotten_total = 0
        self._persistence_path = MEMORY_DIR / "memory_state.json"
        self._load()

    def _load(self):
        if self._persistence_path.exists():
            try:
                data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
                self.episodic = data.get("episodic", [])
                self.semantic = data.get("semantic", {})
                self.procedural = data.get("procedural", [])
                self.meta = data.get("meta", {})
                self.forgotten_total = data.get("forgotten_total", 0)
            except Exception:
                logger.warning("Failed to load memory state from %s — starting empty",
                               self._persistence_path, exc_info=True)

    def save(self):
        data = {
            "episodic": self.episodic[-1000:],
            "semantic": self.semantic,
            "procedural": self.procedural[-100:],
            "meta": self.meta,
            "forgotten_total": self.forgotten_total,
        }
        # Atomic write: dump to a temp file in the same directory, then replace.
        # A crash mid-write must not truncate the existing state and wipe all
        # persisted memory on next load.
        try:
            payload = json.dumps(data, ensure_ascii=False, indent=1)
            tmp = self._persistence_path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._persistence_path)
        except Exception:
            logger.warning("Failed to save memory state to %s", self._persistence_path, exc_info=True)

    def add_working(self, item: dict):
        item["timestamp"] = CLOCK.now()
        self.working.append(item)
        if len(self.working) > MAX_WORKING_MEMORY:
            self.working = self.working[-MAX_WORKING_MEMORY:]

    def add_episodic(self, event: str, emotional_valence: float = 0.0, importance: float = 0.5):
        entry = {
            "event": event,
            "timestamp": CLOCK.now(),
            "valence": emotional_valence,
            "importance": importance,
            "access_count": 0,
            "last_access": CLOCK.now(),
        }
        self.episodic.append(entry)
        # Soft RAM cap: drop the oldest episodic entries on runaway growth.
        if len(self.episodic) > MAX_EPISODIC_RAM:
            self.episodic = self.episodic[-MAX_EPISODIC_RAM:]

    def add_semantic(self, concept: str, relations: dict):
        # Preserve the original creation time when re-learning a known concept —
        # an update must not masquerade as a brand-new concept.
        existing = self.semantic.get(concept)
        created = existing.get("created", CLOCK.now()) if isinstance(existing, dict) else CLOCK.now()
        self.semantic[concept] = {
            "relations": relations,
            "created": created,
            "updated": CLOCK.now(),
            "confidence": relations.get("confidence", 0.8),
        }
        # Soft RAM cap: prune the least-recently-updated concepts on overflow.
        if len(self.semantic) > MAX_SEMANTIC_CONCEPTS:
            self._prune_semantic()

    def _prune_semantic(self):
        """Keep the most-recently-updated MAX_SEMANTIC_CONCEPTS concepts,
        dropping the least-used (oldest-updated) ones."""
        ranked = sorted(self.semantic.items(),
                        key=lambda kv: kv[1].get("updated", 0) if isinstance(kv[1], dict) else 0,
                        reverse=True)
        self.semantic = dict(ranked[:MAX_SEMANTIC_CONCEPTS])

    def add_procedural(self, name: str, procedure: dict):
        self.procedural.append({
            "name": name,
            "procedure": procedure,
            "created": CLOCK.now(),
            "version": 1,
            "success_rate": 1.0,
        })

    def update_meta(self, domain: str, knows: bool, confidence: float):
        self.meta[domain] = {
            "knows": knows,
            "confidence": confidence,
            "updated": CLOCK.now(),
        }

    @staticmethod
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

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """RAG retrieval over semantic memory (point 3).

        Ranks concepts by token-overlap (Jaccard) between the query and each
        concept's key + stored summary/definition. Dependency-free; meant to be
        fed into the LLM context so decisions use relevant knowledge instead of
        just the most-recent N concepts.
        """
        q = self._tokenize(query)
        if not q:
            return []
        scored: list[tuple[float, str, str]] = []
        for concept, val in self.semantic.items():
            rel = val.get("relations", {}) if isinstance(val, dict) else {}
            summary = rel.get("summary") or rel.get("definition") or ""
            doc = self._tokenize(f"{concept} {summary}")
            if not doc:
                continue
            overlap = len(q & doc)
            if overlap == 0:
                continue
            score = overlap / len(q | doc)  # Jaccard
            scored.append((score, concept, summary))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"concept": c, "summary": s[:200], "score": round(sc, 3)}
                for sc, c, s in scored[:k]]

    def recall_episodic(self, query: str = "", limit: int = 10) -> list[dict]:
        results = []
        for ep in reversed(self.episodic):
            event = ep.get("event", "")
            if not query or query.lower() in event.lower():
                ep["access_count"] = ep.get("access_count", 0) + 1
                ep["last_access"] = CLOCK.now()
                results.append(ep)
                if len(results) >= limit:
                    break
        return results

    def apply_forgetting(self):
        now = CLOCK.now()
        surviving = []
        forgotten = 0
        for ep in self.episodic:
            age_hours = (now - ep.get("timestamp", now)) / 3600
            retention = math.exp(-MEMORY_DECAY_RATE * age_hours)
            retention *= (1 + ep.get("importance", 0.5))
            retention *= (1 + 0.1 * ep.get("access_count", 0))
            if retention > 0.1:
                surviving.append(ep)
            else:
                forgotten += 1
        self.episodic = surviving
        self.forgotten_total += forgotten
        return forgotten

    def status(self) -> dict:
        return {
            "working_memory_size": len(self.working),
            "working_memory_max": MAX_WORKING_MEMORY,
            "episodic_count": len(self.episodic),
            "semantic_concepts": len(self.semantic),
            "procedural_count": len(self.procedural),
            "meta_domains": len(self.meta),
            "total_memories": len(self.episodic) + len(self.semantic) + len(self.procedural),
            "forgotten_count": self.forgotten_total,
            "recent_episodic": [{"event": e.get("event", ""), "time": e.get("timestamp"),
                                 "importance": e.get("importance", 0.5)}
                                for e in self.episodic[-5:]],
            "knowledge_graph_sample": list(self.semantic.keys())[:20],
        }
