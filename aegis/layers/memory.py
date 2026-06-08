"""Layer 1: Memory — multi-level memory system (M-001..M-007).

Forgetting is deterministic: memories survive based on retention score
computed from age, importance, and access count — no random threshold.
"""
import json
import math
import time
import logging
from pathlib import Path
from aegis.config import MEMORY_DIR, MAX_WORKING_MEMORY, MEMORY_DECAY_RATE

logger = logging.getLogger("aegis.memory")


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
            except Exception:
                logger.warning("Failed to load memory state from %s — starting empty",
                               self._persistence_path, exc_info=True)

    def save(self):
        data = {
            "episodic": self.episodic[-1000:],
            "semantic": self.semantic,
            "procedural": self.procedural[-100:],
            "meta": self.meta,
        }
        self._persistence_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def add_working(self, item: dict):
        item["timestamp"] = time.time()
        self.working.append(item)
        if len(self.working) > MAX_WORKING_MEMORY:
            self.working = self.working[-MAX_WORKING_MEMORY:]

    def add_episodic(self, event: str, emotional_valence: float = 0.0, importance: float = 0.5):
        entry = {
            "event": event,
            "timestamp": time.time(),
            "valence": emotional_valence,
            "importance": importance,
            "access_count": 0,
            "last_access": time.time(),
        }
        self.episodic.append(entry)

    def add_semantic(self, concept: str, relations: dict):
        self.semantic[concept] = {
            "relations": relations,
            "created": time.time(),
            "updated": time.time(),
            "confidence": relations.get("confidence", 0.8),
        }

    def add_procedural(self, name: str, procedure: dict):
        self.procedural.append({
            "name": name,
            "procedure": procedure,
            "created": time.time(),
            "version": 1,
            "success_rate": 1.0,
        })

    def update_meta(self, domain: str, knows: bool, confidence: float):
        self.meta[domain] = {
            "knows": knows,
            "confidence": confidence,
            "updated": time.time(),
        }

    def recall_episodic(self, query: str = "", limit: int = 10) -> list[dict]:
        results = []
        for ep in reversed(self.episodic):
            if not query or query.lower() in ep["event"].lower():
                ep["access_count"] += 1
                ep["last_access"] = time.time()
                results.append(ep)
                if len(results) >= limit:
                    break
        return results

    def apply_forgetting(self):
        now = time.time()
        surviving = []
        forgotten = 0
        for ep in self.episodic:
            age_hours = (now - ep["timestamp"]) / 3600
            retention = math.exp(-MEMORY_DECAY_RATE * age_hours)
            retention *= (1 + ep["importance"])
            retention *= (1 + 0.1 * ep["access_count"])
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
            "recent_episodic": [{"event": e["event"], "time": e["timestamp"], "importance": e["importance"]}
                                for e in self.episodic[-5:]],
            "knowledge_graph_sample": list(self.semantic.keys())[:20],
        }
