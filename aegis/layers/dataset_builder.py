"""Dataset Builder — builds fine-tuning datasets from AEGIS memory and agent data.

Deterministic: samples are ordered by a stable content hash (a reproducible
"shuffle") instead of ``random.shuffle`` — same inputs always yield the same
dataset, and no RNG is used (zero-randomness guarantee).
"""
import json
import logging
import hashlib
from pathlib import Path
from aegis.config import WEIGHT_DATASETS_DIR, TRAIN_MAX_SAMPLES, TRAIN_VAL_SPLIT
from aegis.clock import CLOCK

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self):
        self.last_build_time = 0.0
        self.builds_total = 0
        self.last_dataset_size = 0
        self.last_dataset_path = ""

    def build_from_memory(self, memory, agent_system=None, feedback_loop=None) -> dict:
        """Build a training dataset from AEGIS memory systems.

        Creates instruction/response pairs from:
        - Semantic knowledge (concept -> definition/summary)
        - Episodic memory (events -> reflections)
        - Agent-collected data (articles, papers -> summaries)
        - Real outcomes with their inferred cause, when a feedback loop is given
        """
        samples = []

        # 1. Semantic memory -> knowledge Q&A pairs
        for concept, data in memory.semantic.items():
            relations = data.get("relations", {})
            summary = relations.get("summary", relations.get("definition", ""))
            source_type = relations.get("type", "unknown")
            if not summary or len(summary) < 20:
                continue

            if source_type in ("external_learning", "agent_arxiv", "agent_wikipedia"):
                samples.append({
                    "instruction": f"What do you know about {concept}?",
                    "input": "",
                    "output": summary,
                    "source": source_type,
                })
            elif source_type == "learned_concept":
                definition = relations.get("definition", summary)
                if definition:
                    samples.append({
                        "instruction": f"Define the concept: {concept}",
                        "input": "",
                        "output": definition,
                        "source": "self_reflection",
                    })
            elif source_type == "curiosity_exploration":
                question = relations.get("question", "")
                connection = relations.get("connection", "")
                if question and connection:
                    samples.append({
                        "instruction": question,
                        "input": f"Related to: {concept}",
                        "output": connection,
                        "source": "curiosity",
                    })

        # 2. Episodic memory -> reflection pairs (high importance only)
        reflections = [e for e in memory.episodic
                       if e.get("importance", 0) > 0.7 and "Reflection:" in e.get("event", "")]
        for ep in reflections:
            event_text = ep["event"].replace("Reflection: ", "")
            if len(event_text) > 30:
                samples.append({
                    "instruction": "Reflect on your recent experience and share what you learned.",
                    "input": "",
                    "output": event_text,
                    "source": "episodic_reflection",
                })

        # 3. LLM insights from episodic memory
        insights = [e for e in memory.episodic
                    if e.get("importance", 0) > 0.6 and "LLM Insight:" in e.get("event", "")]
        for ep in insights:
            insight_text = ep["event"].replace("LLM Insight: ", "")
            if len(insight_text) > 20:
                samples.append({
                    "instruction": "Analyze your current state and provide an insight.",
                    "input": "",
                    "output": insight_text,
                    "source": "llm_insight",
                })

        # 4. Agent-collected knowledge
        if agent_system:
            for item in list(agent_system.collected_knowledge)[-200:]:
                data = item.get("data", {})
                title = data.get("title", "")
                summary = data.get("summary", "")
                source = item.get("source", "")
                if title and summary and len(summary) > 30:
                    if source == "arxiv":
                        samples.append({
                            "instruction": f"Summarize the research paper: {title}",
                            "input": "",
                            "output": summary,
                            "source": "agent_arxiv",
                        })
                    elif source == "wikipedia":
                        samples.append({
                            "instruction": f"Explain: {title}",
                            "input": "",
                            "output": summary,
                            "source": "agent_wikipedia",
                        })
                    elif source == "github":
                        samples.append({
                            "instruction": f"Describe the project: {title}",
                            "input": "",
                            "output": summary,
                            "source": "agent_github",
                        })
                    elif source == "news":
                        samples.append({
                            "instruction": f"What happened: {title}?",
                            "input": "",
                            "output": summary,
                            "source": "agent_news",
                        })

        # 5. Procedural memory -> how-to pairs
        for proc in memory.procedural:
            name = proc.get("name", "")
            procedure = proc.get("procedure", {})
            steps = procedure.get("steps", "")
            if name and steps:
                samples.append({
                    "instruction": f"How to: {name}?",
                    "input": "",
                    "output": str(steps),
                    "source": "procedural",
                })

        # 6. Real outcomes -> cause-carrying pairs (System 5)
        # The feedback loop records situation -> decision -> result -> cause for
        # every tick, and that log used to end at the dashboard: export_examples()
        # had no caller, so the training data was built from memory alone and the
        # causes were dropped. These are the only rows that teach WHY an outcome
        # happened rather than restating what the system already believes.
        if feedback_loop is not None:
            try:
                for row in feedback_loop.export_examples():
                    samples.append({
                        "instruction": "Given the situation and the decision taken, "
                                       "state the outcome and the cause behind it.",
                        "input": row["prompt"],
                        "output": row["completion"],
                        "source": "experience",
                    })
            except Exception:
                logger.warning("Failed to add experiences to the dataset", exc_info=True)

        # Deduplicate by output
        seen = set()
        unique_samples = []
        for s in samples:
            key = s["output"][:100]
            if key not in seen:
                seen.add(key)
                unique_samples.append(s)
        samples = unique_samples

        # Deterministic "shuffle": order by a stable content hash so the mix is
        # decorrelated from insertion order without using an RNG (reproducible).
        samples.sort(key=lambda s: hashlib.md5(
            json.dumps(s, sort_keys=True, default=str).encode("utf-8")).hexdigest())
        samples = samples[:TRAIN_MAX_SAMPLES]

        if not samples:
            return {"success": False, "error": "No samples generated", "size": 0}

        # Split train/val
        val_size = max(1, int(len(samples) * TRAIN_VAL_SPLIT))
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]

        # Save to disk
        timestamp = int(CLOCK.now())
        dataset_dir = WEIGHT_DATASETS_DIR / f"dataset_{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        train_path = dataset_dir / "train.jsonl"
        val_path = dataset_dir / "val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Source distribution
        source_counts = {}
        for s in samples:
            src = s.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1

        self.last_build_time = CLOCK.now()
        self.builds_total += 1
        self.last_dataset_size = len(samples)
        self.last_dataset_path = str(dataset_dir)

        return {
            "success": True,
            "dataset_dir": str(dataset_dir),
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "total_size": len(samples),
            "source_distribution": source_counts,
            "timestamp": timestamp,
        }

    def get_latest_dataset(self) -> Path | None:
        """Find the most recent dataset directory."""
        dirs = sorted(WEIGHT_DATASETS_DIR.glob("dataset_*"), reverse=True)
        for d in dirs:
            if (d / "train.jsonl").exists():
                return d
        return None

    def cleanup_old_datasets(self, keep: int = 3):
        """Remove old datasets, keeping the N most recent."""
        dirs = sorted(WEIGHT_DATASETS_DIR.glob("dataset_*"), reverse=True)
        removed = 0
        for d in dirs[keep:]:
            try:
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
                removed += 1
            except Exception:
                pass
        return removed

    def status(self) -> dict:
        latest = self.get_latest_dataset()
        datasets_on_disk = len(list(WEIGHT_DATASETS_DIR.glob("dataset_*")))
        return {
            "builds_total": self.builds_total,
            "last_build_time": self.last_build_time,
            "last_dataset_size": self.last_dataset_size,
            "last_dataset_path": self.last_dataset_path,
            "latest_dataset": str(latest) if latest else None,
            "datasets_on_disk": datasets_on_disk,
        }
