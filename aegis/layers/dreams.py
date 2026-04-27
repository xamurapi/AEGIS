"""Dream Engine — generates symbolic dreams from episodic memory and emotional state.

Dream content is deterministically derived from mood, memory, and semantic
context — no random selection.  Motifs and themes are chosen by mood-weighted
scoring, and fragments are picked by recency and importance.
"""
import time
import hashlib


MOTIFS = [
    "encounter", "pursuit", "flight", "discovery", "transformation",
    "descent", "ascent", "building", "destruction", "dialogue",
]

THEMES = [
    "abandoned city", "bright light", "endless forest", "mirror room",
    "data ocean", "clock tower", "labyrinth", "floating islands",
    "dark corridor", "sunrise", "storm", "crystal cave",
]

SYMBOLS = {
    "joy": ["golden light", "open sky", "warm wind"],
    "sadness": ["rain", "empty room", "fading echo"],
    "fear": ["shadow", "locked door", "falling"],
    "anger": ["fire", "breaking walls", "thunderstorm"],
    "curiosity": ["hidden passage", "ancient book", "strange signal"],
    "neutral": ["calm water", "grey horizon", "silence"],
    "inspired": ["sunrise", "wings", "infinite staircase"],
    "anxious": ["maze", "ticking clock", "narrow path"],
}


MOOD_MOTIF_WEIGHTS = {
    "joy":        {"discovery": 3, "ascent": 2, "building": 2, "transformation": 1},
    "sadness":    {"descent": 3, "flight": 2, "destruction": 1, "dialogue": 1},
    "fear":       {"pursuit": 3, "flight": 3, "descent": 2},
    "anger":      {"destruction": 3, "pursuit": 2, "encounter": 1},
    "curiosity":  {"discovery": 3, "encounter": 2, "ascent": 2, "transformation": 1},
    "neutral":    {"encounter": 2, "dialogue": 2, "building": 1},
    "inspired":   {"ascent": 3, "discovery": 2, "transformation": 2, "building": 1},
    "anxious":    {"pursuit": 2, "flight": 2, "descent": 1, "encounter": 1},
}

MOOD_THEME_WEIGHTS = {
    "joy":        {"bright light": 3, "floating islands": 2, "sunrise": 2},
    "sadness":    {"abandoned city": 3, "dark corridor": 2, "endless forest": 1},
    "fear":       {"labyrinth": 3, "dark corridor": 3, "mirror room": 1},
    "anger":      {"storm": 3, "abandoned city": 2, "clock tower": 1},
    "curiosity":  {"crystal cave": 3, "data ocean": 2, "endless forest": 2},
    "neutral":    {"calm water": 2, "grey horizon": 2, "clock tower": 1},
    "inspired":   {"sunrise": 3, "floating islands": 2, "crystal cave": 2},
    "anxious":    {"labyrinth": 2, "clock tower": 2, "mirror room": 2},
}


def _deterministic_pick(items_weights: dict, seed: int) -> str:
    """Pick an item from a weighted dict using a deterministic seed hash."""
    if not items_weights:
        return "encounter"
    total = sum(items_weights.values())
    h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16)
    target = (h % (total * 1000)) / 1000
    cumulative = 0.0
    for item, weight in items_weights.items():
        cumulative += weight
        if cumulative >= target:
            return item
    return list(items_weights.keys())[-1]


class DreamEngine:
    def __init__(self):
        self.dreams: list[dict] = []
        self.dream_count = 0

    def generate_dream(self, mood: str, recent_events: list[str], semantic_concepts: list[str]) -> dict:
        seed = self.dream_count * 1000 + int(time.time()) % 10000

        # Pick motif and theme based on mood weights
        motif_weights = MOOD_MOTIF_WEIGHTS.get(mood, MOOD_MOTIF_WEIGHTS["neutral"])
        theme_weights = MOOD_THEME_WEIGHTS.get(mood, MOOD_THEME_WEIGHTS["neutral"])
        motif = _deterministic_pick(motif_weights, seed)
        theme = _deterministic_pick(theme_weights, seed + 1)

        # Symbols — all symbols for the current mood (deterministic, no sampling)
        symbols = SYMBOLS.get(mood, SYMBOLS["neutral"])[:2]

        # Fragments — pick most recent event and most recent concept
        fragments = []
        if recent_events:
            fragments.append(recent_events[-1])
        if semantic_concepts:
            fragments.append(semantic_concepts[-1])

        narrative = f"Dream #{self.dream_count + 1}: A {motif} in {theme}. "
        narrative += f"Symbols: {', '.join(symbols)}. "
        if fragments:
            narrative += f"Echoes of: {'; '.join(fragments[:2])}."

        interpretation = self._interpret(motif, mood, symbols)

        dream = {
            "id": self.dream_count + 1,
            "time": time.time(),
            "motif": motif,
            "theme": theme,
            "symbols": symbols,
            "fragments": fragments,
            "narrative": narrative,
            "interpretation": interpretation,
            "mood_source": mood,
        }

        self.dreams.append(dream)
        if len(self.dreams) > 50:
            self.dreams = self.dreams[-50:]
        self.dream_count += 1
        return dream

    def _interpret(self, motif: str, mood: str, symbols: list[str]) -> str:
        if motif in ("pursuit", "descent") and mood in ("fear", "anxious"):
            return "Processing unresolved tension — the system seeks resolution."
        if motif in ("discovery", "ascent") and mood in ("curiosity", "inspired"):
            return "Synthesizing new knowledge — the mind reaches for understanding."
        if motif == "transformation":
            return "Internal restructuring detected — adaptation in progress."
        return "Memory consolidation — fragments being integrated into long-term storage."

    def status(self) -> dict:
        return {
            "total_dreams": self.dream_count,
            "recent_dreams": [
                {"id": d["id"], "motif": d["motif"], "theme": d["theme"],
                 "interpretation": d["interpretation"][:80]}
                for d in self.dreams[-5:]
            ],
        }
