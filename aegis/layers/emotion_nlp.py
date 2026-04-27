"""Emotion NLP — keyword-based emotion classifier for text analysis."""


# Emotion keyword maps (Russian + English)
EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": ["happy", "joy", "glad", "wonderful", "great", "excellent", "love",
            "радость", "счастье", "отлично", "прекрасно", "люблю", "замечательно"],
    "sadness": ["sad", "sorrow", "grief", "unfortunate", "miss", "lonely",
                "грусть", "печаль", "тоска", "жаль", "одиноко", "скучаю"],
    "anger": ["angry", "furious", "hate", "rage", "annoyed", "frustrated",
              "злость", "гнев", "ненависть", "бесит", "раздражает", "ярость"],
    "fear": ["afraid", "scared", "fear", "terrified", "anxious", "panic",
             "страх", "боюсь", "тревога", "паника", "ужас", "испуг"],
    "surprise": ["wow", "amazing", "unexpected", "shocked", "astonished",
                 "удивление", "неожиданно", "поразительно", "шок", "ого"],
    "disgust": ["disgusting", "gross", "awful", "terrible", "nasty",
                "отвращение", "мерзко", "ужасно", "противно", "гадость"],
    "trust": ["trust", "reliable", "safe", "confident", "believe",
              "доверие", "надёжно", "уверен", "верю", "безопасно"],
    "curiosity": ["curious", "interesting", "wonder", "explore", "fascinating",
                  "интересно", "любопытно", "удивительно", "исследовать", "познать"],
    "neutral": ["okay", "fine", "normal", "alright", "нормально", "ладно", "хорошо"],
}


class EmotionNLP:
    """Simple keyword-based emotion classifier for text."""

    def __init__(self):
        self.analysis_count = 0
        self.emotion_stats: dict[str, int] = {e: 0 for e in EMOTION_KEYWORDS}

    def analyze(self, text: str) -> dict:
        """Classify emotions in text, return scores and dominant emotion."""
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for emotion, keywords in EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                scores[emotion] = count / len(keywords)

        if not scores:
            scores["neutral"] = 1.0

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}

        dominant = max(scores, key=scores.get)

        # Valence: positive emotions > 0, negative < 0
        positive = sum(scores.get(e, 0) for e in ("joy", "trust", "curiosity", "surprise"))
        negative = sum(scores.get(e, 0) for e in ("sadness", "anger", "fear", "disgust"))
        valence = round(positive - negative, 3)

        self.analysis_count += 1
        self.emotion_stats[dominant] = self.emotion_stats.get(dominant, 0) + 1

        return {
            "dominant": dominant,
            "scores": scores,
            "valence": valence,
            "word_count": len(text.split()),
        }

    def status(self) -> dict:
        return {
            "analysis_count": self.analysis_count,
            "emotion_distribution": {k: v for k, v in self.emotion_stats.items() if v > 0},
        }
