"""Tests for the keyword-based Emotion NLP classifier."""
from aegis.layers.emotion_nlp import EmotionNLP


def test_initial_state():
    e = EmotionNLP()
    assert e.analysis_count == 0
    assert all(v == 0 for v in e.emotion_stats.values())


def test_analyze_joy_english():
    e = EmotionNLP()
    res = e.analyze("I am so happy and this is wonderful")
    assert res["dominant"] == "joy"
    assert res["valence"] > 0
    assert res["word_count"] == 8


def test_analyze_negative_valence():
    e = EmotionNLP()
    res = e.analyze("I feel sad and lonely and full of grief")
    assert res["dominant"] == "sadness"
    assert res["valence"] < 0


def test_analyze_russian_keywords():
    e = EmotionNLP()
    res = e.analyze("Я чувствую страх и тревога")
    assert res["dominant"] == "fear"


def test_analyze_neutral_when_no_match():
    e = EmotionNLP()
    res = e.analyze("zzz qqq xyz")
    assert res["dominant"] == "neutral"
    assert res["scores"]["neutral"] == 1.0


def test_scores_are_normalized():
    e = EmotionNLP()
    res = e.analyze("happy joy trust reliable")
    assert abs(sum(res["scores"].values()) - 1.0) < 1e-6


def test_stats_and_count_updated():
    e = EmotionNLP()
    e.analyze("happy wonderful")
    e.analyze("sad lonely")
    assert e.analysis_count == 2
    assert e.emotion_stats["joy"] == 1
    assert e.emotion_stats["sadness"] == 1


def test_curiosity_positive_valence():
    e = EmotionNLP()
    res = e.analyze("this is fascinating and interesting to explore")
    assert res["dominant"] == "curiosity"
    assert res["valence"] > 0


def test_status_only_nonzero():
    e = EmotionNLP()
    e.analyze("happy")
    st = e.status()
    assert st["analysis_count"] == 1
    assert st["emotion_distribution"] == {"joy": 1}
    assert "sadness" not in st["emotion_distribution"]
