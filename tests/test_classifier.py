import pytest

from src.emotion_classifier import EKMAN_LABELS, EmotionClassifier

# These load real models / lexicon data — slow and may need network on first run.
pytestmark = pytest.mark.integration


def test_transformer_returns_all_labels():
    clf = EmotionClassifier(backend="transformer")
    results = clf.predict("I am so happy today!")
    assert isinstance(results, list)
    returned_labels = {r["label"] for r in results}
    assert returned_labels == set(EKMAN_LABELS)


def test_transformer_scores_sum_to_one():
    clf = EmotionClassifier(backend="transformer")
    results = clf.predict("Testing emotion scores.")
    total = sum(r["score"] for r in results)
    assert abs(total - 1.0) < 0.02


def test_transformer_dominant_joy():
    clf = EmotionClassifier(backend="transformer")
    label, score = clf.dominant("I am absolutely thrilled and overjoyed!")
    assert label == "joy"
    assert score > 0.5


def test_transformer_sorted_descending():
    clf = EmotionClassifier(backend="transformer")
    results = clf.predict("I am so happy today!")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_nrc_predict_returns_all_labels():
    clf = EmotionClassifier(backend="nrc")
    results = clf.predict("I am furious and disgusted by this!")
    returned_labels = {r["label"] for r in results}
    assert returned_labels == set(EKMAN_LABELS)


def test_nrc_scores_sum_to_one():
    clf = EmotionClassifier(backend="nrc")
    results = clf.predict("I feel a mix of joy and sadness.")
    total = sum(r["score"] for r in results)
    assert abs(total - 1.0) < 0.02


def test_nrc_neutral_on_empty_text():
    clf = EmotionClassifier(backend="nrc")
    label, _ = clf.dominant("the the the")
    assert label == "neutral"
    # NOTE: the offline `ValueError` case for an unknown backend lives in test_unit.py
