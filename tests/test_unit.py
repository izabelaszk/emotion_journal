"""
Fast, offline unit tests — no model downloads, GPU, or network required.
Run with:  pytest -m "not integration"
"""
import pytest

from src.emotion_classifier import EKMAN_LABELS, EmotionClassifier
from src.stt_tts import EMOTION_VOICE_TEMPLATES, build_tts_feedback


# ─── EmotionClassifier (pure logic, no model load) ──────────────────────────────
def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        EmotionClassifier(backend="does-not-exist")


def test_ekman_labels_are_the_expected_seven():
    assert set(EKMAN_LABELS) == {
        "joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral",
    }


# ─── TTS feedback templates ─────────────────────────────────────────────────────
def test_every_emotion_has_a_feedback_template():
    for label in EKMAN_LABELS:
        assert label in EMOTION_VOICE_TEMPLATES


def test_build_tts_feedback_writes_audio(tmp_path, monkeypatch):
    # Stub the network-dependent synthesiser so the test stays offline.
    from src import stt_tts

    def fake_save(self, text, output_path=None):
        path = output_path or str(tmp_path / "out.mp3")
        with open(path, "w") as f:
            f.write(text)          # capture the spoken text for assertions
        return path

    monkeypatch.setattr(stt_tts.GTTSSpeaker, "speak_to_file", fake_save)

    out = build_tts_feedback(
        transcription="I passed my thesis defence!",
        dominant_emotion="joy",
        dominant_score=0.94,
        save_path=str(tmp_path / "out.mp3"),
    )
    with open(out) as f:
        spoken = f.read()
    assert "joyful" in spoken          # from the joy template
    assert "94 percent" in spoken      # score rendered as a percentage


# ─── Journal store CRUD (real SQLite, temp DB) ──────────────────────────────────
def test_journal_store_crud(tmp_path, monkeypatch):
    import src.journal_store as js

    monkeypatch.setattr(js, "DB_PATH", tmp_path / "journal.db")
    js._init()

    assert js.total_entries() == 0
    assert js.get_entries() == []

    entry = {
        "timestamp": "2026-01-01T10:00:00",
        "transcription": "hello world",
        "emotions": [{"label": "joy", "score": 0.9}],
        "dominant": "joy",
    }
    js.add_entry(entry)
    js.add_entry({**entry, "timestamp": "2026-01-01T11:00:00", "dominant": "neutral"})

    assert js.total_entries() == 2

    rows = js.get_entries()
    assert len(rows) == 2
    # emotions round-trip through JSON as a list of dicts
    assert rows[0]["emotions"] == [{"label": "joy", "score": 0.9}]

    deleted = js.clear_entries()
    assert deleted == 2
    assert js.total_entries() == 0
