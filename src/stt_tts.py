"""
stt_tts.py
==========
Speech-To-Text (Whisper) and Text-To-Speech (gTTS) utilities.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import whisper
from gtts import gTTS


# ─── STT ─────────────────────────────────────────────────────────────────────
class WhisperSTT:
    """
    Whisper-based Speech-To-Text transcription.

    Args:
        model_size: "tiny" | "base" | "small" | "medium" | "large"
        language: ISO-639-1 code or None (auto-detect)
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = "en",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        self.language = language
        self.fp16 = device == "cuda"

    def transcribe_file(self, audio_path: str) -> dict:
        """Transcribe an audio file. Returns Whisper result dict."""
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            fp16=self.fp16,
        )
        return result

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Transcribe raw audio bytes. Returns plain text."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            result = self.transcribe_file(tmp_path)
            return result["text"].strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_with_segments(self, audio_path: str) -> tuple[str, list[dict]]:
        """Returns (full_text, segments) with word-level timestamps."""
        result = self.transcribe_file(audio_path)
        return result["text"].strip(), result.get("segments", [])


# ─── TTS ─────────────────────────────────────────────────────────────────────
class GTTSSpeaker:
    """
    Text-To-Speech using gTTS (Google TTS).

    Args:
        lang: BCP-47 language code (default "en")
        slow: speak slowly (default False)
    """

    def __init__(self, lang: str = "en", slow: bool = False):
        self.lang = lang
        self.slow = slow

    def speak_to_file(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesise speech and save to a file.
        Returns the path of the saved audio file.
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        tts = gTTS(text=text, lang=self.lang, slow=self.slow)
        tts.save(output_path)
        return output_path

    def speak_to_bytes(self, text: str) -> bytes:
        """Synthesise speech and return raw MP3 bytes."""
        path = self.speak_to_file(text)
        try:
            with open(path, "rb") as f:
                return f.read()
        finally:
            Path(path).unlink(missing_ok=True)


# ─── Emotion-aware TTS feedback ──────────────────────────────────────────────
EMOTION_VOICE_TEMPLATES = {
    "joy":      "Great to hear! {snippet} — you seem to be feeling joyful today.",
    "sadness":  "I hear you. {snippet} — it sounds like you're feeling sad.",
    "anger":    "Noted. {snippet} — there seems to be some frustration there.",
    "fear":     "Understood. {snippet} — it sounds like something is worrying you.",
    "surprise": "Interesting! {snippet} — you seem surprised by something.",
    "disgust":  "I see. {snippet} — that sounds unpleasant.",
    "neutral":  "{snippet}.",
}


def build_tts_feedback(
    transcription: str,
    dominant_emotion: str,
    dominant_score: float,
    speaker: Optional[GTTSSpeaker] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate TTS audio feedback based on detected emotion.
    Returns the audio file path.
    """
    snippet = transcription[:70] + ("..." if len(transcription) > 70 else "")
    template = EMOTION_VOICE_TEMPLATES.get(dominant_emotion, "{snippet}.")
    score_pct = round(dominant_score * 100)

    text = (
        template.format(snippet=snippet)
        + f" Confidence: {score_pct} percent."
    )

    if speaker is None:
        speaker = GTTSSpeaker()

    return speaker.speak_to_file(text, output_path=save_path)


# ─── Smoke-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing TTS...")
    speaker = GTTSSpeaker()
    path = speaker.speak_to_file("Hello! Your dominant emotion today is joy. Stay positive!")
    print(f"  Saved TTS to: {path}")

    print("\nTesting emotion-aware feedback TTS...")
    path2 = build_tts_feedback(
        transcription="I just got the results and I passed my thesis defence!",
        dominant_emotion="joy",
        dominant_score=0.94,
    )
    print(f"  Saved feedback TTS to: {path2}")
