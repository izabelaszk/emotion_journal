"""
FastAPI REST API — Emotion-Aware Voice Journal
Wraps the core pipeline for production deployment / external integrations.

Endpoints:
  POST /transcribe   — audio file → transcription
  POST /classify     — text → emotion scores
  POST /analyse      — audio file → full pipeline result
  GET  /journal      — retrieve all stored entries
  GET  /health       — health check
"""

import datetime
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.emotion_classifier import EmotionClassifier
from src.journal_store import add_entry, clear_entries, get_entries, total_entries
from src.stt_tts import WhisperSTT

# ─── Config ──────────────────────────────────────────────────────────────────
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
WHISPER_MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion-Aware Voice Journal API",
    description="STT + Emotion Detection + TTS — portfolio project (master's thesis extension)",
    version="1.0.0",
)

# NOTE: open CORS is intentional for this portfolio demo so the API can be
# called from anywhere. Restrict `allow_origins` to known hosts in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ────────────────────────────────────────────────────────────
# Reuse the same core components as the Gradio app so there is a single,
# shared implementation of transcription and classification.
print("Loading models...")
stt = WhisperSTT(model_size=WHISPER_MODEL_SIZE)
emotion_clf = EmotionClassifier(backend="transformer", model_id=EMOTION_MODEL_ID)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class EmotionScore(BaseModel):
    label: str
    score: float

class AnalysisResult(BaseModel):
    transcription: str
    emotions: list[EmotionScore]
    dominant_emotion: str
    dominant_score: float
    feedback_text: str
    timestamp: str

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _transcribe(audio_bytes: bytes, suffix: str = ".wav") -> str:
    return stt.transcribe_bytes(audio_bytes, suffix=suffix)


def _classify(text: str) -> list[EmotionScore]:
    # EmotionClassifier already returns a score-sorted list of {label, score}.
    return [EmotionScore(**r) for r in emotion_clf.predict(text)]


def _feedback(text: str, emotions: list[EmotionScore]) -> str:
    top = emotions[0]
    score_pct = round(top.score * 100)
    snippet = text[:80] + ("..." if len(text) > 80 else "")
    return (
        f"I heard: \"{snippet}\". "
        f"Dominant emotion detected: {top.label} ({score_pct}% confidence)."
    )

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "entries": total_entries()}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Upload an audio file, get back a transcription."""
    content = await audio.read()
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    try:
        text = _transcribe(content, suffix)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"transcription": text}


@app.post("/classify", response_model=list[EmotionScore])
def classify(req: TextRequest):
    """Classify emotions in a text string."""
    return _classify(req.text)


@app.post("/analyse", response_model=AnalysisResult)
async def analyse(
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
):
    """Full pipeline: audio or text → emotions → feedback."""
    if audio is not None:
        content = await audio.read()
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        transcription = _transcribe(content, suffix)
    elif text:
        transcription = text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide audio file or text form field.")

    emotions = _classify(transcription)
    feedback = _feedback(transcription, emotions)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    add_entry({
        "timestamp": ts,
        "transcription": transcription,
        "emotions": [e.model_dump() for e in emotions],
        "dominant": emotions[0].label,
    })

    return AnalysisResult(
        transcription=transcription,
        emotions=emotions,
        dominant_emotion=emotions[0].label,
        dominant_score=emotions[0].score,
        feedback_text=feedback,
        timestamp=ts,
    )


@app.get("/journal")
def get_journal(limit: int = 50):
    """Return the last `limit` journal entries."""
    entries = get_entries(limit)
    return {"entries": entries, "total": total_entries()}


@app.delete("/journal")
def delete_journal():
    """Clear all journal entries."""
    count = clear_entries()
    return {"deleted": count}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
