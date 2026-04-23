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

import io
import json
import os
import tempfile
import datetime
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import whisper
from transformers import pipeline as hf_pipeline
from gtts import gTTS
from src.journal_store import add_entry, get_entries, clear_entries, total_entries

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ────────────────────────────────────────────────────────────
print("Loading models...")
stt_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
emotion_clf = hf_pipeline(
    "text-classification",
    model=EMOTION_MODEL_ID,
    top_k=None,
    device=0 if DEVICE == "cuda" else -1,
)


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
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        result = stt_model.transcribe(tmp_path, language="en", fp16=(DEVICE == "cuda"))
        return result["text"].strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _classify(text: str) -> list[EmotionScore]:
    raw = emotion_clf(text)[0]
    return sorted(
        [EmotionScore(label=r["label"].lower(), score=round(r["score"], 4)) for r in raw],
        key=lambda x: x.score, reverse=True
    )


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
        raise HTTPException(status_code=422, detail=str(e))
    return {"transcription": text}


@app.post("/classify", response_model=list[EmotionScore])
def classify(req: TextRequest):
    """Classify emotions in a text string."""
    return _classify(req.text)


@app.post("/analyse", response_model=AnalysisResult)
async def analyse(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
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
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    add_entry({
        "timestamp": ts,
        "transcription": transcription,
        "emotions": [e.dict() for e in emotions],
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
