"""
emotion_classifier.py
=====================
Unified emotion classification interface supporting multiple backends:
  1. Fine-tuned transformer (DistilBERT / RoBERTa) — primary
  2. Zero-shot LLM (OpenAI or Anthropic) — optional, matches thesis methodology
  3. NRC lexicon baseline — lightweight fallback

Usage:
    clf = EmotionClassifier(backend="transformer")
    scores = clf.predict("I am so happy today!")
    # [{"label": "joy", "score": 0.92}, ...]
"""

from __future__ import annotations
import os
import json
from typing import Literal, Optional
import torch
from transformers import pipeline as hf_pipeline

# ─── Supported emotion labels (Ekman-6 + neutral) ────────────────────────────
EKMAN_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

# ─── Transformer Backend ─────────────────────────────────────────────────────
class TransformerClassifier:
    """
    Fine-tuned transformer model for emotion classification.
    Defaults to j-hartmann/emotion-english-distilroberta-base.
    Swap `model_id` for your own fine-tuned checkpoint.
    """

    def __init__(self, model_id: str = "j-hartmann/emotion-english-distilroberta-base"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = hf_pipeline(
            "text-classification",
            model=model_id,
            top_k=None,
            device=device,
        )
        self.model_id = model_id

    def predict(self, text: str) -> list[dict]:
        raw = self.pipe(text)[0]
        results = [{"label": r["label"].lower(), "score": round(r["score"], 4)} for r in raw]
        return sorted(results, key=lambda x: x["score"], reverse=True)


# ─── Zero-Shot LLM Backend ───────────────────────────────────────────────────
class LLMClassifier:
    """
    Zero-shot emotion classification using an LLM API.
    Supports OpenAI (gpt-4o-mini) and Anthropic (claude-haiku).
    Mirrors the methodology from the master's thesis.
    """

    SYSTEM_PROMPT = """You are an expert emotion classifier.
Given a text, output ONLY a JSON array of objects with keys "label" and "score",
where labels are from: joy, sadness, anger, fear, surprise, disgust, neutral.
Scores must sum to 1.0. No explanations, no markdown — raw JSON only.
Example: [{"label": "joy", "score": 0.85}, {"label": "neutral", "score": 0.15}]"""

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: Optional[str] = None,
    ):
        self.provider = provider
        if provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = model or "gpt-4o-mini"
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = model or "claude-haiku-4-5-20251001"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def predict(self, text: str) -> list[dict]:
        user_msg = f"Classify the emotions in this text:\n\n{text}"
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            raw_json = response.choices[0].message.content.strip()
        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_json = response.content[0].text.strip()

        results = json.loads(raw_json)
        return sorted(results, key=lambda x: x["score"], reverse=True)


# ─── NRC Lexicon Backend ─────────────────────────────────────────────────────
class NRCClassifier:
    """
    Lexicon-based classifier using the NRC Emotion Lexicon (via nrclex).
    Lightweight — no GPU, no API key. Good as a fast baseline.
    """

    _EKMAN_KEYS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

    def predict(self, text: str) -> list[dict]:
        from nrclex import NRCLex
        freq = NRCLex(text).affect_frequencies
        raw = {k: freq.get(k, 0.0) for k in self._EKMAN_KEYS}
        total_ekman = sum(raw.values())
        raw["neutral"] = max(0.0, 1.0 - total_ekman)
        grand = sum(raw.values())
        if grand > 0:
            scores = {k: round(v / grand, 4) for k, v in raw.items()}
        else:
            n = len(raw)
            scores = {k: round(1 / n, 4) for k in raw}
        return sorted([{"label": k, "score": v} for k, v in scores.items()],
                      key=lambda x: x["score"], reverse=True)


# ─── Ollama Backend ──────────────────────────────────────────────────────────
class OllamaClassifier:
    """
    Zero-shot emotion classification using a locally-running Ollama model.
    Requires `ollama serve` to be running and the chosen model pulled.
    """

    SYSTEM_PROMPT = LLMClassifier.SYSTEM_PROMPT

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def predict(self, text: str) -> list[dict]:
        import ollama
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify the emotions in this text:\n\n{text}"},
            ],
        )
        raw = response["message"]["content"].strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        results = json.loads(raw.strip())
        return sorted(
            [{"label": r["label"].lower(), "score": round(r["score"], 4)} for r in results],
            key=lambda x: x["score"], reverse=True,
        )


# ─── Unified Interface ────────────────────────────────────────────────────────
class EmotionClassifier:
    """
    Unified interface for all emotion classification backends.

    Args:
        backend: "transformer" | "nrc" | "openai" | "anthropic"
        model_id: override default model (transformer backend only)
    """

    def __init__(
        self,
        backend: Literal["transformer", "nrc", "openai", "anthropic", "ollama"] = "transformer",
        model_id: Optional[str] = None,
    ):
        self.backend = backend
        if backend == "transformer":
            self._clf = TransformerClassifier(model_id or "j-hartmann/emotion-english-distilroberta-base")
        elif backend == "nrc":
            self._clf = NRCClassifier()
        elif backend in ("openai", "anthropic"):
            self._clf = LLMClassifier(provider=backend, model=model_id)
        elif backend == "ollama":
            self._clf = OllamaClassifier(model=model_id or "llama3.2")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def predict(self, text: str) -> list[dict]:
        """
        Returns a list of dicts sorted by score descending:
            [{"label": "joy", "score": 0.91}, ...]
        """
        return self._clf.predict(text)

    def dominant(self, text: str) -> tuple[str, float]:
        """Returns (dominant_label, score)."""
        results = self.predict(text)
        return results[0]["label"], results[0]["score"]


# ─── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        "I'm so thrilled to finally finish my thesis!",
        "I hate when everything goes wrong at once.",
        "The weather today is rather ordinary.",
    ]

    print("=== Transformer Backend ===")
    clf = EmotionClassifier(backend="transformer")
    for t in test_texts:
        label, score = clf.dominant(t)
        print(f"  [{label:10s} {score:.2f}]  {t[:60]}")
