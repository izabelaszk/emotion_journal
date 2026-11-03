"""
evaluate.py
===========
Benchmark the emotion classifier backends on the GoEmotions test split,
mapped to Ekman-6 + neutral (the label set the app uses).

Reports accuracy, macro-F1 and per-request latency, prints a Markdown table
and writes the raw numbers to results/eval.json.

Usage (from the repo root):
    python -m scripts.evaluate                          # transformer + nrc
    python -m scripts.evaluate --backends transformer nrc ollama --n 500
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from src.emotion_classifier import EKMAN_LABELS

# Official GoEmotions → Ekman mapping (google-research/goemotions, ekman_mapping.json)
EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
GO_TO_EKMAN = {go: ekman for ekman, gos in EKMAN_MAPPING.items() for go in gos}

RESULTS_PATH = Path("results/eval.json")


# ─── Metrics ─────────────────────────────────────────────────────────────────
def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=True)]))


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Unweighted mean of per-label F1; a label with no support and no predictions scores 0."""
    f1s = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


# ─── Data ────────────────────────────────────────────────────────────────────
def load_goemotions_ekman(n: int, seed: int) -> list[tuple[str, str]]:
    """Single-label GoEmotions test examples as (text, ekman_label), randomly sampled."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    names = ds.features["labels"].feature.names
    examples = [
        (row["text"], GO_TO_EKMAN[names[row["labels"][0]]])
        for row in ds
        if len(row["labels"]) == 1
    ]
    random.Random(seed).shuffle(examples)
    return examples[:n]


# ─── Benchmark ───────────────────────────────────────────────────────────────
def evaluate_backend(backend: str, examples: list[tuple[str, str]]) -> dict:
    from src.emotion_classifier import EmotionClassifier

    clf = EmotionClassifier(backend=backend)
    clf.predict("warm-up")  # exclude model load / first-call overhead from latency

    y_true, y_pred, latencies = [], [], []
    for text, label in examples:
        start = time.perf_counter()
        try:
            pred = clf.predict(text)[0]["label"]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            pred = "invalid"  # e.g. an LLM returned malformed JSON; counts as wrong
        latencies.append((time.perf_counter() - start) * 1000)
        y_true.append(label)
        y_pred.append(pred)

    return {
        "backend": backend,
        "n": len(examples),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred, EKMAN_LABELS), 4),
        "invalid_outputs": y_pred.count("invalid"),
        "latency_ms_mean": round(float(np.mean(latencies)), 1),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
    }


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| Backend | Accuracy | Macro-F1 | Latency mean (ms) | Latency p95 (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
            f"| {r['latency_ms_mean']:.1f} | {r['latency_ms_p95']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backends", nargs="+", default=["transformer", "nrc"])
    parser.add_argument("--n", type=int, default=1000, help="number of test examples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_goemotions_ekman(args.n, args.seed)
    rows = []
    for backend in args.backends:
        try:
            rows.append(evaluate_backend(backend, examples))
        except Exception as exc:  # noqa: BLE001 — e.g. Ollama not running; skip, don't abort the run
            print(f"WARNING: skipping backend {backend!r}: {exc}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "dataset": "go_emotions (simplified, test, single-label, Ekman-mapped)",
        "seed": args.seed,
        "platform": f"{platform.system()} {platform.machine()}, {platform.processor() or 'unknown CPU'}",
        "results": rows,
    }, indent=2))
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
