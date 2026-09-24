"""
Emotion-Aware Voice Journal — Gradio UI
Pipeline: STT (Whisper) → Emotion Classification (RoBERTa | Ollama) → TTS feedback
"""

import datetime

import gradio as gr

from src.emotion_classifier import EmotionClassifier
from src.journal_store import add_entry, clear_entries, get_entries
from src.stt_tts import GTTSSpeaker, WhisperSTT, build_tts_feedback

# ─── Config ───────────────────────────────────────────────────────────────────
EMOTION_MODEL_ID   = "j-hartmann/emotion-english-distilroberta-base"
WHISPER_MODEL_SIZE = "base"
OLLAMA_MODEL       = "llama3.2"

EMOTION_COLORS = {
    "joy": "#FFD166", "sadness": "#6B8CFF", "anger": "#FF6B6B",
    "fear": "#C77DFF", "surprise": "#06D6A0", "disgust": "#95D5B2", "neutral": "#ADB5BD",
}
EMOTION_EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}

SUMMARY_TRIGGER = 5

# ─── Model loading ─────────────────────────────────────────────────────────────
print("Loading Whisper STT model...")
stt     = WhisperSTT(model_size=WHISPER_MODEL_SIZE)
speaker = GTTSSpeaker()

print("Loading RoBERTa emotion classifier...")
transformer_clf = EmotionClassifier(backend="transformer", model_id=EMOTION_MODEL_ID)
ollama_clf      = EmotionClassifier(backend="ollama",       model_id=OLLAMA_MODEL)

# ─── Pipeline functions ────────────────────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    return stt.transcribe_file(audio_path)["text"].strip()


# ─── UI helpers ────────────────────────────────────────────────────────────────
def _emotion_bars_html(emotions: list[dict]) -> str:
    html = '<div style="padding:4px 0;">'
    for e in emotions:
        label = e["label"].lower()
        pct   = round(e["score"] * 100, 1)
        color = EMOTION_COLORS.get(label, "#888")
        emoji = EMOTION_EMOJI.get(label, "")
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;font-size:0.84rem;">
          <span style="width:105px;color:#e8e6e0;">{emoji} {label}</span>
          <div style="flex:1;background:#2a2a38;border-radius:5px;height:18px;overflow:hidden;">
            <div style="width:{pct}%;background:{color};height:100%;border-radius:5px;
                        transition:width 0.5s cubic-bezier(.4,0,.2,1);"></div>
          </div>
          <span style="width:42px;text-align:right;color:#7a7880;font-size:0.78rem;">{pct}%</span>
        </div>"""
    html += "</div>"
    return html


def _empty_bars_html() -> str:
    html = '<div style="padding:4px 0;">'
    for label in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]:
        color = EMOTION_COLORS[label]
        emoji = EMOTION_EMOJI[label]
        html += f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:9px;font-size:0.84rem;">
          <span style="width:105px;color:#3a3840;">{emoji} {label}</span>
          <div style="flex:1;background:#1e1e28;border-radius:5px;height:18px;overflow:hidden;">
            <div style="width:0%;background:{color};height:100%;border-radius:5px;"></div>
          </div>
          <span style="width:42px;text-align:right;color:#3a3840;font-size:0.78rem;">—</span>
        </div>"""
    html += "</div>"
    return html


# ─── Journal helpers ───────────────────────────────────────────────────────────
def _render_history() -> str:
    entries = get_entries(limit=6)
    if not entries:
        return "_No entries yet._"
    rows = []
    for e in reversed(entries):
        ts    = e["timestamp"][:16].replace("T", " ")
        dom   = e["dominant"]
        emoji = EMOTION_EMOJI.get(dom, "")
        pct   = round(e["emotions"][0]["score"] * 100)
        snip  = e["transcription"][:70] + ("…" if len(e["transcription"]) > 70 else "")
        rows.append(f"**{ts}** {emoji} {dom} `{pct}%`  \n_{snip}_")
    return "\n\n---\n\n".join(rows)


def generate_summary() -> str:
    entries = get_entries(limit=10)
    if len(entries) < 3:
        return "_Add at least 3 entries to generate a summary._"
    lines = []
    for e in entries:
        ts   = e["timestamp"][:16].replace("T", " ")
        dom  = e["dominant"]
        pct  = round(e["emotions"][0]["score"] * 100)
        snip = e["transcription"][:100]
        lines.append(f"- [{ts}] {dom} ({pct}%): \"{snip}\"")
    prompt = (
        "You are analyzing someone's emotion journal. Here are their recent entries:\n\n"
        + "\n".join(lines)
        + "\n\nWrite a warm, empathetic 2–3 sentence summary of their emotional pattern. "
        "Note any trends or shifts. Be concise and insightful."
    )
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Ollama unavailable: {e}"


def clear_journal() -> str:
    clear_entries()
    return "_No entries yet._"


# ─── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(audio, text_input: str, backend: str, use_tts: bool):
    # ── resolve input ──────────────────────────────────────────────────────────
    if audio is not None:
        try:
            transcription = transcribe(audio)
        except Exception as e:
            raise gr.Error(f"STT failed: {e}") from e
    elif text_input and text_input.strip():
        transcription = text_input.strip()
    else:
        return (
            "⚠️ Upload an audio file or type an entry.",
            _empty_bars_html(), None, "", _render_history(),
        )

    # ── classify ───────────────────────────────────────────────────────────────
    clf = ollama_clf if backend == "Ollama (zero-shot)" else transformer_clf
    try:
        emotions = clf.predict(transcription)
    except Exception as e:
        raise gr.Error(str(e)) from e

    top       = emotions[0]
    top_label = top["label"].lower()
    top_pct   = round(top["score"] * 100)
    emoji     = EMOTION_EMOJI.get(top_label, "")

    # ── TTS ────────────────────────────────────────────────────────────────────
    tts_path = None
    if use_tts:
        tts_path = build_tts_feedback(
            transcription, top_label, top["score"], speaker=speaker
        )

    # ── store ──────────────────────────────────────────────────────────────────
    add_entry({
        "timestamp":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transcription": transcription,
        "emotions":      emotions,
        "dominant":      top_label,
    })

    status  = f"{emoji} **{top_label.capitalize()}** — {top_pct}% confidence  _(via {backend})_"
    total   = len(get_entries(limit=99999))
    summary = generate_summary() if total % SUMMARY_TRIGGER == 0 else ""

    return (
        status,
        _emotion_bars_html(emotions),
        tts_path,
        summary,
        _render_history(),
    )


# ─── UI ────────────────────────────────────────────────────────────────────────
css = """
body, .gradio-container {
    background: #0f0f13 !important;
    color: #e8e6e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }

.card {
    background: #16161c !important;
    border: 1px solid #2a2a38 !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
}
.step-label {
    font-size: 0.68rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #7a7880;
    margin: 0 0 6px 0;
}
.gr-button-primary {
    background: #c4a882 !important; color: #0f0f13 !important;
    border: none !important; font-weight: 600 !important; border-radius: 8px !important;
}
.gr-button-secondary {
    background: transparent !important; color: #c4a882 !important;
    border: 1px solid #c4a882 !important; border-radius: 8px !important;
}
footer, .footer, .built-with, #footer { display: none !important; }
"""

with gr.Blocks(title="Emotion Voice Journal") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:2rem 1rem 1rem;">
      <h1 style="font-size:2rem;color:#c4a882;margin:0;">Emotion Journal</h1>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: two stacked cards ───────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<p class="step-label">INPUT</p>')
                with gr.Tabs():
                    with gr.Tab("Text"):
                        text_in = gr.Textbox(
                            placeholder="Type your journal entry…",
                            label="",
                            lines=4,
                        )
                    with gr.Tab("Voice"):
                        audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Record or upload audio",
                        )

            with gr.Column(elem_classes=["card"]):
                gr.HTML('<p class="step-label">options</p>')
                backend_radio = gr.Radio(
                    choices=["RoBERTa (transformer)", "Ollama (zero-shot)"],
                    value="RoBERTa (transformer)",
                    label="Emotion backend",
                )
                tts_toggle = gr.Checkbox(label="Speak feedback aloud", value=True)
                run_btn = gr.Button("Analyse →", variant="primary", size="lg")

        # ── Right: results ────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<p class="step-label">EMOTION DETECTION</p>')
            status_out   = gr.Markdown("_waiting for input…_")
            emotion_html = gr.HTML(_empty_bars_html())

            with gr.Column(visible=True) as tts_col:
                gr.HTML('<p class="step-label" style="margin-top:16px;">AUDIO FEEDBACK</p>')
                tts_out = gr.Audio(label="", type="filepath", autoplay=True)

    gr.HTML("<hr style='border-color:#2a2a38;margin:1.2rem 0;'>")

    # ── Summary + History ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<p class="step-label">session summary</p>')
            summary_out = gr.Markdown("_Appears automatically every 5 entries, or click below._")
            summary_btn = gr.Button("Summarise now", variant="secondary", size="sm")

        with gr.Column(scale=1):
            with gr.Row():
                gr.HTML('<p class="step-label" style="flex:1;margin:0;align-self:center;">recent entries</p>')
                clear_btn = gr.Button("Clear", variant="secondary", size="sm")
            history_out = gr.Markdown("_No entries yet._")

    # ── Wire up ────────────────────────────────────────────────────────────────
    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_in, text_in, backend_radio, tts_toggle],
        outputs=[status_out, emotion_html, tts_out, summary_out, history_out],
    )
    summary_btn.click(fn=generate_summary, outputs=[summary_out])
    clear_btn.click(fn=clear_journal, outputs=[history_out])
    tts_toggle.change(fn=lambda on: gr.update(visible=on), inputs=[tts_toggle], outputs=[tts_col])



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, css=css)
