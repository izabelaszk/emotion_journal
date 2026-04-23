FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps (ffmpeg needed by Whisper, rust+cargo needed by tiktoken)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Point both `python` and `pip` at 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python3.11 -m ensurepip --upgrade && \
    ln -sf $(python3.11 -m pip show pip | grep Location | awk '{print $2}')/../../../bin/pip3.11 /usr/local/bin/pip || true && \
    python3.11 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
# Install GPU-enabled PyTorch first, then remaining deps, then whisper explicitly
RUN python -m pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir openai-whisper

COPY . .

# Pre-download Whisper base model at build time (speeds up first run)
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 7860 8000

# Default: run the Gradio demo UI
# Override CMD to run the FastAPI server instead:
#   docker run ... python api.py
CMD ["python", "app.py"]
