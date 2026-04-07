FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg git git-lfs libsndfile1 libgl1 \
    libglib2.0-0 wget espeak curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (which provides uvx) — used by tribev2 for transcription
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download whisperx model (public repo, no token needed)
# This avoids 429 rate limits when uvx whisperx runs at inference time
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3', cache_dir='/tmp/hf_hub_cache')"

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# Create shared HF cache directories
RUN mkdir -p /tmp/hf_hub_cache /tmp/whisper_cache && chown -R appuser:appuser /tmp/hf_hub_cache /tmp/whisper_cache
# Make uvx available to appuser
RUN cp /root/.local/bin/uvx /usr/local/bin/uvx 2>/dev/null || true
RUN cp /root/.local/bin/uv /usr/local/bin/uv 2>/dev/null || true

USER appuser

ENV CUDA_VISIBLE_DEVICES=""
ENV HF_TOKEN=${HF_TOKEN}
ENV TRIBE_CKPT=facebook/tribev2
ENV MAX_VIDEO_DURATION=120
ENV MAX_AUDIO_DURATION=120

EXPOSE 7860
CMD ["python", "app.py"]
