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
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

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

ENV HF_HUB_CACHE=/tmp/hf_hub_cache
ENV WHISPER_CACHE_DIR=/tmp/whisper_cache
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 7860
CMD ["python", "app.py"]
