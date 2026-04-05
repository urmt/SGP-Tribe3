FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg git git-lfs libsndfile1 libgl1 \
    libglib2.0-0 wget espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first to avoid nvidia resolver backtracking
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchaudio==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["python", "app.py"]
