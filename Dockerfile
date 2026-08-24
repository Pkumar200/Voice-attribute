FROM python:3.11-slim

# ffmpeg: audio transcoding for arbitrary input codecs (Task 1)
# curl: used only by the container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
# torch/torchaudio from the CPU-only wheel index: this service never uses a
# GPU, and the default PyPI wheels drag in ~2GB of unused nvidia-*/CUDA
# packages that just make the build slower and flakier for no benefit here.
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 \
        torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

COPY app ./app

# Model weights are pulled from the Hugging Face Hub on first startup and
# cached in this volume-backed directory (see docker-compose.yml) so
# restarts don't re-download ~300MB every time. No caller audio is ever
# written here - this cache holds model weights only.
ENV HF_HOME=/srv/.cache/huggingface
RUN mkdir -p /srv/.cache/huggingface

ENV MODEL_BACKEND=auto \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
