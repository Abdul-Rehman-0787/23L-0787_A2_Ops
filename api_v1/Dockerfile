FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app
USER appuser

# Download the model into the image at build time (public repo, free), then stay offline at runtime.
RUN python -c "from huggingface_hub import snapshot_download; print(snapshot_download('JONNYVERSE/ai-image-detect-distilled-ONNX'))"
ENV HF_HUB_OFFLINE=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]