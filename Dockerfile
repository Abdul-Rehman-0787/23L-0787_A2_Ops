FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when the requirements file changes.
#    Hosted image = no TensorFlow (inference is remote).
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

# 2) source code last: it changes often, so it must not invalidate the pip layer
COPY app ./app
COPY src ./src
COPY knowledge ./knowledge
COPY run.py .

RUN mkdir -p /app/instance /app/.cache && chown -R appuser:appuser /app
USER appuser

# 3) Bake the document index and the embedding model into the image.
#    Render's disk is wiped on every restart, so doing this at build time means
#    Artist answers its first question without downloading anything.
RUN python -m src.index_knowledge

ENV HOST=0.0.0.0 PORT=5000
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '5000'), timeout=4)" || exit 1

CMD ["python", "run.py"]