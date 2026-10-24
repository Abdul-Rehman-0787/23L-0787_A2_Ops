FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt
# 2) source code last: it changes often, so it must not invalidate the pip layer
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser run.py .

RUN mkdir -p /app/instance /app/models && chown -R appuser:appuser /app/instance
USER appuser

ENV HOST=0.0.0.0 PORT=5000
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)" || exit 1

CMD ["python", "run.py"]