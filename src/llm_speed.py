import sys
import time
from datetime import datetime

import requests

from app.artist import buildMessages
from app.config import Config


class FakeScan:
    id = 295
    created_at = datetime(2026, 11, 17, 12, 0)
    filename = "example.png"
    label = "Real"
    prob_ai = 0.31
    threshold = 0.48
    model_version = "2"


hits = [{"source": "faq.md", "section": "Why was my painting flagged as AI?",
         "text": "The model reacts to style more than to real signs of AI generation. Human-made "
                 "paintings, illustrations and collages with bright colours or painterly texture "
                 "are often flagged as AI. This is a known weakness of a model trained on a small "
                 "dataset.", "score": 0.8}]
messages = buildMessages([FakeScan()], [], "Why was my painting flagged as AI?", hits)

models = sys.argv[1:] or [Config.OLLAMA_MODEL]
for model in models:
    options = {"temperature": 0.2, "num_predict": Config.LLM_MAX_TOKENS, "num_ctx": Config.LLM_CONTEXT}
    try:
        requests.post(Config.OLLAMA_URL + "/api/chat", timeout=300, json={
            "model": model, "stream": False, "keep_alive": "5m", "options": {"num_predict": 1},
            "messages": [{"role": "user", "content": "hi"}]})          # loads the model first
        start = time.perf_counter()
        reply = requests.post(Config.OLLAMA_URL + "/api/chat", timeout=300, json={
            "model": model, "stream": False, "keep_alive": "5m", "options": options,
            "messages": messages})
    except requests.exceptions.RequestException as error:
        print(model, "failed:", error)
        continue
    if reply.status_code != 200:
        print(model, "failed (is it installed? run: ollama pull " + model + ")")
        continue
    total = time.perf_counter() - start
    data = reply.json()
    generated = data.get("eval_count", 0)
    seconds = max(data.get("eval_duration", 1) / 1e9, 0.001)
    print()
    print(model, "| prompt", data.get("prompt_eval_count", 0), "tokens in",
          round(data.get("prompt_eval_duration", 0) / 1e9, 1), "s | answer", generated,
          "tokens at", round(generated / seconds, 1), "tokens/s | total", round(total, 1), "s")
    print("ANSWER:", data["message"]["content"][:400])