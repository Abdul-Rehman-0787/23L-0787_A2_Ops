import logging

import requests

log = logging.getLogger("visionguard")


class LlmError(Exception):
    """Artist could not answer. The message is safe to show to the user."""


class OllamaLlm:
    def __init__(self, url, model, timeout, maxTokens, contextSize=2048):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.maxTokens = maxTokens
        self.contextSize = contextSize

    def chat(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": self.maxTokens,
                        "num_ctx": self.contextSize},
        }
        try:
            response = requests.post(self.url + "/api/chat", json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError:
            raise LlmError("Artist is offline. Please start Ollama and try again.") from None
        except requests.exceptions.Timeout:
            raise LlmError("Artist took too long to answer. Please try a shorter question.") from None
        if response.status_code == 404:
            raise LlmError("The language model is not installed. Run: ollama pull " + self.model)
        if response.status_code != 200:
            raise LlmError("Artist could not answer right now. Please try again.")
        data = response.json()
        answer = data.get("message", {}).get("content", "").strip()
        promptTokens = data.get("prompt_eval_count", 0)
        log.info("Ollama %s: prompt %d tokens in %.1f s, answer %d tokens in %.1f s",
                 self.model, promptTokens, data.get("prompt_eval_duration", 0) / 1e9,
                 data.get("eval_count", 0), data.get("eval_duration", 0) / 1e9)
        if promptTokens > self.contextSize * 0.8:
            log.warning("The prompt is close to the context limit (%d of %d tokens)",
                        promptTokens, self.contextSize)
        if answer == "":
            raise LlmError("Artist returned an empty answer. Please try again.")
        return answer


class GroqLlm:
    """Hosted language model through Groq's OpenAI-compatible chat endpoint."""

    def __init__(self, apiKey, model, timeout, maxTokens,
                 url="https://api.groq.com/openai/v1/chat/completions"):
        self.apiKey = apiKey
        self.model = model
        self.timeout = timeout
        self.maxTokens = maxTokens
        self.url = url

    def chat(self, messages):
        if self.apiKey == "":
            log.error("GROQ_API_KEY is not set")
            raise LlmError("Artist is not set up yet. Please try again later.")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.maxTokens,
        }
        headers = {"Authorization": "Bearer " + self.apiKey}
        try:
            response = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise LlmError("Artist took too long to answer. Please try a shorter question.") from None
        except requests.exceptions.RequestException:
            raise LlmError("Artist is offline. Please try again in a moment.") from None
        if response.status_code == 429:
            raise LlmError("Artist is busy right now. Please wait a minute and try again.")
        if response.status_code in (401, 403):
            log.error("Groq rejected the API key (HTTP %d)", response.status_code)
            raise LlmError("Artist is not set up correctly. Please try again later.")
        if response.status_code != 200:
            log.error("Groq error HTTP %d: %s", response.status_code, response.text[:300])
            raise LlmError("Artist could not answer right now. Please try again.")
        data = response.json()
        choices = data.get("choices") or [{}]
        answer = (choices[0].get("message", {}).get("content") or "").strip()
        usage = data.get("usage", {})
        log.info("Groq %s: prompt %d tokens, answer %d tokens", self.model,
                 usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        if answer == "":
            raise LlmError("Artist returned an empty answer. Please try again.")
        return answer


def makeLlm(config):
    """Picks the language model from LLM_PROVIDER ("ollama" locally, "groq" when hosted)."""
    provider = str(getattr(config, "LLM_PROVIDER", "ollama")).strip().lower()
    if provider == "groq":
        return GroqLlm(config.GROQ_API_KEY, config.GROQ_MODEL, config.LLM_TIMEOUT,
                       config.LLM_MAX_TOKENS, config.GROQ_URL)
    return OllamaLlm(config.OLLAMA_URL, config.OLLAMA_MODEL, config.LLM_TIMEOUT,
                     config.LLM_MAX_TOKENS, config.LLM_CONTEXT)