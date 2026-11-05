import requests


class LlmError(Exception):
    """Artist could not answer. The message is safe to show to the user."""


class OllamaLlm:
    def __init__(self, url, model, timeout, maxTokens):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.maxTokens = maxTokens

    def chat(self, messages):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": self.maxTokens, "num_ctx": 2048},
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
        answer = response.json().get("message", {}).get("content", "").strip()
        if answer == "":
            raise LlmError("Artist returned an empty answer. Please try again.")
        return answer