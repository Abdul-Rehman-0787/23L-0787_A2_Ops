from types import SimpleNamespace

from app.llm import GroqLlm, OllamaLlm, makeLlm


def configWith(provider):
    return SimpleNamespace(
        LLM_PROVIDER=provider,
        OLLAMA_URL="http://localhost:11434",
        OLLAMA_MODEL="llama3.2:3b",
        GROQ_API_KEY="test-key",
        GROQ_MODEL="llama-3.1-8b-instant",
        GROQ_URL="https://api.groq.com/openai/v1/chat/completions",
        LLM_TIMEOUT=60,
        LLM_MAX_TOKENS=300,
        LLM_CONTEXT=2048,
    )


def testMakeLlmUsesGroqWhenConfigured():
    llm = makeLlm(configWith("groq"))
    assert isinstance(llm, GroqLlm)


def testMakeLlmUsesGroqCaseInsensitively():
    llm = makeLlm(configWith(" GROQ "))
    assert isinstance(llm, GroqLlm)


def testMakeLlmKeepsOllamaAsLocalDefault():
    llm = makeLlm(configWith("ollama"))
    assert isinstance(llm, OllamaLlm)
