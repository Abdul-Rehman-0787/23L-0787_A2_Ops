import re

import pytest
from fastapi.testclient import TestClient

from app import createApp


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value

    def exists(self, key):
        return 1 if key in self.values else 0

    def delete(self, key):
        self.values.pop(key, None)

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    def expire(self, key, seconds):
        pass

    def ping(self):
        return True


class FakePredictor:
    def predict(self, data):
        return {"label": "Real", "confidence": 0.9, "prob_ai": 0.1, "threshold": 0.48}


class FakeLlm:
    def __init__(self):
        self.lastMessages = None
        self.error = ""

    def chat(self, messages):
        from app.llm import LlmError
        self.lastMessages = messages
        if self.error:
            raise LlmError(self.error)
        return "Fake answer"


class FakeKnowledge:
    def __init__(self):
        self.error = False
        self.hits = [{"source": "faq.md", "title": "FAQ", "section": "Are my images stored?",
                      "text": "Images are not stored.", "score": 0.8}]

    def search(self, question, count=3):
        if self.error:
            raise RuntimeError("index is broken")
        return self.hits


class FakeGraph:
    """Stands in for Neo4j in tests."""

    def __init__(self):
        self.enabled = True


class GraphWriter:
    """Stands in for the Celery task that updates the graph."""

    def __init__(self):
        self.calls = []
        self.error = False

    def __call__(self, analysis):
        if self.error:
            raise RuntimeError("queue is down")
        self.calls.append(analysis)


@pytest.fixture
def mails():
    return []


@pytest.fixture
def knowledge():
    return FakeKnowledge()


@pytest.fixture
def llm():
    return FakeLlm()


@pytest.fixture
def graphStore():
    return FakeGraph()


@pytest.fixture
def graphWriter():
    return GraphWriter()


@pytest.fixture
def client(tmp_path, mails, llm, knowledge, graphStore, graphWriter):
    databaseUrl = "sqlite:///" + str(tmp_path / "test.db").replace("\\", "/")

    def collectMail(toAddress, subject, bodyText):
        mails.append({"to": toAddress, "subject": subject, "body": bodyText})

    app = createApp(databaseUrl=databaseUrl, predictor=FakePredictor(),
                    redisClient=FakeRedis(), mailSender=collectMail, llm=llm,
                    knowledge=knowledge, graph=graphStore, graphWriter=graphWriter)
    return TestClient(app)


@pytest.fixture
def lastCode(mails):
    def read():
        return re.search(r"\b(\d{6})\b", mails[-1]["body"]).group(1)
    return read


@pytest.fixture
def signIn(client, lastCode):
    def run(email="user@example.com", password="correct horse"):
        client.post("/register", data={"email": email, "password": password})
        client.post("/verify", data={"email": email, "code": lastCode()})
        return email
    return run