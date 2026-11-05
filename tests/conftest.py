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


@pytest.fixture
def mails():
    return []


@pytest.fixture
def client(tmp_path, mails):
    databaseUrl = "sqlite:///" + str(tmp_path / "test.db").replace("\\", "/")

    def collectMail(toAddress, subject, bodyText):
        mails.append({"to": toAddress, "subject": subject, "body": bodyText})

    app = createApp(databaseUrl=databaseUrl, predictor=FakePredictor(),
                    redisClient=FakeRedis(), mailSender=collectMail)
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