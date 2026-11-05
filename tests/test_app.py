import io

from fastapi.testclient import TestClient
from PIL import Image

from app import createApp


class FakePredictor:
    def predict(self, data):
        return {"label": "Real", "confidence": 0.9, "prob_ai": 0.1, "threshold": 0.48}


def makeClient(tmpPath):
    databaseUrl = "sqlite:///" + str(tmpPath / "test.db").replace("\\", "/")
    return TestClient(createApp(databaseUrl=databaseUrl, predictor=FakePredictor()))


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def testHealth(tmp_path):
    response = makeClient(tmp_path).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def testRejectsWrongFileType(tmp_path):
    files = {"image": ("notes.txt", b"hello", "text/plain")}
    assert makeClient(tmp_path).post("/predict", files=files).status_code == 400


def testRejectsUndecodableImage(tmp_path):
    files = {"image": ("fake.png", b"this is not an image", "image/png")}
    assert makeClient(tmp_path).post("/predict", files=files).status_code == 400


def testRejectsOversizedFile(tmp_path):
    client = makeClient(tmp_path)
    justOver = {"image": ("big.png", b"0" * (5 * 1024 * 1024 + 500_000), "image/png")}
    assert client.post("/predict", files=justOver).status_code == 413
    veryBig = {"image": ("huge.png", b"0" * (7 * 1024 * 1024), "image/png")}
    assert client.post("/predict", files=veryBig).status_code == 413


def testValidUploadAppearsInHistory(tmp_path):
    client = makeClient(tmp_path)
    files = {"image": ("red.png", makePng(), "image/png")}
    response = client.post("/predict", files=files, headers={"X-Requested-With": "fetch"})
    assert response.status_code == 200
    assert response.json()["label"] == "Real"
    assert response.json()["saved"] is True
    assert "red.png" in client.get("/history").text