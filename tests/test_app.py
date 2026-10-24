import io

import pytest
from flask import Flask
from PIL import Image

from app.db import Database
from app.icons import icon
from app.routes import bp


class FakePredictor:
    def predict(self, data):
        return {"label": "Real", "confidence": 0.9, "prob_ai": 0.1, "threshold": 0.5}


@pytest.fixture
def client(tmp_path):
    app = Flask("app")
    app.config.update(TESTING=True, MODEL_NAME="visionguard", MODEL_VERSION="test",
                      THRESHOLD=0.5, MAX_CONTENT_LENGTH=1024 * 1024)
    app.jinja_env.globals["icon"] = icon
    app.extensions["db"] = Database(f"sqlite:///{tmp_path.as_posix()}/test.db")
    app.extensions["predictor"] = FakePredictor()
    app.register_blueprint(bp)
    return app.test_client()


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "PNG")
    return buf.getvalue()


JSON_HEADER = {"X-Requested-With": "fetch"}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["database"] == "ok"


def test_rejects_wrong_file_type(client):
    data = {"image": (io.BytesIO(b"hello"), "notes.txt")}
    r = client.post("/predict", data=data, headers=JSON_HEADER)
    assert r.status_code == 400
    assert "Unsupported file type" in r.get_json()["error"]


def test_rejects_undecodable_image(client):
    data = {"image": (io.BytesIO(b"this is not an image"), "fake.jpg")}
    r = client.post("/predict", data=data, headers=JSON_HEADER)
    assert r.status_code == 400
    assert "could not be decoded" in r.get_json()["error"]


def test_rejects_oversized_file(client):
    data = {"image": (io.BytesIO(b"x" * (2 * 1024 * 1024)), "big.png")}
    r = client.post("/predict", data=data, headers=JSON_HEADER)
    assert r.status_code == 413


def test_valid_upload_is_logged_to_history(client):
    data = {"image": (io.BytesIO(png_bytes()), "ok.png")}
    r = client.post("/predict", data=data, headers=JSON_HEADER)
    assert r.status_code == 200
    assert r.get_json()["label"] == "Real"
    page = client.get("/history")
    assert page.status_code == 200
    assert b"ok.png" in page.data