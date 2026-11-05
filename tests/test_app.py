import io

from PIL import Image


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def testHealth(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["redis"] == "ok"


def testRejectsWrongFileType(client, signIn):
    signIn()
    files = {"image": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/predict", files=files).status_code == 400


def testRejectsUndecodableImage(client, signIn):
    signIn()
    files = {"image": ("fake.png", b"this is not an image", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def testRejectsOversizedFile(client, signIn):
    signIn()
    justOver = {"image": ("big.png", b"0" * (5 * 1024 * 1024 + 500_000), "image/png")}
    assert client.post("/predict", files=justOver).status_code == 413
    veryBig = {"image": ("huge.png", b"0" * (7 * 1024 * 1024), "image/png")}
    assert client.post("/predict", files=veryBig).status_code == 413


def testValidUploadAppearsInHistory(client, signIn):
    signIn()
    files = {"image": ("red.png", makePng(), "image/png")}
    response = client.post("/predict", files=files, headers={"X-Requested-With": "fetch"})
    assert response.status_code == 200
    assert response.json()["label"] == "Real"
    assert response.json()["saved"] is True
    assert "red.png" in client.get("/history").text


def testUploadNeedsLogin(client):
    files = {"image": ("red.png", makePng(), "image/png")}
    response = client.post("/predict", files=files, headers={"X-Requested-With": "fetch"})
    assert response.status_code == 401


def testHistoryNeedsLogin(client):
    response = client.get("/history", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


def testPagesLoad(client):
    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 200
    assert client.get("/register").status_code == 200
    assert client.get("/no-such-page").status_code == 404