import io

from PIL import Image


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def upload(client):
    files = {"image": ("red.png", makePng(), "image/png")}
    return client.post("/predict", files=files, headers={"X-Requested-With": "fetch"})


def testUploadQueuesGraphUpdate(client, signIn, graphWriter):
    signIn()
    assert upload(client).status_code == 200
    assert len(graphWriter.calls) == 1
    payload = graphWriter.calls[0]
    assert payload["userId"] == 1
    assert payload["analysisId"] == 1
    assert payload["label"] == "Real"
    assert payload["filename"] == "red.png"
    assert len(payload["sha256"]) == 64
    assert "@" not in str(payload)          # no email address ever goes into the graph


def testGraphProblemDoesNotBreakUpload(client, signIn, graphWriter):
    signIn()
    graphWriter.error = True
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["saved"] is True


def testGraphOffQueuesNothing(client, signIn, graphWriter, graphStore):
    signIn()
    graphStore.enabled = False
    assert upload(client).status_code == 200
    assert graphWriter.calls == []