import io

from PIL import Image

from app.routes import safeNext


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def testRegisterSendsCode(client, mails):
    response = client.post("/register", data={"email": "New@Example.com", "password": "correct horse"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/verify?email=")
    assert len(mails) == 1
    assert mails[0]["to"] == "new@example.com"


def testWrongCodeIsRefused(client):
    client.post("/register", data={"email": "a@example.com", "password": "correct horse"})
    response = client.post("/verify", data={"email": "a@example.com", "code": "abcdef"})
    assert response.status_code == 400
    assert client.get("/history", follow_redirects=False).status_code == 303


def testLoginAndLogout(client, signIn):
    email = signIn()
    assert email in client.get("/").text
    assert client.get("/history", follow_redirects=False).status_code == 200
    client.post("/logout")
    assert client.get("/history", follow_redirects=False).status_code == 303


def testWrongPasswordAndUnknownEmailLookTheSame(client, signIn):
    signIn()
    client.post("/logout")
    wrong = client.post("/login", data={"email": "user@example.com", "password": "wrong password"})
    unknown = client.post("/login", data={"email": "nobody@example.com", "password": "wrong password"})
    assert wrong.status_code == 401
    assert unknown.status_code == 401
    assert "incorrect" in wrong.text
    assert "incorrect" in unknown.text


def testSecondLoginSendsNewCode(client, mails, signIn, lastCode):
    signIn()
    client.post("/logout")
    before = len(mails)
    response = client.post("/login", data={"email": "user@example.com", "password": "correct horse"},
                           follow_redirects=False)
    assert response.status_code == 303
    assert len(mails) == before + 1
    client.post("/verify", data={"email": "user@example.com", "code": lastCode()})
    assert client.get("/history", follow_redirects=False).status_code == 200


def testHistoryIsPrivate(client, signIn):
    signIn("a@example.com")
    files = {"image": ("alice-photo.png", makePng(), "image/png")}
    response = client.post("/predict", files=files, headers={"X-Requested-With": "fetch"})
    assert response.status_code == 200
    assert "alice-photo.png" in client.get("/history").text
    client.post("/logout")
    signIn("b@example.com")
    assert "alice-photo.png" not in client.get("/history").text


def testVerifiedEmailCannotBeRegisteredAgain(client, signIn):
    signIn()
    client.post("/logout")
    response = client.post("/register", data={"email": "user@example.com", "password": "another one 1"})
    assert response.status_code == 400


def testNextCannotLeaveTheSite():
    assert safeNext("/history") == "/history"
    assert safeNext("//evil.example") == "/"
    assert safeNext("https://evil.example") == "/"
    assert safeNext("/\\evil.example") == "/"

def testFivePasswordMistakesLockTheAccount(client, signIn):
    signIn()
    client.post("/logout")
    wrongData = {"email": "user@example.com", "password": "wrong password"}
    for _ in range(4):
        assert client.post("/login", data=wrongData).status_code == 401
    assert client.post("/login", data=wrongData).status_code == 429
    rightData = {"email": "user@example.com", "password": "correct horse"}
    assert client.post("/login", data=rightData).status_code == 429     # still locked


def testCorrectPasswordResetsTheCounter(client, signIn):
    signIn()
    client.post("/logout")
    wrongData = {"email": "user@example.com", "password": "wrong password"}
    rightData = {"email": "user@example.com", "password": "correct horse"}
    for _ in range(4):
        assert client.post("/login", data=wrongData).status_code == 401
    assert client.post("/login", data=rightData, follow_redirects=False).status_code == 303
    for _ in range(4):
        assert client.post("/login", data=wrongData).status_code == 401


def testUnknownEmailIsCountedToo(client):
    data = {"email": "nobody@example.com", "password": "wrong password"}
    codes = [client.post("/login", data=data).status_code for _ in range(5)]
    assert codes == [401, 401, 401, 401, 429]