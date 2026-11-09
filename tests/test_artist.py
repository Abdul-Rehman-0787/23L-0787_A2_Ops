import io

from PIL import Image

from app.artist import buildMessages

headers = {"X-Requested-With": "fetch"}


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def ask(client, question):
    return client.post("/artist/ask", json={"question": question}, headers=headers)


def testArtistNeedsLogin(client):
    assert client.get("/artist", follow_redirects=False).status_code == 303
    assert ask(client, "hello").status_code == 401


def testArtistAnswersAndSavesChat(client, signIn):
    signIn()
    response = ask(client, "What does 70 percent mean?")
    assert response.status_code == 200
    assert response.json()["answer"] == "Fake answer"
    page = client.get("/artist").text
    assert "What does 70 percent mean?" in page
    assert "Fake answer" in page


def testArtistSeesOnlyOwnData(client, signIn, llm):
    signIn("a@example.com")
    files = {"image": ("alice-photo.png", makePng(), "image/png")}
    client.post("/predict", files=files, headers=headers)
    ask(client, "Alice private question")
    assert "alice-photo.png" in llm.lastMessages[0]["content"]
    client.post("/logout")
    signIn("b@example.com")
    ask(client, "Bob question")
    assert "alice-photo.png" not in llm.lastMessages[0]["content"]
    assert "Alice private question" not in client.get("/artist").text


def testQuestionValidation(client, signIn):
    signIn()
    assert ask(client, "").status_code == 400
    assert ask(client, "x" * 501).status_code == 400
    assert client.post("/artist/ask", content=b"not json", headers=headers).status_code == 400


def testArtistOfflineMessage(client, signIn, llm):
    signIn()
    llm.error = "Artist is offline. Please start Ollama and try again."
    response = ask(client, "hello")
    assert response.status_code == 503
    assert "offline" in response.json()["error"]
    assert "hello" not in client.get("/artist").text


def testClearChat(client, signIn):
    signIn()
    ask(client, "remember this question")
    assert "remember this question" in client.get("/artist").text
    client.post("/artist/clear")
    assert "remember this question" not in client.get("/artist").text


def testPromptHasRulesAndLimits():
    messages = buildMessages([], [], "Hello?")
    assert messages[0]["role"] == "system"
    assert "not proof" in messages[0]["content"]
    assert "none yet" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Hello?"}

def testAnswerComesWithSources(client, signIn, llm):
    signIn()
    response = ask(client, "Are my images stored?")
    sources = response.json()["sources"]
    assert sources[0]["source"] == "faq.md"
    assert "Images are not stored." in llm.lastMessages[0]["content"]
    assert "faq.md" in client.get("/artist").text


def testWeakPassagesAreNotUsed(client, signIn, llm, knowledge):
    signIn()
    knowledge.hits[0]["score"] = 0.1
    response = ask(client, "something unrelated")
    assert response.json()["sources"] == []
    assert "Images are not stored." not in llm.lastMessages[0]["content"]


def testArtistStillAnswersWhenSearchBreaks(client, signIn, knowledge):
    signIn()
    knowledge.error = True
    assert ask(client, "hello").status_code == 200


def testPromptContainsPassages():
    hits = [{"source": "faq.md", "section": "Storage", "text": "Images are not stored.", "score": 0.7}]
    content = buildMessages([], [], "Hello?", hits)[0]["content"]
    assert "[1] faq.md > Storage" in content
    assert "Images are not stored." in content
    assert "No reference passage matched" in buildMessages([], [], "Hello?")[0]["content"]