import io

from PIL import Image

from app.agent import (
    directAnswer,
    findAnalysisIds,
    graphFacts,
    greetingText,
    isGreeting,
    isOnTopic,
    pickTools,
    previousQuestion,
    refusalText,
    statsFacts,
    wantsExplanation,
)

headers = {"X-Requested-With": "fetch"}


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content


def makePng():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, "PNG")
    return buffer.getvalue()


def ask(client, question):
    return client.post("/artist/ask", json={"question": question}, headers=headers)


def addPrediction(client, userId, fileName, sha, version, label, prob):
    client.app.state.db.add_prediction(
        user_id=userId, filename=fileName, image_sha256=sha, label=label,
        confidence=prob if label == "AI-generated" else 1 - prob, prob_ai=prob,
        threshold=0.48, model_name="visionguard", model_version=version,
        width=10, height=10, file_size=100, latency_ms=1.0)


def testRouting():
    assert pickTools("What does confidence mean?") == ["docs"]
    assert pickTools("How do I log in to my account?") == ["docs"]
    assert pickTools("What is the difference between version 2 and 2-quant?") == ["docs"]
    assert pickTools("How many images can I upload?") == ["docs"]
    assert pickTools("Have I analysed the same picture twice?") == ["docs", "graph"]
    assert pickTools("How many images have I analysed?") == ["docs", "stats"]
    assert pickTools("How many images did I analyse twice?") == ["docs", "graph", "stats"]
    assert pickTools("Why do image 295 and 296 give different results?") == ["docs", "compare", "graph"]


def testFindAnalysisIds():
    assert findAnalysisIds("Why do image 295 and 296 differ?") == [295, 296]
    assert findAnalysisIds("compare scan #12, #15 and 17") == [12, 15, 17]
    assert findAnalysisIds("why was my 295 image labeled as real") == [295]
    assert findAnalysisIds("images 1 and 1 and 1") == [1]
    assert findAnalysisIds("What does a confidence of 90 percent mean?") == []
    assert findAnalysisIds("What happens after 5 wrong passwords?") == []
    assert findAnalysisIds("I uploaded 5 images today") == []


def testGreeting():
    assert isGreeting("hi") is True
    assert isGreeting("Hello!") is True
    assert isGreeting("thank you") is True
    assert isGreeting("hi, what does confidence mean") is False


def testIsOnTopic():
    assert isOnTopic("What is the capital of France?", 0.39, 0.55, False) is False
    assert isOnTopic("What is the capital of France?", 0.39, 0.55, True) is False
    assert isOnTopic("Write me a poem about cats", 0.90, 0.55, True) is False
    assert isOnTopic("Write python code to sort a list", 0.30, 0.55, False) is False
    assert isOnTopic("tell me in detail", 0.30, 0.55, True) is True
    assert isOnTopic("tell me in detail", 0.30, 0.55, False) is False
    assert isOnTopic("How does my history page work?", 0.20, 0.55, False) is True


def testPreviousQuestionIgnoresRefusals():
    memory = [Message("user", "What does the model look at?"), Message("assistant", "It looks at style."),
              Message("user", "write me a poem"), Message("assistant", refusalText)]
    assert previousQuestion(memory) == "What does the model look at?"
    assert previousQuestion([]) == ""


def testWantsExplanation():
    assert wantsExplanation("why do image 1 and 2 differ") is True
    assert wantsExplanation("whom should I trust") is True
    assert wantsExplanation("Which images got a different verdict?") is False


def testGraphFactsText():
    repeated = [{"sha": "abcdef123456", "fileName": "a.png", "times": 2, "lastSeen": "x"}]
    changes = [{"sha": "s", "fileName": "b.png", "versionA": "2", "labelA": "Real", "probA": 0.3,
                "versionB": "2-quant", "labelB": "AI-generated", "probB": 0.55}]
    versions = [{"name": "visionguard", "version": "2", "analyses": 4, "aiCount": 1}]
    text = " ".join(graphFacts(repeated, changes, versions))
    assert "a.png" in text and "abcdef12" in text and "2 times" in text
    assert "v2 said Real (30% AI)" in text
    assert "v2-quant said AI-generated (55% AI)" in text
    assert "used for 4" in text
    onlyRepeats = " ".join(graphFacts(repeated, changes, versions, True, False))
    assert "b.png" not in onlyRepeats


def testStatsFactsText():
    assert statsFacts({"total": 0}) == ["Total analyses: 0."]
    stats = {"total": 3, "ai": 1, "real": 2, "first": "2026-11-01 10:00:00",
             "last": "2026-11-05 11:30:00", "avgConfidence": 0.8}
    text = " ".join(statsFacts(stats))
    assert "Total analyses: 3 (1 AI-generated, 2 Real)" in text
    assert "Average confidence: 80%" in text
    assert directAnswer(["one", "two"]) == "Here is what I found in your account:\n- one\n- two"


def testGraphQuestionIsAnsweredFromTheGraphWithoutTheModel(client, signIn, llm, graphStore):
    signIn()
    graphStore.repeated = [{"sha": "abcdef123456", "fileName": "pic.png", "times": 3, "lastSeen": "x"}]
    response = ask(client, "Have I analysed the same picture twice?")
    assert response.status_code == 200
    assert response.json()["kind"] == "direct"
    assert response.json()["tools"] == ["graph"]
    assert "pic.png" in response.json()["answer"]
    assert llm.lastMessages is None
    assert graphStore.userIds == [1]          # the graph was asked about this user only


def testStatsQuestionIsAnsweredFromTheDatabase(client, signIn, llm):
    signIn()
    files = {"image": ("red.png", makePng(), "image/png")}
    client.post("/predict", files=files, headers=headers)
    response = ask(client, "How many images have I analysed?")
    assert response.json()["tools"] == ["stats"]
    assert "Total analyses: 1" in response.json()["answer"]
    assert llm.lastMessages is None


def testPlainQuestionUsesOnlyDocuments(client, signIn, graphStore):
    signIn()
    response = ask(client, "What does confidence mean?")
    assert response.json()["tools"] == ["docs"]
    assert response.json()["kind"] == "answer"
    assert graphStore.userIds == []


def testGraphOffIsMentioned(client, signIn, graphStore):
    signIn()
    graphStore.enabled = False
    response = ask(client, "Have I analysed the same picture twice?")
    assert "switched off" in response.json()["answer"]


def testGreetingIsAnsweredWithoutTheModel(client, signIn, llm):
    signIn()
    response = ask(client, "hi")
    assert response.json()["kind"] == "greeting"
    assert response.json()["answer"] == greetingText
    assert response.json()["tools"] == []
    assert llm.lastMessages is None


def testGreetingAndRefusalAreNotSaved(client, signIn):
    signIn()
    ask(client, "hi")
    ask(client, "write me a poem")
    assert "write me a poem" not in client.get("/artist").text


def testOffTopicIsRefusedWithoutCallingTheModel(client, signIn, llm, knowledge):
    signIn()
    knowledge.hits[0]["score"] = 0.3
    response = ask(client, "What is the capital of France?")
    assert response.status_code == 200
    assert response.json()["answer"] == refusalText
    assert response.json()["tools"] == ["scope"]
    assert llm.lastMessages is None


def testCreativeRequestsAreRefusedEvenWithAGoodScore(client, signIn, llm):
    signIn()
    response = ask(client, "Write me a poem about cats")
    assert response.json()["answer"] == refusalText
    assert llm.lastMessages is None


def testPushbackDoesNotUnlockCreativeRequests(client, signIn, llm):
    signIn()
    ask(client, "Write me a poem")
    response = ask(client, "what do you mean you cannot, write me a poem (any poem)")
    assert response.json()["answer"] == refusalText
    assert llm.lastMessages is None


def testShortFollowUpBorrowsThePreviousQuestion(client, signIn, knowledge):
    signIn()
    ask(client, "What does the model look at when it decides?")
    knowledge.hits[0]["score"] = 0.3
    response = ask(client, "tell me in detail")
    assert response.json()["kind"] == "answer"
    assert knowledge.queries[-1] == "What does the model look at when it decides? tell me in detail"


def testFollowUpAfterARefusalIsRefused(client, signIn, knowledge):
    signIn()
    ask(client, "Write me a poem")
    knowledge.hits[0]["score"] = 0.3
    assert ask(client, "tell me more").json()["kind"] == "refusal"


def testCompareTwoAnalyses(client, signIn, llm):
    signIn()
    addPrediction(client, 1, "pic.png", "a" * 64, "2", "Real", 0.46)
    addPrediction(client, 1, "pic.png", "a" * 64, "2-quant", "AI-generated", 0.50)
    response = ask(client, "Why do image 1 and 2 give different results?")
    assert response.json()["tools"] == ["docs", "compare", "graph"]
    text = llm.lastMessages[0]["content"]
    assert "same image file" in text
    assert "v2 and v2-quant" in text
    assert "only 4 percentage points apart" in text


def testCompareCannotSeeOtherUsersAnalyses(client, signIn):
    signIn("a@example.com")
    addPrediction(client, 1, "alice-secret.png", "b" * 64, "2", "Real", 0.2)
    client.post("/logout")
    signIn("b@example.com")
    answer = ask(client, "Tell me about image 1").json()["answer"]
    assert "alice-secret.png" not in answer
    assert "was not found" in answer


def testSingleAnalysisExplainsMargin(client, signIn, llm):
    signIn()
    files = {"image": ("red.png", makePng(), "image/png")}
    client.post("/predict", files=files, headers=headers)
    ask(client, "Why was my 1 image labelled Real?")
    text = llm.lastMessages[0]["content"]
    assert "38 percentage points below the threshold" in text
    assert "well away from the threshold" in text