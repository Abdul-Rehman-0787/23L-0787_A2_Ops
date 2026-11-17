import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .artist import buildMessages

log = logging.getLogger("visionguard")

repeatPhrases = ["same image", "same picture", "same photo", "same file", "analysed again",
                 "analyzed again", "uploaded again", "twice", "duplicate", "repeated"]
changePhrases = ["changed", "different verdict", "different result", "disagree", "lineage"]
countPhrases = ["how many", "how much", "total", "number of", "average", "how often"]
recentPhrases = ["most recent", "latest", "last time", "when did"]
ownDataPhrases = ["analys", "analyz", "uploaded", "history", "my image", "my result", "my scan"]

explainWords = ["why", "explain", "mean", "means", "meaning", "trust", "should", "reason",
                "because", "cause", "caused", "interpret"]
followUpWords = ["detail", "details", "more", "explain", "elaborate", "example", "examples",
                 "simpler", "simply", "shorter", "longer", "summary", "summarize", "continue", "why"]
greetingWords = ["hi", "hello", "hey", "hiya", "salam", "assalamualaikum", "thanks", "thank",
                 "you", "ok", "okay", "bye", "good", "morning", "afternoon", "evening"]

# Requests that VisionGuard is not for.
blockedTasks = ["poem", "poems", "poetry", "story", "stories", "song", "songs", "lyrics",
                "joke", "jokes", "essay", "recipe", "recipes", "python", "javascript"]
# Words that show a question is about VisionGuard even when the document search is unsure.
appWords = ["visionguard", "artist", "image", "images", "picture", "pictures", "photo", "photos",
            "model", "models", "verdict", "verdicts", "result", "results", "confidence",
            "probability", "threshold", "upload", "uploads", "history", "account", "login",
            "logout", "register", "password", "otp", "analysis", "analyses", "analyse",
            "analysed", "analyze", "analyzed", "generated", "version", "versions", "detect",
            "detection", "classifier", "score", "accuracy", "dataset", "training", "trained"]
# Words that introduce history numbers, such as "image 295 and 296".
singularWords = ["image", "picture", "photo", "analysis", "scan", "result", "row", "number", "id"]
pluralWords = ["images", "pictures", "photos", "analyses", "scans", "results", "rows",
               "numbers", "ids"]
linkWords = ["and", "vs", "versus", "with", "to", "or"]

refusalText = ("I can only help with VisionGuard: how it works, what your results mean, "
               "your history and your account. Try asking, for example: "
               "'What does a confidence of 70 percent mean?'")
greetingText = ("Hello! I'm Artist. I can explain your VisionGuard results, how the model works "
                "and your history. For example: 'What does a confidence of 70 percent mean?'")


class AgentState(TypedDict, total=False):
    question: str
    userId: int
    tools: list
    done: list
    scans: list
    memory: list
    hits: list
    facts: list
    bestScore: float
    onTopic: bool
    greeting: bool
    kind: str
    answer: str


def splitWords(text):
    cleaned = ""
    for character in text.lower():
        if character.isalnum():
            cleaned = cleaned + character
        else:
            cleaned = cleaned + " "
    return cleaned.split()


def hasAny(text, phrases):
    for phrase in phrases:
        if phrase in text:
            return True
    return False


def isGreeting(question):
    words = splitWords(question)
    if len(words) == 0:
        return False
    for word in words:
        if word not in greetingWords:
            return False
    return True


def isFollowUp(question):
    words = splitWords(question)
    if len(words) > 5:
        return False
    for word in words:
        if word in followUpWords:
            return True
    return False


def wantsExplanation(question):
    for word in splitWords(question):
        if word in explainWords:
            return True
    return False


def findAnalysisIds(question):
    """Finds history numbers such as 'image 295 and 296', 'scan #12, #15' or 'my 295 image'."""
    words = splitWords(question)
    ids = []
    collecting = False
    for position in range(len(words)):
        word = words[position]
        if word in singularWords or word in pluralWords:
            collecting = True
        elif word.isdigit() and len(word) <= 9:
            nextWord = ""
            if position + 1 < len(words):
                nextWord = words[position + 1]
            if collecting or nextWord in singularWords:
                if int(word) not in ids:
                    ids.append(int(word))
                collecting = True
        elif word not in linkWords:
            collecting = False
    return ids[:4]


ownWords = ["my", "mine", "latest", "last", "recent"]
trustText = ("A verdict is an estimate, not proof. If two results disagree, or a score is close "
             "to the threshold, treat the image as an unclear case and check it by hand.")


def usesOwnData(question):
    for word in splitWords(question):
        if word in ownWords:
            return True
    return False


def premiseNote(question, rows):
    """Corrects a wrong assumption, such as asking why an AI-labelled image was called Real."""
    words = splitWords(question)
    saysReal = "real" in words
    saysAi = "ai" in words or "fake" in words or "generated" in words
    notes = []
    for row in rows:
        if saysReal and not saysAi and row.label != "Real":
            notes.append("Note: you asked about #" + str(row.id) + " being labelled Real, "
                         "but it was labelled " + row.label + ".")
        elif saysAi and not saysReal and row.label == "Real":
            notes.append("Note: you asked about #" + str(row.id) + " being labelled AI, "
                         "but it was labelled Real.")
    return notes


def pickTools(question):
    """Plain rules decide where to look. Documents are always searched."""
    if len(findAnalysisIds(question)) > 0:
        return ["docs", "compare"]
    text = question.lower()
    tools = ["docs"]
    if hasAny(text, repeatPhrases) or hasAny(text, changePhrases):
        tools.append("graph")
    if (hasAny(text, countPhrases) or hasAny(text, recentPhrases)) and hasAny(text, ownDataPhrases):
        tools.append("stats")
    return tools


def previousQuestion(memory):
    """The last question that Artist really answered (refusals and greetings do not count)."""
    previous = ""
    pending = ""
    for item in memory:
        if item.role == "user":
            pending = item.content
        elif item.content != refusalText and item.content != greetingText:
            previous = pending
    return previous


def isOnTopic(question, ownScore, minScore, hasPrevious):
    """The scope gate. Creative requests are refused. Otherwise the question must match the
    documents by itself, use VisionGuard words, or be a short follow-up to an answered question."""
    words = splitWords(question)
    for word in words:
        if word in blockedTasks:
            return False
    if ownScore >= minScore:
        return True
    for word in words:
        if word in appWords:
            return True
    return hasPrevious and isFollowUp(question)


def pickHits(found, minScore, maxPassages):
    hits = []
    for hit in found:
        if hit["score"] >= minScore and len(hits) < maxPassages:
            hits.append(hit)
    return hits


def percent(value):
    return str(round(value * 100)) + "%"


def graphFacts(repeated, changes, versions, askRepeat=True, askChange=True):
    facts = []
    if askRepeat:
        if len(repeated) == 0:
            facts.append("No image was analysed more than once.")
        for row in repeated:
            facts.append("The image '" + row["fileName"] + "' (fingerprint " + row["sha"][:8]
                         + ") was analysed " + str(row["times"]) + " times.")
    if askChange:
        for row in versions:
            facts.append("Model " + row["name"] + " v" + row["version"] + " was used for "
                         + str(row["analyses"]) + " of the user's analyses ("
                         + str(row["aiCount"]) + " labelled AI-generated).")
        if len(changes) == 0:
            facts.append("No image received different verdicts from different model versions.")
        for row in changes:
            facts.append("Verdict changed for '" + row["fileName"] + "': model v" + row["versionA"]
                         + " said " + row["labelA"] + " (" + str(round(row["probA"] * 100))
                         + "% AI), model v" + row["versionB"] + " said " + row["labelB"]
                         + " (" + str(round(row["probB"] * 100)) + "% AI).")
    return facts


def statsFacts(stats):
    if stats["total"] == 0:
        return ["Total analyses: 0."]
    return ["Total analyses: " + str(stats["total"]) + " (" + str(stats["ai"])
            + " AI-generated, " + str(stats["real"]) + " Real).",
            "First analysis: " + str(stats["first"])[:16] + " UTC. Latest analysis: "
            + str(stats["last"])[:16] + " UTC.",
            "Average confidence: " + str(round(stats["avgConfidence"] * 100)) + "%."]


def pairFact(first, second):
    names = "#" + str(first.id) + " and #" + str(second.id)
    if first.image_sha256 == "" or first.image_sha256 != second.image_sha256:
        return names + " are different images (their fingerprints differ)."
    text = names + " are the same image file (identical fingerprint). "
    if first.model_name == second.model_name and first.model_version == second.model_version:
        return text + "Both used the same model version, so the same result is expected."
    text = text + ("They used different model versions (v" + first.model_version + " and v"
                   + second.model_version + "). ")
    if first.label == second.label:
        return text + "Both gave the verdict " + first.label + "."
    gap = round(abs(first.prob_ai - second.prob_ai) * 100)
    text = text + ("The verdicts differ: the probabilities of AI are " + percent(first.prob_ai)
                   + " and " + percent(second.prob_ai) + ", and the threshold is "
                   + percent(first.threshold) + ". ")
    if gap <= 10:
        return text + ("The scores are only " + str(gap) + " percentage points apart, so the "
                       "models agree closely and the label flipped because the score sits near "
                       "the threshold.")
    return text + ("The scores are " + str(gap) + " percentage points apart, which is a large "
                   "disagreement, so treat this image as an unclear case.")


def rowFact(row):
    gap = round(abs(row.prob_ai - row.threshold) * 100)
    side = "below"
    if row.prob_ai >= row.threshold:
        side = "above"
    closeness = "well away from the threshold"
    if gap <= 15:
        closeness = "close to the threshold, so this is an unclear case"
    return ("#" + str(row.id) + ": file '" + row.filename[:40] + "', model " + row.model_name
            + " v" + row.model_version + ", verdict " + row.label + ", probability of AI "
            + percent(row.prob_ai) + ", threshold " + percent(row.threshold)
            + ", image fingerprint " + row.image_sha256[:8] + ". The score is " + str(gap)
            + " percentage points " + side + " the threshold, " + closeness + ".")


def compareFacts(ids, rows):
    found = {}
    for row in rows:
        found[row.id] = row
    facts = []
    ordered = []
    for number in ids:
        if number in found:
            ordered.append(found[number])
        else:
            facts.append("Analysis #" + str(number) + " was not found in this user's history.")
    for row in ordered:
        facts.append(rowFact(row))
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            facts.append(pairFact(ordered[i], ordered[j]))
    return facts


def directAnswer(facts, closing=""):
    text = "Here is what I found in your account:"
    for fact in facts:
        text = text + "\n- " + fact
    if closing != "":
        text = text + "\n\n" + closing
    return text


def buildAgent(db, knowledge, graph, llm, minScore, maxPassages):
    """LangGraph workflow: greeting? then the chosen tools, a scope gate, and then either a
    direct answer from the facts, a refusal, or a language-model answer."""

    def routeStep(state):
        userId = state["userId"]
        return {"tools": pickTools(state["question"]), "done": [], "hits": [], "facts": [],
                "bestScore": 0.0, "kind": "answer", "greeting": isGreeting(state["question"]),
                "scans": db.recent(3, userId), "memory": db.recentChat(userId, 2)}

    def nextStep(state):
        if state["greeting"]:
            return "greet"
        for name in state["tools"]:
            if name not in state["done"]:
                return name
        return "gate"

    def searchDocs(state):
        question = state["question"]
        previous = previousQuestion(state["memory"])
        followUp = previous != "" and isFollowUp(question)
        ownScore = 0.0
        hits = []
        try:
            found = knowledge.search(question, 3)
            if len(found) > 0:
                ownScore = found[0]["score"]
            extra = 0
            if followUp:
                found = knowledge.search(previous + " " + question, 3)
                extra = 1
            hits = pickHits(found, minScore, maxPassages + extra)
        except Exception:
            log.exception("Knowledge search failed")
        return {"hits": hits, "bestScore": ownScore, "done": state["done"] + ["docs"]}

    def askCompare(state):
        ids = findAnalysisIds(state["question"])
        rows = db.predictionsByIds(state["userId"], ids)
        facts = (list(state["facts"]) + premiseNote(state["question"], rows)
                 + compareFacts(ids, rows))
        return {"facts": facts, "done": state["done"] + ["compare"]}

    def afterGate(state):
        if not state["onTopic"]:
            return "refuse"
        if len(state["facts"]) > 0 and ("compare" in state["tools"]
                                         or not wantsExplanation(state["question"])):
            return "direct"
        return "write"

    def gate(state):
        previous = previousQuestion(state["memory"])
        return {"onTopic": isOnTopic(state["question"], state.get("bestScore", 0.0),
                                     minScore, previous != "")}

    def greet(state):
        return {"answer": greetingText, "kind": "greeting", "tools": []}

    def refuse(state):
        return {"answer": refusalText, "kind": "refusal", "tools": ["scope"]}

    def askGraph(state):
        facts = list(state["facts"])
        if not getattr(graph, "enabled", True):
            facts.append("The graph lookup is switched off for this deployment.")
            return {"facts": facts, "done": state["done"] + ["graph"]}
        repeated = graph.repeatedImages(state["userId"])
        changes = graph.verdictChanges(state["userId"])
        versions = graph.versionsUsed(state["userId"])
        facts.extend(graphFacts(repeated, changes, versions))
        return {"facts": facts, "done": state["done"] + ["graph"]}

    def askStats(state):
        stats = db.userStats(state["userId"])
        return {"facts": list(state["facts"]) + statsFacts(stats),
                "done": state["done"] + ["stats"]}

    def direct(state):
        tools = []
        for name in state["tools"]:
            if name != "docs":
                tools.append(name)
        closing = ""
        if "compare" in state["tools"]:
            closing = trustText
        return {"answer": directAnswer(state["facts"], closing), "kind": "direct",
                "tools": tools, "hits": []}

    def writeAnswer(state):
        question = state["question"]
        followUp = isFollowUp(question) and previousQuestion(state["memory"]) != ""
        scans = None
        hasRecentScans = state.get("scans") is not None and len(state["scans"]) > 0
        if len(state["facts"]) > 0 or usesOwnData(question) or hasRecentScans:
            scans = state["scans"]
        memory = []
        if followUp:
            memory = state["memory"]
        messages = buildMessages(scans, memory, question, state["hits"], state["facts"], followUp)
        return {"answer": llm.chat(messages)}

    workflow = StateGraph(AgentState)
    workflow.add_node("routeStep", routeStep)
    workflow.add_node("searchDocs", searchDocs)
    workflow.add_node("askCompare", askCompare)
    workflow.add_node("askGraph", askGraph)
    workflow.add_node("askStats", askStats)
    workflow.add_node("gate", gate)
    workflow.add_node("greet", greet)
    workflow.add_node("refuse", refuse)
    workflow.add_node("direct", direct)
    workflow.add_node("writeAnswer", writeAnswer)
    workflow.add_edge(START, "routeStep")
    nodeNames = {"greet": "greet", "docs": "searchDocs", "compare": "askCompare",
                 "graph": "askGraph", "stats": "askStats", "gate": "gate"}
    for source in ["routeStep", "searchDocs", "askCompare", "askGraph", "askStats"]:
        workflow.add_conditional_edges(source, nextStep, nodeNames)
    workflow.add_conditional_edges("gate", afterGate, {"write": "writeAnswer", "refuse": "refuse",
                                                       "direct": "direct"})
    for finish in ["greet", "refuse", "direct", "writeAnswer"]:
        workflow.add_edge(finish, END)
    return workflow.compile()