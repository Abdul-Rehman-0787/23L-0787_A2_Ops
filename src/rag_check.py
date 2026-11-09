import json

from app.config import Config
from app.knowledge import KnowledgeBase

base = KnowledgeBase(Config.CHROMA_PATH, Config.EMBEDDING_CACHE, Config.KNOWLEDGE_FOLDER)
with open("tests/golden_questions.json", encoding="utf-8") as openFile:
    questions = json.load(openFile)

onTopicTotal = 0
onTopicFound = 0
offTopicTotal = 0
offTopicRefused = 0

for item in questions:
    hits = base.search(item["question"], 3)
    topScore = 0.0
    sourcesFound = []
    if len(hits) > 0:
        topScore = hits[0]["score"]
    for hit in hits:
        sourcesFound.append(hit["source"])

    if len(item["expected"]) == 0:
        offTopicTotal = offTopicTotal + 1
        passed = topScore < Config.RAG_MIN_SIMILARITY
        if passed:
            offTopicRefused = offTopicRefused + 1
    else:
        onTopicTotal = onTopicTotal + 1
        passed = False
        for name in item["expected"]:
            if name in sourcesFound:
                passed = True
        if passed:
            onTopicFound = onTopicFound + 1

    mark = "OK  "
    if not passed:
        mark = "MISS"
    print(mark, "top score", round(topScore, 2), "|", item["question"], "->", sourcesFound)

print()
print("right document in top 3:", onTopicFound, "of", onTopicTotal,
      "(", round(100 * onTopicFound / onTopicTotal), "% )")
print("off-topic questions below the", Config.RAG_MIN_SIMILARITY, "cutoff:",
      offTopicRefused, "of", offTopicTotal)