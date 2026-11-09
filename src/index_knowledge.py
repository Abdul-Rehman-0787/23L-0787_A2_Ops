import time

from app.config import Config
from app.knowledge import KnowledgeBase

startTime = time.perf_counter()
base = KnowledgeBase(Config.CHROMA_PATH, Config.EMBEDDING_CACHE, Config.KNOWLEDGE_FOLDER)
count = base.rebuild()
print("indexed", count, "passages in", round(time.perf_counter() - startTime, 1), "seconds")