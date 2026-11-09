import os
import threading

maxChunkChars = 700
embeddingName = "BAAI/bge-small-en-v1.5"
collectionName = "docs"


def readSections(filePath):
    """Split a markdown file into its title and a list of (heading, text) sections."""
    with open(filePath, encoding="utf-8") as openFile:
        allLines = openFile.read().split("\n")
    title = os.path.basename(filePath)
    heading = "Overview"
    lines = []
    sections = []
    for line in allLines:
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((heading, "\n".join(lines).strip()))
            heading = line[3:].strip()
            lines = []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines).strip()))
    return title, sections


def splitText(text, limit):
    """Cut long text into pieces at paragraph breaks, each piece about `limit` characters."""
    pieces = []
    current = ""
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph == "":
            continue
        if current == "":
            current = paragraph
        elif len(current) + len(paragraph) + 2 > limit:
            pieces.append(current)
            current = paragraph
        else:
            current = current + "\n\n" + paragraph
    if current != "":
        pieces.append(current)
    return pieces


def makeChunks(folder):
    chunks = []
    for fileName in sorted(os.listdir(folder)):
        if not fileName.endswith(".md"):
            continue
        title, sections = readSections(os.path.join(folder, fileName))
        for heading, text in sections:
            number = 0
            for piece in splitText(text, maxChunkChars):
                chunks.append({"id": fileName + "#" + heading + "#" + str(number),
                               "source": fileName, "title": title,
                               "section": heading, "text": piece})
                number = number + 1
    return chunks


class KnowledgeBase:
    """Reads the documents, turns them into vectors and finds the closest passages."""

    def __init__(self, chromaPath, cacheFolder, docsFolder):
        self.chromaPath = chromaPath
        self.cacheFolder = cacheFolder
        self.docsFolder = docsFolder
        self.lock = threading.RLock()
        self.embedder = None
        self.client = None
        self.collection = None

    def getEmbedder(self):
        with self.lock:
            if self.embedder is None:
                from fastembed import TextEmbedding
                self.embedder = TextEmbedding(model_name=embeddingName, cache_dir=self.cacheFolder)
            return self.embedder

    def embed(self, texts):
        vectors = []
        for vector in self.getEmbedder().embed(texts):
            vectors.append(vector.tolist())
        return vectors

    def getClient(self):
        with self.lock:
            if self.client is None:
                import chromadb
                os.makedirs(self.chromaPath, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.chromaPath, settings=chromadb.Settings(anonymized_telemetry=False))
            return self.client

    def getCollection(self):
        with self.lock:
            if self.collection is None:
                self.collection = self.getClient().get_or_create_collection(
                    name=collectionName, metadata={"hnsw:space": "cosine"})
            return self.collection

    def rebuild(self):
        """Throw the old index away and index every document again. Returns the passage count."""
        chunks = makeChunks(self.docsFolder)
        with self.lock:
            client = self.getClient()
            try:
                client.get_collection(name=collectionName)
            except ValueError:
                pass                       # first run: there was nothing to delete
            else:
                client.delete_collection(collectionName)
            self.collection = None
            collection = self.getCollection()
            if len(chunks) == 0:
                return 0
            texts = []
            ids = []
            documents = []
            metadatas = []
            for chunk in chunks:
                texts.append(chunk["title"] + " - " + chunk["section"] + "\n" + chunk["text"])
                ids.append(chunk["id"])
                documents.append(chunk["text"])
                metadatas.append({"source": chunk["source"], "title": chunk["title"],
                                  "section": chunk["section"]})
            collection.add(ids=ids, documents=documents, embeddings=self.embed(texts),
                           metadatas=metadatas)
        return len(chunks)

    def search(self, question, count=3):
        """The closest passages, best first. Score is cosine similarity (1.0 = same meaning)."""
        collection = self.getCollection()
        if collection.count() == 0:
            self.rebuild()
            collection = self.getCollection()
        vector = self.embed([question])[0]
        result = collection.query(query_embeddings=[vector], n_results=count,
                                  include=["documents", "metadatas", "distances"])
        hits = []
        for position in range(len(result["ids"][0])):
            info = result["metadatas"][0][position]
            hits.append({"source": info["source"], "title": info["title"],
                         "section": info["section"], "text": result["documents"][0][position],
                         "score": round(1 - result["distances"][0][position], 3)})
        return hits