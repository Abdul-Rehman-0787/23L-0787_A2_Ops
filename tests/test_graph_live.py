import pytest

from app.config import Config
from app.graph import GraphStore

testUser = 987654321


def makeStore():
    store = GraphStore(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    if not store.enabled or not store.ping():
        pytest.skip("Neo4j is not running")
    return store


def makeAnalysis(analysisId, sha, label, probAi, version):
    return {"analysisId": analysisId, "userId": testUser, "sha256": sha, "filename": "pic.png",
            "label": label, "probAi": probAi, "threshold": 0.48,
            "createdAt": "2026-11-10T10:00:00", "modelName": "testmodel", "modelVersion": version}


def cleanUp(store):
    store.run("MATCH (a:Analysis) WHERE a.id >= 9000000 DETACH DELETE a")
    store.run("MATCH (i:Image) WHERE i.sha256 STARTS WITH 'test-' DETACH DELETE i")
    store.run("MATCH (u:User {id: $userId}) DETACH DELETE u", {"userId": testUser})
    store.run("MATCH (m:ModelVersion) WHERE m.name = 'testmodel' DETACH DELETE m")


def testLineageQueries():
    store = makeStore()
    cleanUp(store)
    try:
        store.setup()
        store.recordAnalysis(makeAnalysis(9000001, "test-aaa", "Real", 0.30, "2"))
        store.recordAnalysis(makeAnalysis(9000001, "test-aaa", "Real", 0.30, "2"))   # repeat is harmless
        store.recordAnalysis(makeAnalysis(9000002, "test-aaa", "AI-generated", 0.55, "3"))
        store.recordAnalysis(makeAnalysis(9000003, "test-bbb", "Real", 0.10, "2"))

        repeated = store.repeatedImages(testUser)
        assert len(repeated) == 1
        assert repeated[0]["sha"] == "test-aaa"
        assert repeated[0]["times"] == 2

        changes = store.verdictChanges(testUser)
        assert len(changes) == 1
        assert changes[0]["versionA"] == "2" and changes[0]["labelA"] == "Real"
        assert changes[0]["versionB"] == "3" and changes[0]["labelB"] == "AI-generated"

        versions = {}
        for row in store.versionsUsed(testUser):
            versions[row["version"]] = (row["analyses"], row["aiCount"])
        assert versions == {"2": (2, 0), "3": (1, 1)}

        assert store.repeatedImages(testUser + 1) == []          # other users see nothing
        assert store.verdictChanges(testUser + 1) == []
    finally:
        cleanUp(store)