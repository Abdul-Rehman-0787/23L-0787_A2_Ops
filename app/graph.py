import logging

log = logging.getLogger("visionguard")

constraints = [
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT image_sha IF NOT EXISTS FOR (n:Image) REQUIRE n.sha256 IS UNIQUE",
    "CREATE CONSTRAINT analysis_id IF NOT EXISTS FOR (n:Analysis) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT model_key IF NOT EXISTS FOR (n:ModelVersion) REQUIRE n.key IS UNIQUE",
]

saveAnalysisQuery = """
MERGE (u:User {id: $userId})
MERGE (i:Image {sha256: $sha256})
MERGE (m:ModelVersion {key: $modelKey})
  ON CREATE SET m.name = $modelName, m.version = $modelVersion
MERGE (a:Analysis {id: $analysisId})
  ON CREATE SET a.label = $label, a.probAi = $probAi, a.threshold = $threshold,
                a.createdAt = $createdAt, a.filename = $filename
MERGE (u)-[:SUBMITTED]->(a)
MERGE (a)-[:OF]->(i)
MERGE (a)-[:USED]->(m)
"""

repeatedImagesQuery = """
MATCH (u:User {id: $userId})-[:SUBMITTED]->(a:Analysis)-[:OF]->(i:Image)
WITH i, count(a) AS times, collect(a.filename)[0] AS fileName, max(a.createdAt) AS lastSeen
WHERE times > 1
RETURN i.sha256 AS sha, fileName, times, lastSeen
ORDER BY times DESC, lastSeen DESC
LIMIT 5
"""

verdictChangesQuery = """
MATCH (u:User {id: $userId})-[:SUBMITTED]->(a1:Analysis)-[:OF]->(i:Image)
      <-[:OF]-(a2:Analysis)<-[:SUBMITTED]-(u)
MATCH (a1)-[:USED]->(m1:ModelVersion), (a2)-[:USED]->(m2:ModelVersion)
WHERE a1.id < a2.id AND m1.key <> m2.key AND a1.label <> a2.label
RETURN i.sha256 AS sha, a1.filename AS fileName,
       m1.version AS versionA, a1.label AS labelA, a1.probAi AS probA,
       m2.version AS versionB, a2.label AS labelB, a2.probAi AS probB
ORDER BY a2.createdAt DESC
LIMIT 5
"""

versionsUsedQuery = """
MATCH (u:User {id: $userId})-[:SUBMITTED]->(a:Analysis)-[:USED]->(m:ModelVersion)
RETURN m.name AS name, m.version AS version, count(a) AS analyses,
       sum(CASE WHEN a.label = 'AI-generated' THEN 1 ELSE 0 END) AS aiCount
ORDER BY version
"""

clearQuery = "MATCH (n) WHERE n:User OR n:Image OR n:Analysis OR n:ModelVersion DETACH DELETE n"


class GraphStore:
    """The lineage map in Neo4j. Reading fails softly, so the app works without it."""

    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.enabled = password != ""

    def getDriver(self):
        if self.driver is None:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self.driver

    def run(self, query, params=None):
        """Runs one query and returns a list of dictionaries. Raises if Neo4j is unreachable."""
        records, _, _ = self.getDriver().execute_query(
            query, parameters_=params or {}, database_="neo4j")
        rows = []
        for record in records:
            rows.append(record.data())
        return rows

    def softRun(self, query, params):
        if not self.enabled:
            return []
        try:
            return self.run(query, params)
        except Exception:
            log.warning("Graph query failed", exc_info=True)
            return []

    def ping(self):
        from neo4j.exceptions import AuthError, ServiceUnavailable
        try:
            self.run("RETURN 1 AS ok")
            return True
        except (ServiceUnavailable, AuthError, OSError):
            return False

    def setup(self):
        for statement in constraints:
            self.run(statement)

    def recordAnalysis(self, analysis):
        """Adds one analysis. Safe to repeat: the same analysis is never added twice."""
        params = dict(analysis)
        params["modelKey"] = analysis["modelName"] + ":" + analysis["modelVersion"]
        self.run(saveAnalysisQuery, params)

    def repeatedImages(self, userId):
        return self.softRun(repeatedImagesQuery, {"userId": userId})

    def verdictChanges(self, userId):
        return self.softRun(verdictChangesQuery, {"userId": userId})

    def versionsUsed(self, userId):
        return self.softRun(versionsUsedQuery, {"userId": userId})

    def clearAll(self):
        self.run(clearQuery)

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None