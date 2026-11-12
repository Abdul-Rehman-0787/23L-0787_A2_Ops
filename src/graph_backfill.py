import sys

from sqlalchemy import select

from app.config import Config
from app.db import Database, Prediction
from app.graph import GraphStore

store = GraphStore(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
if not store.enabled:
    print("NEO4J_PASSWORD is empty in .env, so the graph is switched off.")
    sys.exit(1)
if not store.ping():
    print("Cannot reach Neo4j at", Config.NEO4J_URI, "- is the container running?")
    sys.exit(1)

store.setup()
if "--reset" in sys.argv:
    store.clearAll()
    print("graph cleared")

db = Database(Config.DATABASE_URL)
with db.Session() as session:
    rows = list(session.scalars(select(Prediction).order_by(Prediction.id)))

count = 0
for row in rows:
    if row.user_id is None or row.image_sha256 == "":
        continue                      # old results from before accounts had no owner or hash
    store.recordAnalysis({
        "analysisId": row.id, "userId": row.user_id, "sha256": row.image_sha256,
        "filename": row.filename[:40], "label": row.label, "probAi": row.prob_ai,
        "threshold": row.threshold, "createdAt": row.created_at.isoformat(),
        "modelName": row.model_name, "modelVersion": row.model_version})
    count = count + 1
print("copied", count, "analyses into the graph")