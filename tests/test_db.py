from sqlalchemy import inspect

from app.db import Database


def makeDb(tmpPath):
    return Database("sqlite:///" + str(tmpPath / "test.db").replace("\\", "/"))


def testMigrationCreatesTables(tmp_path):
    names = inspect(makeDb(tmp_path).engine).get_table_names()
    for table in ["users", "auth_events", "predictions", "chat_messages", "alembic_version"]:
        assert table in names


def testUsers(tmp_path):
    db = makeDb(tmp_path)
    user = db.createUser("a@b.com", "hash")
    assert user is not None
    assert db.createUser("a@b.com", "other") is None
    assert db.findUserByEmail("a@b.com").is_verified is False
    db.recordLogin(user.id)
    assert db.findUserById(user.id).is_verified is True
    assert db.findUserByEmail("nobody@b.com") is None

def testChatSourcesColumn(tmp_path):
    columns = []
    for column in inspect(makeDb(tmp_path).engine).get_columns("chat_messages"):
        columns.append(column["name"])
    assert "sources" in columns

def testUserStats(tmp_path):
    db = makeDb(tmp_path)
    user = db.createUser("a@b.com", "hash")
    for label, confidence in [("AI-generated", 0.8), ("Real", 0.6)]:
        db.add_prediction(user_id=user.id, filename="f.png", image_sha256="x" * 64, label=label,
                          confidence=confidence, prob_ai=0.5, threshold=0.48,
                          model_name="visionguard", model_version="2", width=1, height=1,
                          file_size=1, latency_ms=1.0)
    stats = db.userStats(user.id)
    assert stats["total"] == 2 and stats["ai"] == 1 and stats["real"] == 1
    assert round(stats["avgConfidence"], 2) == 0.7
    assert db.userStats(user.id + 1)["total"] == 0


def testPredictionsByIdsOnlyReturnsOwnRows(tmp_path):
    db = makeDb(tmp_path)
    first = db.createUser("a@b.com", "hash")
    second = db.createUser("c@d.com", "hash")
    for owner in [first, second]:
        db.add_prediction(user_id=owner.id, filename="f.png", image_sha256="x" * 64, label="Real",
                          confidence=0.9, prob_ai=0.1, threshold=0.48, model_name="visionguard",
                          model_version="2", width=1, height=1, file_size=1, latency_ms=1.0)
    rows = db.predictionsByIds(first.id, [1, 2])
    assert [row.user_id for row in rows] == [first.id]
    assert db.predictionsByIds(first.id, []) == []