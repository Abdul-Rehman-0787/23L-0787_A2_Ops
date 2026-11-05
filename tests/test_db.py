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