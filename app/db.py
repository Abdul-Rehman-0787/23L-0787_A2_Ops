import os
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcNow():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcNow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)


class AuthEvent(Base):
    __tablename__ = "auth_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    event: Mapped[str] = mapped_column(String(40))
    ip: Mapped[str] = mapped_column(String(45))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcNow)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcNow)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64), default="")
    label: Mapped[str] = mapped_column(String(20))          # what was decided
    confidence: Mapped[float] = mapped_column(Float)        # how confidently
    prob_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(80))     # by which model
    model_version: Mapped[str] = mapped_column(String(40))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    file_size: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)


def runMigrations(url):
    """Bring the database up to date with the files in app/migrations."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    alembicConfig = AlembicConfig()
    alembicConfig.set_main_option("script_location",
                                  os.path.join(os.path.dirname(__file__), "migrations"))
    alembicConfig.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.upgrade(alembicConfig, "head")


class Database:
    def __init__(self, url):
        if url.startswith("sqlite:///"):
            folder = os.path.dirname(url.replace("sqlite:///", "", 1))
            if folder:
                os.makedirs(folder, exist_ok=True)
        runMigrations(url)
        args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=args)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def add_prediction(self, **fields):
        with self.Session() as s:
            s.add(Prediction(**fields))
            s.commit()

    def recent(self, limit=100, userId=None):
        with self.Session() as s:
            query = select(Prediction).order_by(Prediction.id.desc()).limit(limit)
            if userId is not None:
                query = query.where(Prediction.user_id == userId)
            return list(s.scalars(query))

    def createUser(self, email, passwordHash):
        """Returns the new user, or None if the email is already registered."""
        with self.Session() as s:
            user = User(email=email, password_hash=passwordHash)
            s.add(user)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                return None
            return user

    def findUserByEmail(self, email):
        with self.Session() as s:
            return s.scalar(select(User).where(User.email == email))

    def findUserById(self, userId):
        with self.Session() as s:
            return s.get(User, userId)

    def updatePassword(self, userId, passwordHash):
        with self.Session() as s:
            user = s.get(User, userId)
            if user is not None:
                user.password_hash = passwordHash
                s.commit()

    def recordLogin(self, userId):
        with self.Session() as s:
            user = s.get(User, userId)
            if user is not None:
                user.last_login_at = utcNow()
                user.is_verified = True
                s.commit()

    def addEvent(self, event, userId=None, ip=""):
        with self.Session() as s:
            s.add(AuthEvent(event=event, user_id=userId, ip=ip[:45]))
            s.commit()

    def ping(self):
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False