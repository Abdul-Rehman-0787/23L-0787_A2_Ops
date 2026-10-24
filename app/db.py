import os
from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255))
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


class Database:
    def __init__(self, url):
        if url.startswith("sqlite:///"):
            folder = os.path.dirname(url.replace("sqlite:///", "", 1))
            if folder:
                os.makedirs(folder, exist_ok=True)
        args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=args)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def add_prediction(self, **fields):
        with self.Session() as s:
            s.add(Prediction(**fields))
            s.commit()

    def recent(self, limit=100):
        with self.Session() as s:
            return list(s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))

    def ping(self):
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False