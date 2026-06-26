from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, DateTime, LargeBinary, Integer, ForeignKey, Text, event
from sqlalchemy.orm import DeclarativeBase, relationship, Session
import datetime
import numpy as np


DATABASE_URL = "sqlite:///people.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id         = Column(String, primary_key=True)
    name       = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    features   = relationship("UserFeature", back_populates="user", cascade="all, delete-orphan")
    sessions   = relationship("UserSession", back_populates="user")


class UserFeature(Base):
    __tablename__ = "user_features"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    user_id   = Column(String, ForeignKey("users.id"))
    embedding = Column(LargeBinary, nullable=False)  # float32 numpy array bytes
    user      = relationship("User", back_populates="features")


class UserSession(Base):
    __tablename__ = "sessions"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(String, ForeignKey("users.id"), nullable=True)
    track_id   = Column(Integer)
    start_time = Column(DateTime)
    end_time   = Column(DateTime, nullable=True)
    video_path = Column(Text, nullable=True)
    user       = relationship("User", back_populates="sessions")


class Recording(Base):
    """One continuous video file covering a scene (one or many people)."""
    __tablename__ = "recordings"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, nullable=False)
    end_time   = Column(DateTime, nullable=True)
    video_path = Column(Text, nullable=True)
    events     = relationship("PersonEvent", back_populates="recording",
                              cascade="all, delete-orphan")


class PersonEvent(Base):
    """One person detected within a Recording."""
    __tablename__ = "person_events"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    recording_id  = Column(Integer, ForeignKey("recordings.id"), nullable=False)
    user_id       = Column(String, ForeignKey("users.id"), nullable=True)
    track_id      = Column(Integer, nullable=False)
    first_seen    = Column(DateTime, nullable=False)
    last_seen     = Column(DateTime, nullable=True)
    snapshot_path = Column(Text, nullable=True)
    recording     = relationship("Recording", back_populates="events")


Base.metadata.create_all(engine)


@contextmanager
def get_session():
    with Session(engine) as session:
        with session.begin():
            yield session


def emb_to_bytes(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def bytes_to_emb(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()
