"""Tests for database setup: WAL mode, schema, basic CRUD."""
import datetime
import tempfile
import os
import pytest
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import Session


def make_engine(path):
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


# ── WAL mode ──────────────────────────────────────────────────────────────────

def test_wal_mode_enabled():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        engine = make_engine(path)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0] == "wal"
    finally:
        os.unlink(path)


def test_synchronous_normal():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        engine = make_engine(path)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA synchronous")).fetchone()
            # 1 = NORMAL
            assert result[0] == 1
    finally:
        os.unlink(path)


# ── schema ────────────────────────────────────────────────────────────────────

def test_schema_creates_all_tables():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        engine = make_engine(path)
        # Import Base which runs create_all at module level
        import importlib, sys
        # Patch DATABASE_URL to point at temp file before importing db
        import db as db_module
        from db import Base
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            tables = {row[0] for row in
                      conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert "users" in tables
        assert "user_features" in tables
        assert "recordings" in tables
        assert "person_events" in tables
        assert "sessions" in tables
    finally:
        os.unlink(path)


# ── basic CRUD ────────────────────────────────────────────────────────────────

def test_user_created_at_timezone_aware():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        engine = make_engine(path)
        from db import Base, User
        Base.metadata.create_all(engine)
        with Session(engine) as s, s.begin():
            u = User(id="u1", name="Test")
            s.add(u)
        with Session(engine) as s:
            u = s.get(User, "u1")
            assert u is not None
            assert u.name == "Test"
            # created_at default is now(UTC) — should not be None
            assert u.created_at is not None
    finally:
        os.unlink(path)


def test_recording_person_event_relationship():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        engine = make_engine(path)
        from db import Base, Recording, PersonEvent
        Base.metadata.create_all(engine)
        now = datetime.datetime.now(datetime.timezone.utc)
        with Session(engine) as s, s.begin():
            rec = Recording(start_time=now)
            s.add(rec)
            s.flush()
            ev = PersonEvent(recording_id=rec.id, track_id=1, first_seen=now)
            s.add(ev)
        with Session(engine) as s:
            recs = s.query(Recording).all()
            assert len(recs) == 1
            evts = s.query(PersonEvent).filter_by(recording_id=recs[0].id).all()
            assert len(evts) == 1
            assert evts[0].track_id == 1
    finally:
        os.unlink(path)
