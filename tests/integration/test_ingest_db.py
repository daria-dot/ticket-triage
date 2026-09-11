"""Runs against a dedicated `triage_test` database on the local Postgres
server started by `make up` -- never the `triage` dev database, since this
suite truncates raw_issues and must not be able to touch real ingested data.
"""

import os
from collections.abc import Generator
from pathlib import Path

import psycopg
import pytest
from sqlalchemy import Engine, create_engine, text

from triage.ingest.db import last_seen_updated_at, upsert_issues

TEST_DB_NAME = os.environ.get("TEST_POSTGRES_DB", "triage_test")
ADMIN_DATABASE_URL = os.environ.get(
    "ADMIN_DATABASE_URL", "postgresql://triage:changeme@localhost:5432/postgres"
)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    f"postgresql+psycopg://triage:changeme@localhost:5432/{TEST_DB_NAME}",
)
SCHEMA_SQL = (Path(__file__).resolve().parents[2] / "sql" / "schema.sql").read_text()

if TEST_DB_NAME == "triage":
    raise RuntimeError("refusing to run integration tests against the triage dev database")


def _ensure_test_database() -> None:
    with psycopg.connect(ADMIN_DATABASE_URL, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB_NAME,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')


@pytest.fixture
def engine() -> Generator[Engine]:
    try:
        _ensure_test_database()
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {ADMIN_DATABASE_URL}: {exc}")

    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.execute(text("TRUNCATE raw_issues"))
    yield engine
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE raw_issues"))


def test_upsert_is_idempotent_on_repo_and_issue_number(engine: Engine):
    issue = {"number": 42, "updated_at": "2024-01-01T00:00:00Z", "title": "first"}
    upsert_issues(engine, "owner/repo", [issue])
    upsert_issues(engine, "owner/repo", [{**issue, "title": "edited"}])

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT payload FROM raw_issues")).all()

    assert len(rows) == 1
    assert rows[0][0]["title"] == "edited"


def test_last_seen_updated_at_tracks_the_max(engine: Engine):
    upsert_issues(
        engine,
        "owner/repo",
        [
            {"number": 1, "updated_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "updated_at": "2024-06-01T00:00:00Z"},
        ],
    )

    assert last_seen_updated_at(engine, "owner/repo") == "2024-06-01T00:00:00Z"


def test_last_seen_updated_at_is_none_for_unknown_repo(engine: Engine):
    assert last_seen_updated_at(engine, "nobody/nothing") is None


def test_upsert_strips_embedded_nul_bytes(engine: Engine):
    # Postgres text/jsonb cannot store the NUL codepoint at all; some real issue bodies have one.
    issue = {"number": 1, "updated_at": "2024-01-01T00:00:00Z", "body": "corrupted\x00paste"}
    upsert_issues(engine, "owner/repo", [issue])

    with engine.connect() as conn:
        row = conn.execute(text("SELECT payload FROM raw_issues")).one()

    assert row[0]["body"] == "corruptedpaste"


def test_upsert_preserves_backslashes_next_to_nul(engine: Engine):
    # A naive replace on the *encoded* JSON text (rather than the raw string
    # value) can mistake an escaped backslash for part of a NUL escape sequence
    # and corrupt unrelated content, e.g. a pasted regex like re.sub(r'[^\-]').
    body = "corrupted\x00paste with re.sub(r'[^\\-a-z]', ' ', text)"
    issue = {"number": 1, "updated_at": "2024-01-01T00:00:00Z", "body": body}
    upsert_issues(engine, "owner/repo", [issue])

    with engine.connect() as conn:
        row = conn.execute(text("SELECT payload FROM raw_issues")).one()

    assert row[0]["body"] == "corruptedpaste with re.sub(r'[^\\-a-z]', ' ', text)"
