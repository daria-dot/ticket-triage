"""Runs against a dedicated `triage_test` database on the local Postgres
server started by `make up` -- never the `triage` dev database, since this
suite truncates raw_issues and must not be able to touch real ingested data.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from triage.ingest.db import last_seen_updated_at, upsert_issues

SCHEMA_SQL = (Path(__file__).resolve().parents[2] / "sql" / "schema.sql").read_text()


@pytest.fixture
def engine(pg_engine: Engine) -> Generator[Engine]:
    with pg_engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.execute(text("TRUNCATE raw_issues"))
    yield pg_engine
    with pg_engine.begin() as conn:
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
