"""Hits the local Postgres started by `make up`. Requires DATABASE_URL (or the
docker-compose defaults) to point at a real, reachable instance.
"""

import os
from collections.abc import Generator

import pytest
from sqlalchemy import Engine, create_engine, text

from triage.ingest.db import last_seen_updated_at, upsert_issues

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://triage:changeme@localhost:5432/triage"
)


@pytest.fixture
def engine() -> Generator[Engine]:
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {TEST_DATABASE_URL}: {exc}")

    with engine.begin() as conn:
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
