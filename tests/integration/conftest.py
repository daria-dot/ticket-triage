"""Shared Postgres test-database plumbing.

Runs against a dedicated `triage_test` database on the local Postgres server
started by `make up` -- never the `triage` dev database.
"""

import os

import psycopg
import pytest
from sqlalchemy import Engine, create_engine

TEST_DB_NAME = os.environ.get("TEST_POSTGRES_DB", "triage_test")
ADMIN_DATABASE_URL = os.environ.get(
    "ADMIN_DATABASE_URL", "postgresql://triage:changeme@localhost:5432/postgres"
)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    f"postgresql+psycopg://triage:changeme@localhost:5432/{TEST_DB_NAME}",
)

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
def pg_engine() -> Engine:
    try:
        _ensure_test_database()
    except Exception as exc:
        pytest.skip(f"Postgres not reachable at {ADMIN_DATABASE_URL}: {exc}")
    return create_engine(TEST_DATABASE_URL)
