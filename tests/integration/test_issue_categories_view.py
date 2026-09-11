"""Validates the issue_categories view's label-mapping and PR-exclusion logic
against seeded synthetic data -- this is the "data validation" CI check, since
re-ingesting the real ~467k-row dataset isn't practical on every CI run.
"""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
SCHEMA_SQL = (SQL_DIR / "schema.sql").read_text()
VIEWS_SQL = "\n".join(
    (SQL_DIR / "views" / name).read_text()
    for name in ("issues_only.sql", "label_category_map.sql", "issue_categories.sql")
)


@pytest.fixture
def engine(pg_engine: Engine) -> Generator[Engine]:
    with pg_engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.execute(text(VIEWS_SQL))
        conn.execute(text("TRUNCATE raw_issues"))
    yield pg_engine
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE raw_issues"))


def _insert_issue(engine: Engine, repo: str, issue_number: int, payload: dict) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO raw_issues (repo, issue_number, payload) "
                "VALUES (:repo, :issue_number, :payload)"
            ),
            {"repo": repo, "issue_number": issue_number, "payload": json.dumps(payload)},
        )


def test_mapped_label_sets_its_category_true(engine: Engine):
    _insert_issue(
        engine,
        "huggingface/transformers",
        1,
        {"title": "Crash", "body": "trace", "labels": [{"name": "bug"}]},
    )

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT is_bug, is_feature, is_docs, is_question, is_duplicate "
                "FROM issue_categories WHERE issue_number = 1"
            )
        ).one()

    assert row == (True, False, False, False, False)


def test_unlabeled_issue_is_an_all_false_row_not_dropped(engine: Engine):
    _insert_issue(
        engine,
        "pandas-dev/pandas",
        2,
        {"title": "No labels here", "body": "", "labels": []},
    )

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT is_bug, is_feature, is_docs, is_question, is_duplicate "
                "FROM issue_categories WHERE issue_number = 2"
            )
        ).one()

    assert row == (False, False, False, False, False)


def test_pull_requests_are_excluded(engine: Engine):
    _insert_issue(
        engine,
        "pandas-dev/pandas",
        3,
        {"title": "A PR", "body": "", "labels": [], "pull_request": {"url": "..."}},
    )

    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT count(*) FROM issue_categories WHERE issue_number = 3")
        ).scalar()

    assert count == 0


def test_unmapped_label_does_not_set_any_category(engine: Engine):
    _insert_issue(
        engine,
        "pandas-dev/pandas",
        4,
        {"title": "Has a component tag", "body": "", "labels": [{"name": "Groupby"}]},
    )

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT is_bug, is_feature, is_docs, is_question, is_duplicate "
                "FROM issue_categories WHERE issue_number = 4"
            )
        ).one()

    assert row == (False, False, False, False, False)
