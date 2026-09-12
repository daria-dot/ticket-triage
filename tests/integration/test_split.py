"""The hashed train/test split is what makes comparing model approaches honest,
so it needs to actually be stable and disjoint rather than merely look it.
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
        for number in range(400):
            conn.execute(
                text(
                    "INSERT INTO raw_issues (repo, issue_number, payload) "
                    "VALUES (:repo, :n, :payload)"
                ),
                {
                    "repo": "owner/repo",
                    "n": number,
                    "payload": json.dumps({"title": f"issue {number}", "labels": []}),
                },
            )
    yield pg_engine
    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE raw_issues"))


def test_every_issue_is_on_exactly_one_side(engine: Engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT split, count(*) FROM issue_categories GROUP BY split")
        ).all()

    counts = dict(rows)
    assert set(counts) == {"train", "test"}
    assert sum(counts.values()) == 400


def test_the_test_side_is_roughly_a_fifth(engine: Engine):
    with engine.connect() as conn:
        share = conn.execute(
            text("SELECT avg((split = 'test')::int) FROM issue_categories")
        ).scalar()

    # Hashing won't land on exactly 20% for a small sample; the point is that
    # it's near, not that it's exact.
    assert 0.12 < float(share) < 0.28


def test_an_issue_keeps_its_side_when_its_content_changes(engine: Engine):
    with engine.connect() as conn:
        before = conn.execute(
            text("SELECT split FROM issue_categories WHERE issue_number = 7")
        ).scalar()

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE raw_issues SET payload = :p WHERE issue_number = 7"),
            {"p": json.dumps({"title": "rewritten", "body": "new", "labels": []})},
        )

    with engine.connect() as conn:
        after = conn.execute(
            text("SELECT split FROM issue_categories WHERE issue_number = 7")
        ).scalar()

    assert before == after


def test_arrival_of_new_issues_does_not_move_existing_ones(engine: Engine):
    with engine.connect() as conn:
        before = dict(conn.execute(text("SELECT issue_number, split FROM issue_categories")).all())

    with engine.begin() as conn:
        for number in range(400, 500):
            conn.execute(
                text(
                    "INSERT INTO raw_issues (repo, issue_number, payload) "
                    "VALUES (:repo, :n, :payload)"
                ),
                {
                    "repo": "owner/repo",
                    "n": number,
                    "payload": json.dumps({"title": f"later {number}", "labels": []}),
                },
            )

    with engine.connect() as conn:
        after = dict(conn.execute(text("SELECT issue_number, split FROM issue_categories")).all())

    assert all(after[number] == side for number, side in before.items())
