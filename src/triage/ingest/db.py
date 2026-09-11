"""Persistence for raw issue payloads. No transformation — see sql/schema.sql."""

import json
from typing import Any

from sqlalchemy import Engine, text


def last_seen_updated_at(engine: Engine, repo: str) -> str | None:
    """Most recent `updated_at` already stored for `repo`, or None if we have nothing yet."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT max(payload ->> 'updated_at') FROM raw_issues WHERE repo = :repo"),
            {"repo": repo},
        ).one()
    value: str | None = row[0]
    return value


def _strip_nul(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {key: _strip_nul(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_strip_nul(v) for v in value]
    return value


def _dump_payload(issue: dict[str, Any]) -> str:
    # Postgres's text type cannot store the NUL codepoint at all, in json/jsonb
    # or otherwise. A handful of issue bodies contain one (pasted binary /
    # corrupted content) -- strip it from the actual string values before
    # serializing, not from the encoded JSON text (a text-level replace risks
    # matching an escaped backslash that happens to precede similar digits,
    # corrupting otherwise-valid JSON).
    return json.dumps(_strip_nul(issue))


def upsert_issues(engine: Engine, repo: str, issues: list[dict[str, Any]]) -> None:
    if not issues:
        return
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO raw_issues (repo, issue_number, payload)
                VALUES (:repo, :issue_number, :payload)
                ON CONFLICT (repo, issue_number)
                DO UPDATE SET payload = EXCLUDED.payload, fetched_at = now()
                """
            ),
            [
                {"repo": repo, "issue_number": issue["number"], "payload": _dump_payload(issue)}
                for issue in issues
            ],
        )
