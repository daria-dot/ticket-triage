"""Hits the live API against the isolated triage_test database.

Env vars must be set before triage.api.main is first imported anywhere in the
test session -- its module-level Settings() and engine are built at import
time. Hermetic dummy values, not the developer's real .env, so this never
depends on local config.
"""

import os

os.environ["POSTGRES_DB"] = os.environ.get("TEST_POSTGRES_DB", "triage_test")
os.environ.setdefault("GITHUB_TOKEN", "unused-in-tests")
os.environ.setdefault("TARGET_REPOS", "owner/repo")
os.environ.setdefault("MODEL_VERSION", "unused-in-tests")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import Engine  # noqa: E402

from triage.api.main import app  # noqa: E402

client = TestClient(app)


def test_health_is_ok_against_a_reachable_database(pg_engine: Engine):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_an_empty_title():
    response = client.post("/predict", json={"title": "", "body": "test"})
    assert response.status_code == 422


def test_the_page_is_served_and_calls_the_same_endpoint():
    """The demo page ships with the package, so a built image serves it too --
    it is package data, which a wheel omits unless asked."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # It has no model of its own: whatever it draws came from /predict.
    assert '"/predict"' in response.text
