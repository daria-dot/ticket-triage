"""Exercises SageMaker's serving contract against a stand-in model.

A throwaway pipeline rather than the real registered one, so this runs in CI
with no MLflow server and no trained artifact -- what's under test is the
contract (/ping, /invocations, the response shape), not the model's quality.
"""

import mlflow.sklearn
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from triage.features.dataset import LABEL_COLUMNS

CATEGORIES = [label.removeprefix("is_") for label in LABEL_COLUMNS]


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", OneVsRestClassifier(LogisticRegression())),
        ]
    )
    texts = ["crash on startup", "please add dark mode", "typo in the readme"]
    labels = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
    pipeline.fit(texts, labels)

    model_dir = tmp_path_factory.mktemp("model") / "pipeline"
    mlflow.sklearn.save_model(pipeline, str(model_dir))

    from triage.api import sagemaker

    sagemaker.MODEL_DIR = str(model_dir)
    sagemaker.get_model.cache_clear()

    with TestClient(sagemaker.app) as test_client:
        yield test_client


def test_ping_reports_healthy(client: TestClient):
    response = client.get("/ping")
    assert response.status_code == 200


def test_invocations_returns_a_probability_per_category(client: TestClient):
    response = client.post("/invocations", json={"title": "crash on startup"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(CATEGORIES)
    assert all(0.0 <= score <= 1.0 for score in body.values())


def test_invocations_rejects_an_empty_title(client: TestClient):
    assert client.post("/invocations", json={"title": ""}).status_code == 422
