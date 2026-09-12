"""The backend switch decides whether a prediction comes from this process or
from AWS, so it needs to actually switch -- and to fail loudly rather than
quietly serving a different model than the caller thinks they asked for.
"""

import json
from io import BytesIO

import pytest

from triage.api import inference
from triage.config import get_settings


@pytest.fixture(autouse=True)
def _clear_cached_settings():
    get_settings.cache_clear()
    inference._runtime_client.cache_clear()
    yield
    get_settings.cache_clear()
    inference._runtime_client.cache_clear()


class _FakeRuntime:
    def __init__(self, payload: dict[str, float]):
        self.payload = payload
        self.calls: list[dict] = []

    def invoke_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        return {"Body": BytesIO(json.dumps(self.payload).encode())}


def test_sagemaker_backend_calls_the_endpoint(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "sagemaker")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "some-endpoint")
    get_settings.cache_clear()

    fake = _FakeRuntime(
        {"bug": 0.9, "feature": 0.1, "docs": 0.0, "question": 0.0, "duplicate": 0.0}
    )
    monkeypatch.setattr(inference, "_runtime_client", lambda: fake)

    scores = inference.predict_probabilities("Crash on launch", "stack trace")

    assert scores["bug"] == 0.9
    assert len(fake.calls) == 1
    assert fake.calls[0]["EndpointName"] == "some-endpoint"
    assert json.loads(fake.calls[0]["Body"]) == {"title": "Crash on launch", "body": "stack trace"}


def test_sagemaker_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "sagemaker")
    get_settings.cache_clear()

    class _Broken:
        def invoke_endpoint(self, **kwargs):
            raise RuntimeError("endpoint unreachable")

    monkeypatch.setattr(inference, "_runtime_client", lambda: _Broken())

    # No silent fallback to the in-process model: serving predictions from a
    # different model than the caller believes is worse than failing.
    with pytest.raises(RuntimeError, match="endpoint unreachable"):
        inference.predict_probabilities("Crash on launch", None)


def test_local_backend_does_not_touch_aws(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "local")
    get_settings.cache_clear()

    def _fail():
        raise AssertionError("local backend must not construct an AWS client")

    monkeypatch.setattr(inference, "_runtime_client", _fail)

    class _Pipeline:
        def predict_proba(self, texts):
            assert texts == ["Crash on launch"]
            return [[0.7, 0.1, 0.05, 0.1, 0.05]]

    monkeypatch.setattr(inference, "_local_model", lambda: _Pipeline())

    scores = inference.predict_probabilities("Crash on launch", None)
    assert set(scores) == set(inference.CATEGORIES)
    assert scores["bug"] == 0.7
