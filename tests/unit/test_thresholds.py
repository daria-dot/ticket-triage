"""Per-label thresholds are the operating point, so applying the wrong one --
or quietly falling back to 0.5 -- changes what the service decides without
changing anything a caller could see.
"""

import pytest

from triage.api import inference
from triage.config import get_settings

FITTED = {
    "bug": 0.553,
    "feature": 0.647,
    "docs": 0.947,
    "question": 0.841,
    "duplicate": 0.584,
}


@pytest.fixture(autouse=True)
def _clear_caches():
    get_settings.cache_clear()
    inference.thresholds.cache_clear()
    yield
    get_settings.cache_clear()
    inference.thresholds.cache_clear()


def test_each_label_is_judged_against_its_own_threshold(monkeypatch):
    monkeypatch.setattr(inference, "thresholds", lambda: FITTED)

    # 0.8 clears bug and feature, but not docs or question -- under a single
    # 0.5 cut every one of these would have fired.
    probabilities = {
        "bug": 0.80,
        "feature": 0.80,
        "docs": 0.80,
        "question": 0.80,
        "duplicate": 0.55,
    }

    assert inference.decide(probabilities) == ["bug", "feature"]


def test_a_probability_exactly_on_its_threshold_counts(monkeypatch):
    monkeypatch.setattr(inference, "thresholds", lambda: FITTED)

    probabilities = dict.fromkeys(inference.CATEGORIES, 0.0) | {"docs": FITTED["docs"]}

    assert inference.decide(probabilities) == ["docs"]


def test_a_model_version_without_thresholds_fails_loudly(monkeypatch):
    """The default this replaced was 0.5. Falling back to it would restore the
    bug silently, which is worse than refusing to serve."""

    class _Version:
        run_id = "whatever"

    monkeypatch.setattr(
        inference,
        "MlflowClient",
        lambda: type("_C", (), {"get_model_version": staticmethod(lambda *a: _Version())})(),
    )

    def _missing(**kwargs):
        raise OSError("no such artifact")

    monkeypatch.setattr(inference.mlflow.artifacts, "download_artifacts", _missing)

    with pytest.raises(RuntimeError, match="no fitted thresholds"):
        inference.thresholds()


def test_incomplete_thresholds_are_rejected(monkeypatch, tmp_path):
    """A file covering four of five labels would otherwise KeyError deep inside
    a request, rather than at the point the mistake was made."""
    import json

    partial = tmp_path / "thresholds.json"
    partial.write_text(json.dumps({"bug": 0.5, "feature": 0.5}))

    class _Version:
        run_id = "whatever"

    monkeypatch.setattr(
        inference,
        "MlflowClient",
        lambda: type("_C", (), {"get_model_version": staticmethod(lambda *a: _Version())})(),
    )
    monkeypatch.setattr(inference.mlflow.artifacts, "download_artifacts", lambda **k: str(partial))

    with pytest.raises(RuntimeError, match="missing categories"):
        inference.thresholds()
