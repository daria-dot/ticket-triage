"""Loads the registered baseline model once per process, pinned to an exact version."""

from functools import lru_cache

import mlflow.sklearn
from sklearn.pipeline import Pipeline

from triage.config import get_settings
from triage.models.train import REGISTERED_MODEL_NAME


@lru_cache
def get_model() -> Pipeline:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{settings.model_version}")
