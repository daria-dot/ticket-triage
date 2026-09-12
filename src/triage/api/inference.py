"""Where inference happens, and nothing else.

The API owns prediction logging because the SageMaker endpoint cannot: a model
server has no database to write to, and giving it one would put application
concerns inside the thing whose only job is to turn text into numbers. Routing
inference through here means the log covers both paths identically -- the only
difference is whether the model runs in this process or in AWS.

There is deliberately no fallback between the two. If the endpoint is
unreachable, that surfaces as an error rather than quietly serving predictions
from a different model than the caller believes they are getting.
"""

import json
from functools import lru_cache

import boto3
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from triage.config import get_settings
from triage.features.dataset import LABEL_COLUMNS, combine_title_body
from triage.models.train import REGISTERED_MODEL_NAME

CATEGORIES = [label.removeprefix("is_") for label in LABEL_COLUMNS]


@lru_cache
def _local_model() -> Pipeline:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{settings.model_version}")


@lru_cache
def _runtime_client():  # type: ignore[no-untyped-def]
    return boto3.client("sagemaker-runtime", region_name=get_settings().aws_region)


def _via_sagemaker(title: str, body: str | None) -> dict[str, float]:
    settings = get_settings()
    response = _runtime_client().invoke_endpoint(
        EndpointName=settings.sagemaker_endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"title": title, "body": body}),
    )
    scores: dict[str, float] = json.loads(response["Body"].read())
    return scores


def _in_process(title: str, body: str | None) -> dict[str, float]:
    probabilities = _local_model().predict_proba([combine_title_body(title, body)])[0]
    return {category: float(p) for category, p in zip(CATEGORIES, probabilities, strict=True)}


def predict_probabilities(title: str, body: str | None) -> dict[str, float]:
    if get_settings().inference_backend == "sagemaker":
        return _via_sagemaker(title, body)
    return _in_process(title, body)
