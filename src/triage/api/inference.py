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
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
def thresholds() -> dict[str, float]:
    """The decision thresholds fitted for the pinned model version.

    Read from the registry rather than configured here, because a threshold is
    only meaningful against the probability distribution it was fitted on --
    pairing one model's cut with another's scores is silently wrong in a way no
    error would reveal. Tying them to the version means MODEL_VERSION alone
    describes what the API is doing.

    Raises if the version has none. A default of 0.5 is exactly the bug this
    replaced, and falling back to it would reintroduce it invisibly.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    version = MlflowClient().get_model_version(REGISTERED_MODEL_NAME, settings.model_version)
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="thresholds.json"
        )
    except OSError as exc:
        raise RuntimeError(
            f"{REGISTERED_MODEL_NAME} version {settings.model_version} has no fitted "
            "thresholds. Serve a version registered with them rather than assuming 0.5."
        ) from exc

    with open(path) as handle:
        loaded: dict[str, float] = json.load(handle)

    missing = set(CATEGORIES) - set(loaded)
    if missing:
        raise RuntimeError(f"thresholds are missing categories: {sorted(missing)}")
    return loaded


def decide(probabilities: dict[str, float]) -> list[str]:
    """Categories whose probability clears their own threshold.

    Applied here rather than in either backend so the decision is identical
    whether the model ran in this process or in SageMaker -- the same reason
    logging lives in the API.
    """
    cuts = thresholds()
    return [category for category in CATEGORIES if probabilities[category] >= cuts[category]]


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
