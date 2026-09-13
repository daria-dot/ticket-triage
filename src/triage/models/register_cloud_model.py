"""Registers a pipeline trained by a SageMaker job into the MLflow registry.

Training in the cloud and serving from MLflow leaves a gap: the job writes a
tarball to S3, and nothing in the registry knows it exists. This closes it, so
what gets served is reachable by version number like every other model rather
than by remembering an S3 key.

The thresholds travel with the model deliberately. They were fitted against
this pipeline's probabilities on this split; pairing them with any other model
would be quietly wrong, since a threshold means nothing apart from the
distribution it was fitted on. Registering them together is what makes
`MODEL_VERSION` sufficient to describe what the API is doing.

Loading the pipeline here rather than passing the tarball through also forces
the artifact to prove it survives the version gap between the training image
and this environment, before anything depends on it.
"""

import argparse
import json
import tarfile
import tempfile
import warnings
from pathlib import Path

import boto3
import joblib
import mlflow
import mlflow.sklearn

from triage.config import get_settings

EXPERIMENT_NAME = "ticket-triage-baseline"
REGISTERED_MODEL_NAME = "ticket-triage-baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", required=True, help="s3:// URI of the job's model.tar.gz")
    args = parser.parse_args()

    settings = get_settings()
    bucket, _, key = args.model_data.removeprefix("s3://").partition("/")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.gz"
        boto3.client("s3", region_name=settings.aws_region).download_file(bucket, key, str(archive))
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")

        metrics = json.loads((Path(tmp) / "metrics.json").read_text())
        thresholds = json.loads((Path(tmp) / "thresholds.json").read_text())

        # The training image and this environment pin different scikit-learn
        # versions, so unpickling warns. It is surfaced rather than silenced:
        # the metrics were reproduced under the serving version before this was
        # relied on, and if that ever stops holding the warning is the clue.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = joblib.load(Path(tmp) / "pipeline.joblib")
        for warning in caught:
            print(f"{type(warning.message).__name__}: {warning.message}")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("trained_on", "sagemaker")
        mlflow.log_param("model_data", args.model_data)
        for name, value in metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(name, value)
            else:
                mlflow.log_param(name, value)

        mlflow.log_dict(thresholds, "thresholds.json")
        result = mlflow.sklearn.log_model(
            pipeline, "model", registered_model_name=REGISTERED_MODEL_NAME
        )

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    print(f"registered {REGISTERED_MODEL_NAME} version {latest} ({result.model_uri})")
    print(f"set MODEL_VERSION={latest} to serve it")


if __name__ == "__main__":
    main()
