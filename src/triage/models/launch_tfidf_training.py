"""Submits the TF-IDF + threshold-fitting job to SageMaker, and records it in MLflow.

CPU, not GPU: TF-IDF and logistic regression get nothing from a GPU. What moving
this off the laptop buys is 16GB of RAM for a corpus that does not fit
comfortably in 8GB beside a running Docker stack, and a machine that stays
usable while it runs. Spot pricing applies for the same reason it does on the
embedding job -- an interruption on a job this short just means running it again.

Uses the PyTorch image on a CPU instance rather than the SKLearn one, which
reads oddly for a job with no tensors in it. The newest SKLearn image is Python
3.9 with numpy 1.x, and installing a scikit-learn recent enough to match the
serving runtime breaks its compiled extensions at import. The PyTorch py312
image already carries a mutually consistent numpy and scikit-learn, and is the
image the embedding job proved out in this account. The instance type is what
selects the CPU build of it.
"""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import boto3
import mlflow
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

from triage.config import get_settings

EXPERIMENT_NAME = "ticket-triage-baseline"
SOURCE_DIR = Path(__file__).resolve().parents[3] / "sagemaker_job_tfidf"


def _log_outputs(model_data: str, region: str, extra_params: dict[str, object]) -> None:
    """Pull metrics.json out of the job's output archive and into MLflow."""
    bucket, _, key = model_data.removeprefix("s3://").partition("/")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "model.tar.gz"
        boto3.client("s3", region_name=region).download_file(bucket, key, str(archive))
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        metrics = json.loads((Path(tmp) / "metrics.json").read_text())

    with mlflow.start_run():
        mlflow.log_param("trained_on", "sagemaker")
        mlflow.log_params(extra_params)
        for name, value in metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(name, value)
            else:
                mlflow.log_param(name, value)
        mlflow.log_param("model_data", model_data)

    for name, value in sorted(metrics.items()):
        print(f"{name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--input-key", default="datasets/issue_categories.parquet")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument("--framework-version", default="2.6.0")
    parser.add_argument("--py-version", default="py312")
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--on-demand", action="store_true", help="Disable spot pricing.")
    # Bounded so a hang cannot bill indefinitely. The fit is minutes, not hours.
    parser.add_argument("--max-run", type=int, default=5400)
    args = parser.parse_args()

    settings = get_settings()
    session = Session(boto3.Session(region_name=settings.aws_region))
    role = boto3.client("iam").get_role(RoleName="ticket-triage-sagemaker-execution")["Role"]["Arn"]

    spot = not args.on_demand
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        role=role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=args.framework_version,
        py_version=args.py_version,
        sagemaker_session=session,
        output_path=f"s3://{args.bucket}/training-output",
        hyperparameters={"max-features": args.max_features},
        use_spot_instances=spot,
        max_run=args.max_run,
        max_wait=args.max_run + 3600 if spot else None,
    )

    print(f"submitting {args.instance_type} ({'spot' if spot else 'on-demand'})")
    estimator.fit({"training": f"s3://{args.bucket}/{args.input_key}"})

    print(f"model artifact: {estimator.model_data}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    _log_outputs(
        str(estimator.model_data),
        settings.aws_region,
        {
            "approach": "tfidf",
            "instance_type": args.instance_type,
            "spot": spot,
            "max_features": args.max_features,
        },
    )


if __name__ == "__main__":
    main()
