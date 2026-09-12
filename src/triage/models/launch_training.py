"""Submits the embedding job to SageMaker and records the result in MLflow.

Runs on a spot instance: a GPU hour is roughly a third the on-demand price, and
an interruption on a job this short just means running it again. Uses AWS's
prebuilt PyTorch training image rather than a custom container, so nothing has
to build and push a multi-gigabyte image to run a twenty-minute job.
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
SOURCE_DIR = Path(__file__).resolve().parents[3] / "sagemaker_job"


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
        mlflow.log_param("approach", "embeddings")
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
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--max-seq-length", type=int, default=0, help="0 keeps the model's own default."
    )
    # The 2.3 image ships PyTorch 2.3, and current transformers requires >= 2.5.
    # On the older image transformers quietly disables its torch integration and
    # then dies on a type annotation referencing nn.Module.
    parser.add_argument("--framework-version", default="2.6.0")
    parser.add_argument("--py-version", default="py312")
    parser.add_argument("--on-demand", action="store_true", help="Disable spot pricing.")
    # Generous enough for a larger encoder over the full corpus, still bounded
    # so a hang cannot bill indefinitely. MiniLM needs ~20 minutes; bge-base at
    # 512 tokens is roughly five times the work.
    parser.add_argument("--max-run", type=int, default=10800)
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
        hyperparameters={
            "embedding-model": args.embedding_model,
            "max-seq-length": args.max_seq_length,
        },
        use_spot_instances=spot,
        # max_run caps the job itself so a hang can't quietly bill for hours;
        # max_wait additionally covers queueing for spot capacity and so must
        # exceed it.
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
            "embedding_model": args.embedding_model,
            "instance_type": args.instance_type,
            "spot": spot,
        },
    )


if __name__ == "__main__":
    main()
