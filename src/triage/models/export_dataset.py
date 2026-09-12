"""Exports the training frame to S3 so cloud compute can reach it.

The corpus lives in local Postgres, which a SageMaker job cannot see. This
writes it out once as Parquet -- text, labels and the hashed split together, so
the job trains and evaluates on exactly the partition everything else uses.
"""

import argparse
from pathlib import Path

import boto3
from sqlalchemy import create_engine

from triage.config import get_settings
from triage.features.dataset import LABEL_COLUMNS, load_issue_categories

DEFAULT_KEY = "datasets/issue_categories.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--local-path", default="data/issue_categories.parquet")
    args = parser.parse_args()

    settings = get_settings()
    df = load_issue_categories(create_engine(settings.database_url))

    columns = ["repo", "issue_number", "title", "body", "split", *LABEL_COLUMNS]
    df = df[columns]

    local = Path(args.local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local, index=False, compression="snappy")

    size_mb = local.stat().st_size / 1e6
    counts = df["split"].value_counts().to_dict()
    print(f"{len(df)} rows ({counts}) -> {local} ({size_mb:.1f} MB)")

    boto3.client("s3", region_name=settings.aws_region).upload_file(
        str(local), args.bucket, args.key
    )
    print(f"uploaded to s3://{args.bucket}/{args.key}")


if __name__ == "__main__":
    main()
