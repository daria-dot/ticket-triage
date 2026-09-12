"""Runs inside a SageMaker training job. Embeds the corpus on GPU, then fits
the same one-vs-rest classifier the local baseline uses.

Standalone by design: SageMaker uploads this directory on its own and installs
only requirements.txt, so it cannot import the triage package. The handful of
constants duplicated from it are asserted against the data at runtime rather
than trusted, since a silent divergence would make the comparison meaningless.

SageMaker's contract: input arrives under SM_CHANNEL_TRAINING, the fitted model
goes to SM_MODEL_DIR, and a non-zero exit marks the job failed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

LABEL_COLUMNS = ["is_bug", "is_feature", "is_docs", "is_question", "is_duplicate"]


def combine_title_body(df: pd.DataFrame) -> list[str]:
    return (df["title"].fillna("") + "\n" + df["body"].fillna("")).str.strip().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=1000)
    # 0 keeps whatever the model was trained with. Raising it past that is
    # possible but degrades quality, since position embeddings beyond the
    # training length were never learned.
    parser.add_argument("--max-seq-length", type=int, default=0)
    parser.add_argument(
        "--train-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    )
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = parser.parse_args()

    parquet = next(Path(args.train_dir).glob("*.parquet"))
    df = pd.read_parquet(parquet)
    print(f"loaded {len(df)} rows from {parquet}", flush=True)

    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"exported dataset is missing label columns: {sorted(missing)}")
    if set(df["split"].unique()) != {"train", "test"}:
        raise ValueError(f"unexpected split values: {sorted(df['split'].unique())}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    model = SentenceTransformer(args.embedding_model, device=device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
    print(f"truncating at {model.max_seq_length} tokens", flush=True)

    started = time.perf_counter()
    embeddings = model.encode(
        combine_title_body(df),
        batch_size=args.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embed_seconds = time.perf_counter() - started
    print(
        f"embedded {len(df)} docs in {embed_seconds:.0f}s ({len(df) / embed_seconds:.0f}/sec)",
        flush=True,
    )

    is_train = (df["split"] == "train").to_numpy()
    y = df[LABEL_COLUMNS].astype(int).to_numpy()

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
    )
    classifier.fit(embeddings[is_train], y[is_train])
    predictions = classifier.predict(embeddings[~is_train])

    precision, recall, f1, support = precision_recall_fscore_support(
        y[~is_train], predictions, average=None, zero_division=0
    )

    metrics = {
        "embed_seconds": embed_seconds,
        "device": device,
        "embedding_model": args.embedding_model,
        "max_seq_length": model.max_seq_length,
        "embedding_dims": int(embeddings.shape[1]),
    }
    for label, p, r, f, s in zip(LABEL_COLUMNS, precision, recall, f1, support, strict=True):
        name = label.removeprefix("is_")
        metrics |= {
            f"precision_{name}": float(p),
            f"recall_{name}": float(r),
            f"f1_{name}": float(f),
            f"support_{name}": int(s),
        }
        print(f"{name:10s} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}", flush=True)

    metrics["macro_f1"] = float(np.mean(f1))
    print(f"{'macro':10s} f1={metrics['macro_f1']:.3f}", flush=True)

    model_dir = Path(args.model_dir)
    dump(classifier, model_dir / "classifier.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "embedding_model.txt").write_text(args.embedding_model)
    print(f"wrote model and metrics to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
