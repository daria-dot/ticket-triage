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
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

LABEL_COLUMNS = ["is_bug", "is_feature", "is_docs", "is_question", "is_duplicate"]
DEFAULT_THRESHOLD = 0.5


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """The threshold maximising F1 for one label, and the F1 it achieves.

    Deliberately a copy of the TF-IDF job's function rather than an import.
    SageMaker uploads one source directory per job and installs only its
    requirements, so neither job can import the other or the triage package --
    the same reason LABEL_COLUMNS is duplicated here. The risk that matters is
    the two drifting apart, so this is why they must not: an approach scored
    under a different threshold rule than the one it is compared against is not
    being compared at all.
    """
    if int(y_true.sum()) == 0:
        raise ValueError("cannot fit a threshold for a label with no positive examples")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]

    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def score(y_true: np.ndarray, y_pred: np.ndarray, suffix: str) -> tuple[dict, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics: dict[str, float | int] = {}
    for label, p, r, f, s in zip(LABEL_COLUMNS, precision, recall, f1, support, strict=True):
        name = label.removeprefix("is_")
        metrics |= {
            f"precision_{name}{suffix}": float(p),
            f"recall_{name}{suffix}": float(r),
            f"f1_{name}{suffix}": float(f),
            f"support_{name}": int(s),
        }
        print(f"  {name:10s} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}", flush=True)

    macro = float(np.mean(f1))
    metrics[f"macro_f1{suffix}"] = macro
    print(f"  {'macro':10s} f1={macro:.3f}", flush=True)
    return metrics, macro


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
    splits = set(df["split"].unique())
    if splits != {"train", "val", "test"}:
        raise ValueError(f"unexpected split values: {sorted(splits)}")

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

    # `~is_train` would now sweep val into the evaluation set. Every side is
    # named explicitly so adding a split later cannot silently change what
    # "test" means.
    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "val").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    y = df[LABEL_COLUMNS].astype(int).to_numpy()
    print(f"train={is_train.sum()} val={is_val.sum()} test={is_test.sum()}", flush=True)

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
    )
    classifier.fit(embeddings[is_train], y[is_train])

    val_scores = classifier.predict_proba(embeddings[is_val])
    test_scores = classifier.predict_proba(embeddings[is_test])

    thresholds: dict[str, float] = {}
    print("fitted thresholds (on val):", flush=True)
    for index, label in enumerate(LABEL_COLUMNS):
        name = label.removeprefix("is_")
        threshold, val_f1 = best_threshold(y[is_val][:, index], val_scores[:, index])
        thresholds[name] = threshold
        print(f"  {name:10s} threshold={threshold:.4f} (val f1={val_f1:.3f})", flush=True)

    threshold_vector = np.array([thresholds[c.removeprefix("is_")] for c in LABEL_COLUMNS])

    print(f"test @ default {DEFAULT_THRESHOLD}:", flush=True)
    default_metrics, default_macro = score(
        y[is_test], (test_scores >= DEFAULT_THRESHOLD).astype(int), "_at_default"
    )

    print("test @ fitted thresholds:", flush=True)
    tuned_metrics, tuned_macro = score(
        y[is_test], (test_scores >= threshold_vector).astype(int), "_tuned"
    )

    print(f"macro f1 {default_macro:.3f} -> {tuned_macro:.3f}", flush=True)

    metrics: dict[str, object] = {
        "approach": "embeddings",
        "embed_seconds": embed_seconds,
        "device": device,
        "embedding_model": args.embedding_model,
        "max_seq_length": model.max_seq_length,
        "embedding_dims": int(embeddings.shape[1]),
        "train_rows": int(is_train.sum()),
        "val_rows": int(is_val.sum()),
        "test_rows": int(is_test.sum()),
        **default_metrics,
        **tuned_metrics,
        **{f"threshold_{name}": value for name, value in thresholds.items()},
    }

    model_dir = Path(args.model_dir)
    dump(classifier, model_dir / "classifier.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "embedding_model.txt").write_text(args.embedding_model)
    print(f"wrote model, thresholds and metrics to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
