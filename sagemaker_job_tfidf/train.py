"""Runs inside a SageMaker training job. Fits the TF-IDF baseline, then fits a
decision threshold per label on the validation split.

Why this exists as a job at all, when TF-IDF needs no GPU: the corpus does not
fit comfortably in 8GB alongside a running Docker stack, and the point of
moving it is the machine staying usable, not the arithmetic being faster.

What it is actually correcting: the baseline reported its metrics through
`predict()`, which cuts every label at 0.5. With `class_weight="balanced"` on
labels that run from 19.8% positive (bug) down to 1.3% (docs), 0.5 sits nowhere
near the F1-optimal point -- and the published results ordered exactly by label
rarity, which is the signature of a threshold artefact rather than five
independent modelling failures. The serving path never applied 0.5 at all; it
returns probabilities. So this fits the operating point that was previously
left at a library default.

Standalone by design: SageMaker uploads this directory on its own and installs
only requirements.txt, so it cannot import the triage package. Constants
duplicated from it are asserted against the data rather than trusted.

SageMaker's contract: input arrives under SM_CHANNEL_TRAINING, artefacts go to
SM_MODEL_DIR, and a non-zero exit marks the job failed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

LABEL_COLUMNS = ["is_bug", "is_feature", "is_docs", "is_question", "is_duplicate"]
REQUIRED_SPLITS = {"train", "val", "test"}
DEFAULT_THRESHOLD = 0.5


def combine_title_body(df: pd.DataFrame) -> list[str]:
    return (df["title"].fillna("") + "\n" + df["body"].fillna("")).str.strip().tolist()


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """The threshold maximising F1 for one label, and the F1 it achieves.

    Uses the exact set of thresholds where predictions change rather than a
    fixed grid, so nothing is missed between grid points and nothing is spent
    evaluating cuts that produce an identical confusion matrix.

    Raises rather than guessing if the label has no positives to fit against:
    a silently returned default here would be indistinguishable, downstream,
    from a threshold that had genuinely been fitted.
    """
    positives = int(y_true.sum())
    if positives == 0:
        raise ValueError("cannot fit a threshold for a label with no positive examples")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve returns one more point than threshold: the final
    # point is recall=0, precision=1, which corresponds to no threshold at all.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
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
    if splits != REQUIRED_SPLITS:
        raise ValueError(f"expected splits {sorted(REQUIRED_SPLITS)}, got {sorted(splits)}")

    text = combine_title_body(df)
    y = df[LABEL_COLUMNS].astype(int).to_numpy()
    is_train = (df["split"] == "train").to_numpy()
    is_val = (df["split"] == "val").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    print(
        f"train={is_train.sum()} val={is_val.sum()} test={is_test.sum()}",
        flush=True,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    min_df=args.min_df,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=args.max_iter, class_weight="balanced")
                ),
            ),
        ]
    )

    started = time.perf_counter()
    pipeline.fit([text[i] for i in np.flatnonzero(is_train)], y[is_train])
    fit_seconds = time.perf_counter() - started
    print(f"fitted in {fit_seconds:.0f}s", flush=True)

    val_scores = pipeline.predict_proba([text[i] for i in np.flatnonzero(is_val)])
    test_scores = pipeline.predict_proba([text[i] for i in np.flatnonzero(is_test)])

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

    import sklearn

    metrics: dict[str, object] = {
        "approach": "tfidf",
        # Recorded, not assumed: the pipeline is unpickled by the API under
        # whatever it has installed, and a mismatch should be visible here
        # rather than inferred from which image happened to run.
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "fit_seconds": fit_seconds,
        "train_rows": int(is_train.sum()),
        "val_rows": int(is_val.sum()),
        "test_rows": int(is_test.sum()),
        "max_features": args.max_features,
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
        **default_metrics,
        **tuned_metrics,
        **{f"threshold_{name}": value for name, value in thresholds.items()},
    }

    model_dir = Path(args.model_dir)
    dump(pipeline, model_dir / "pipeline.joblib")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote model, thresholds and metrics to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
