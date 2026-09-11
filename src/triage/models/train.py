"""Baseline model: TF-IDF + one-vs-rest logistic regression, multi-label.

Per-label precision/recall only, never a single accuracy number -- the five
categories are independent and unevenly covered (e.g. `is_duplicate` has no
mapped label at all for two of the four ingested repos), so one aggregate
score would hide more than it shows.
"""

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from triage.config import get_settings
from triage.features.dataset import LABEL_COLUMNS, combine_text, load_issue_categories

EXPERIMENT_NAME = "ticket-triage-baseline"
REGISTERED_MODEL_NAME = "ticket-triage-baseline"

VECTORIZER_PARAMS = {"max_features": 20_000, "ngram_range": (1, 2), "min_df": 5}
CLASSIFIER_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)

    df = load_issue_categories(engine)
    x = combine_text(df)
    y = df[LABEL_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**VECTORIZER_PARAMS)),
            ("clf", OneVsRestClassifier(LogisticRegression(**CLASSIFIER_PARAMS))),
        ]
    )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        mlflow.log_params({f"tfidf_{k}": v for k, v in VECTORIZER_PARAMS.items()})
        mlflow.log_params({f"clf_{k}": v for k, v in CLASSIFIER_PARAMS.items()})
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))

        precision, recall, _, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        for label, p, r, s in zip(LABEL_COLUMNS, precision, recall, support, strict=True):
            name = label.removeprefix("is_")
            mlflow.log_metric(f"precision_{name}", p)
            mlflow.log_metric(f"recall_{name}", r)
            mlflow.log_metric(f"support_{name}", int(s))
            print(f"{name:10s} precision={p:.3f} recall={r:.3f} support={s}")

        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    main()
