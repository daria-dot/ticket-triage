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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from triage.config import get_settings
from triage.features.dataset import (
    LABEL_COLUMNS,
    combine_text,
    load_issue_categories,
    split_frames,
)
from triage.models.evaluation import log_per_label_metrics

EXPERIMENT_NAME = "ticket-triage-baseline"
REGISTERED_MODEL_NAME = "ticket-triage-baseline"

VECTORIZER_PARAMS = {"max_features": 20_000, "ngram_range": (1, 2), "min_df": 5}
CLASSIFIER_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)

    train_df, test_df = split_frames(load_issue_categories(engine))
    x_train, y_train = combine_text(train_df), train_df[LABEL_COLUMNS]
    x_test, y_test = combine_text(test_df), test_df[LABEL_COLUMNS]

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

        mlflow.log_param("approach", "tfidf")
        mlflow.log_params({f"tfidf_{k}": v for k, v in VECTORIZER_PARAMS.items()})
        mlflow.log_params({f"clf_{k}": v for k, v in CLASSIFIER_PARAMS.items()})
        mlflow.log_param("train_rows", len(x_train))
        mlflow.log_param("test_rows", len(x_test))

        log_per_label_metrics(y_test, y_pred)

        mlflow.sklearn.log_model(pipeline, "model", registered_model_name=REGISTERED_MODEL_NAME)


if __name__ == "__main__":
    main()
