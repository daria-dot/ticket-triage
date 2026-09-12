"""Scoring shared by every approach, so the comparison is like-for-like.

Per-label precision and recall stay the primary reading: the five categories
are independent and unevenly covered, and one aggregate would hide that docs
has a twentieth of bug's support. Macro F1 is logged alongside purely because
comparing approaches needs a single ordering, not because it describes the
model well.
"""

from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from triage.features.dataset import LABEL_COLUMNS


def log_per_label_metrics(y_true: Any, y_pred: Any) -> float:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    for label, p, r, f, s in zip(LABEL_COLUMNS, precision, recall, f1, support, strict=True):
        name = label.removeprefix("is_")
        mlflow.log_metric(f"precision_{name}", p)
        mlflow.log_metric(f"recall_{name}", r)
        mlflow.log_metric(f"f1_{name}", f)
        mlflow.log_metric(f"support_{name}", int(s))
        print(f"{name:10s} precision={p:.3f} recall={r:.3f} f1={f:.3f} support={s}")

    macro_f1 = float(np.mean(f1))
    mlflow.log_metric("macro_f1", macro_f1)
    print(f"{'macro':10s} f1={macro_f1:.3f}")
    return macro_f1
