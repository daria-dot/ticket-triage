"""Loads the training frame from the `issue_categories` view -- see sql/views/."""

import pandas as pd
from sqlalchemy import Engine

LABEL_COLUMNS = ["is_bug", "is_feature", "is_docs", "is_question", "is_duplicate"]


def load_issue_categories(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM issue_categories", engine)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(int)
    return df


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test frames, partitioned by the view's hashed `split` column.

    Deliberately not train_test_split: a seed fixes which positions go to test,
    not which issues, so comparing one approach against another needs the split
    pinned to issue identity instead. See sql/views/issue_categories.sql.
    """
    return df[df["split"] == "train"], df[df["split"] == "test"]


def combine_text(df: pd.DataFrame) -> pd.Series:
    return (df["title"].fillna("") + "\n" + df["body"].fillna("")).str.strip()


def combine_title_body(title: str, body: str | None) -> str:
    """Single-record equivalent of combine_text, for online inference. Must stay in sync."""
    return f"{title}\n{body or ''}".strip()
