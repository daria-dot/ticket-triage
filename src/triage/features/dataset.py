"""Loads the training frame from the `issue_categories` view -- see sql/views/."""

import pandas as pd
from sqlalchemy import Engine

LABEL_COLUMNS = ["is_bug", "is_feature", "is_docs", "is_question", "is_duplicate"]


def load_issue_categories(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM issue_categories", engine)
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].astype(int)
    return df


def combine_text(df: pd.DataFrame) -> pd.Series:
    return (df["title"].fillna("") + "\n" + df["body"].fillna("")).str.strip()
