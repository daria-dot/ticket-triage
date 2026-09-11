"""Persistence for logged predictions -- see sql/schema.sql."""

import json

from sqlalchemy import Engine, text


def log_prediction(
    engine: Engine,
    *,
    input_hash: str,
    input_text: str,
    output: dict[str, float],
    model_version: str,
    latency_ms: float,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO predictions
                    (input_hash, input_text, output, model_version, latency_ms)
                VALUES
                    (:input_hash, :input_text, :output, :model_version, :latency_ms)
                """
            ),
            {
                "input_hash": input_hash,
                "input_text": input_text,
                "output": json.dumps(output),
                "model_version": model_version,
                "latency_ms": latency_ms,
            },
        )
