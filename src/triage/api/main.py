"""FastAPI service: /predict, /health, /metrics."""

import hashlib
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

from triage.api.model_loader import get_model
from triage.api.predictions_db import log_prediction
from triage.api.schemas import PredictRequest, PredictResponse
from triage.config import get_settings
from triage.features.dataset import LABEL_COLUMNS, combine_title_body

settings = get_settings()
engine = create_engine(settings.database_url)

app = FastAPI(title="Ticket Triage API")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    return {"status": "ok", "model_version": settings.model_version}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                    SELECT count(*) AS total_predictions,
                           avg(latency_ms) AS avg_latency_ms,
                           max(created_at) AS last_prediction_at
                    FROM predictions
                    """
                )
            )
            .mappings()
            .one()
        )
    return dict(row)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    text_input = combine_title_body(request.title, request.body)
    input_hash = hashlib.sha256(text_input.encode("utf-8")).hexdigest()

    model = get_model()
    start = time.perf_counter()
    probabilities = model.predict_proba([text_input])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    predictions = {
        label.removeprefix("is_"): float(p)
        for label, p in zip(LABEL_COLUMNS, probabilities, strict=True)
    }

    log_prediction(
        engine,
        input_hash=input_hash,
        input_text=text_input,
        output=predictions,
        model_version=settings.model_version,
        latency_ms=latency_ms,
    )

    return PredictResponse(
        predictions=predictions,
        model_version=settings.model_version,
        latency_ms=latency_ms,
    )
