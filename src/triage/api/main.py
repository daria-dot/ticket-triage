"""FastAPI service: /predict, /health, /metrics, and a page to drive them."""

import hashlib
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, text

from triage.api.inference import decide, predict_probabilities, thresholds
from triage.api.predictions_db import log_prediction
from triage.api.schemas import PredictRequest, PredictResponse
from triage.config import get_settings
from triage.features.dataset import combine_title_body

settings = get_settings()
engine = create_engine(settings.database_url)

app = FastAPI(title="Ticket Triage API")

STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A page that drives /predict, mostly so the thresholds can be seen.

    Five probabilities and five different cuts is the one part of this service
    that does not survive being read as JSON -- the whole point is that the cut
    sits somewhere different for each label, which is a picture rather than a
    number. It calls the same endpoint as any other client and applies no logic
    of its own beyond drawing what comes back.
    """
    return (STATIC / "index.html").read_text()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    return {
        "status": "ok",
        "model_version": settings.model_version,
        "inference_backend": settings.inference_backend,
    }


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

    # Latency covers whichever backend served this, so the logged figure means
    # the same thing whether the model ran here or in SageMaker.
    start = time.perf_counter()
    predictions = predict_probabilities(request.title, request.body)
    latency_ms = (time.perf_counter() - start) * 1000

    # Probabilities are logged, not the decision. The thresholds belong to the
    # model version, which is logged beside them, so any past decision can be
    # reconstructed exactly -- while a logged decision could not be reinterpreted
    # if the operating point ever moves.
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
        categories=decide(predictions),
        thresholds=thresholds(),
        model_version=settings.model_version,
        latency_ms=latency_ms,
    )
