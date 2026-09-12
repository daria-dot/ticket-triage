"""SageMaker's serving contract: GET /ping and POST /invocations on port 8080.

Deliberately separate from triage.api.main. Inside SageMaker there is no
Postgres to log predictions to and no MLflow server to resolve a registry URI
against, so this loads the model from the artifact SageMaker unpacks and does
inference and nothing else. Logging and business rules belong to the service
that calls this endpoint, not to the model server.
"""

import os
from contextlib import asynccontextmanager
from functools import lru_cache

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

from triage.features.dataset import LABEL_COLUMNS, combine_title_body

# Where SageMaker unpacks model.tar.gz. Overridable so the container can be
# exercised locally against an exported model directory.
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")


@lru_cache
def get_model() -> Pipeline:
    return mlflow.sklearn.load_model(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load before accepting traffic: a container that can't load its model
    # should fail to start, not fail the first real request.
    get_model()
    yield


app = FastAPI(title="Ticket Triage SageMaker serving", lifespan=lifespan)


class InvocationRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str | None = None


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: InvocationRequest) -> dict[str, float]:
    text = combine_title_body(request.title, request.body)
    probabilities = get_model().predict_proba([text])[0]
    return {
        label.removeprefix("is_"): float(p)
        for label, p in zip(LABEL_COLUMNS, probabilities, strict=True)
    }
