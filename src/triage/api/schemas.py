"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str | None = None


class PredictResponse(BaseModel):
    predictions: dict[str, float]
    model_version: str
    latency_ms: float
