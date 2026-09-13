"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str | None = None


class PredictResponse(BaseModel):
    """`predictions` are probabilities; `categories` is the decision.

    Both are returned rather than one or the other. The probabilities are what
    the model actually produces and what a caller needs to apply its own
    operating point; the categories are what this deployment decided, at the
    thresholds named alongside them. Returning the decision without the
    threshold that produced it would make the answer impossible to audit.
    """

    predictions: dict[str, float]
    categories: list[str]
    thresholds: dict[str, float]
    model_version: str
    latency_ms: float
