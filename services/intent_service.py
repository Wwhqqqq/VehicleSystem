from __future__ import annotations

import asyncio

from fastapi import FastAPI
import uvicorn

from test_backend.shared.legacy_models import get_intent_predictor
from test_backend.shared.logging import get_logger
from test_backend.shared.schemas import HealthResponse, IntentRequest
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.intent")
SETTINGS = get_settings()
TOPK = 5

app = FastAPI(title="intent-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _predict(query: str) -> tuple[list[int], list[float]]:
    predictor = get_intent_predictor()
    return predictor.predict_topk(query, TOPK)


@app.post(SETTINGS.service("intent").route)
async def inference(payload: IntentRequest) -> dict[str, str]:
    try:
        labels, scores = await asyncio.to_thread(_predict, payload.query)
    except Exception as exc:
        LOGGER.warning("Intent inference failed, fallback to unknown: %s", exc)
        labels = [3] * TOPK
        scores = [1.0] * TOPK

    result = {
        "data": ",".join(str(item) for item in labels),
        "score": ",".join(str(item) for item in scores),
    }
    LOGGER.info("Trace ID: %s, Request: %s, response: %s, confidence: %s", payload.trace_id, payload.query, result["data"], result["score"])
    return result


if __name__ == "__main__":
    endpoint = SETTINGS.service("intent")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
