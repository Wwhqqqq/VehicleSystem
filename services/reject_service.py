from __future__ import annotations

import asyncio

from fastapi import FastAPI
import uvicorn

from test_backend.shared.legacy_models import get_reject_predictor
from test_backend.shared.logging import get_logger
from test_backend.shared.schemas import HealthResponse, RejectRequest
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.reject")
SETTINGS = get_settings()

app = FastAPI(title="reject-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _predict(query: str, threshold: float) -> tuple[int, float]:
    predictor = get_reject_predictor()
    return predictor.predict_binary(query, threshold)


@app.post(SETTINGS.service("reject").route)
async def inference(payload: RejectRequest) -> dict[str, str | int]:
    try:
        result, score = await asyncio.to_thread(_predict, payload.query, payload.thres)
    except Exception as exc:
        LOGGER.warning("Reject inference failed, fallback to accept: %s", exc)
        result, score = 1, 1.0
    return {"data": result, "score": str(score)}


if __name__ == "__main__":
    endpoint = SETTINGS.service("reject")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
