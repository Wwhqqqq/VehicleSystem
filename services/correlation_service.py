from __future__ import annotations

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
import uvicorn

from test_backend.shared.langchain_utils import build_chat_model, message_text
from test_backend.shared.legacy_loader import load_prompts_module
from test_backend.shared.logging import get_logger
from test_backend.shared.redis_store import RedisStateStore
from test_backend.shared.schemas import CorrelationRequest, HealthResponse
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.correlation")
PROMPTS = load_prompts_module()
SETTINGS = get_settings()
STORE = RedisStateStore()
LAST_SERVICE_KEY = SETTINGS.runtime.redis_keys.last_service

app = FastAPI(title="correlation-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post(SETTINGS.service("correlation").route)
async def inference(payload: CorrelationRequest) -> dict[str, str]:
    last_info = STORE.get(LAST_SERVICE_KEY.format(sender_id=payload.sender_id))
    if not last_info:
        return {"data": "否"}

    try:
        _, last_query, last_reject, _ = last_info.split("#", 3)
    except ValueError:
        return {"data": "否"}

    if len(payload.query) > 1 and last_query == payload.query:
        return {"data": "是"}

    if last_reject == "N":
        return {"data": "否"}

    try:
        response = await build_chat_model("correlation").ainvoke(
            [
                SystemMessage(content=PROMPTS.CORRELATION_SYSTEM),
                HumanMessage(content=PROMPTS.CORRELATION_PROMPT.format(last_query, payload.query)),
            ]
        )
        answer = (message_text(response) or "否").strip()[:1]
    except Exception as exc:
        LOGGER.warning("Correlation failed: %s", exc)
        answer = "否"

    if answer != "是":
        answer = "否"
    return {"data": answer}


if __name__ == "__main__":
    endpoint = SETTINGS.service("correlation")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
