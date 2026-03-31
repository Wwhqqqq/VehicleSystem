from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uvicorn

from test_backend.shared.langchain_utils import build_chat_model, message_text
from test_backend.shared.legacy_loader import load_prompts_module
from test_backend.shared.logging import get_logger
from test_backend.shared.redis_store import RedisStateStore
from test_backend.shared.schemas import ArbitrationRequest, HealthResponse
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.arbitration")
PROMPTS = load_prompts_module()
SETTINGS = get_settings()
STORE = RedisStateStore()
MAX_HISTORY = SETTINGS.runtime.max_history
TTL = SETTINGS.runtime.arbitration_history_ttl
HISTORY_KEY = SETTINGS.runtime.redis_keys.arbitration_history
ASSISTANT_CHAT_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"^你叫(什么|啥)名字[？?]?$",
        r"^你是谁[？?]?$",
        r"^介绍(一下|下)?你自己[。！!？?]?$",
        r"^你能做什么[？?]?$",
        r"^你会什么[？?]?$",
        r"^你从哪里来[？?]?$",
        r"^你多大(了)?[？?]?$",
        r"^你几岁(了)?[？?]?$",
        r"^你是(谁家|哪家|哪个公司).*[？?]?$",
        r"^你的名字是(什么|啥)[？?]?$",
    )
]

app = FastAPI(title="arbitration-service")


def _to_message(item: dict[str, Any]):
    role = item.get("role")
    content = item.get("content", "")
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _heuristic_code(query: str) -> str | None:
    normalized = str(query or "").strip()
    if not normalized:
        return None
    for pattern in ASSISTANT_CHAT_PATTERNS:
        if pattern.match(normalized):
            return "C"
    return None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post(SETTINGS.service("arbitration").route)
async def inference(payload: ArbitrationRequest) -> dict[str, str]:
    history_key = HISTORY_KEY.format(sender_id=payload.sender_id)
    history_raw = STORE.get(history_key)
    history = json.loads(history_raw) if history_raw else []
    history.append({"role": "user", "content": payload.query})

    code = _heuristic_code(payload.query)
    if code is None:
        messages = [SystemMessage(content=PROMPTS.ARBITRAION_SYSTEM_PROMPT)]
        messages.extend(_to_message(item) for item in history)

        try:
            response = await build_chat_model("arbitration").ainvoke(messages)
            text = (message_text(response) or "A").strip()
            code = text[0] if text else "A"
        except Exception as exc:
            LOGGER.warning("Arbitration failed, fallback to task: %s", exc)
            code = "A"

    if code not in {"A", "B", "C", "D"}:
        code = "A"

    history.append({"role": "assistant", "content": code})
    STORE.set(history_key, json.dumps(history[-MAX_HISTORY:], ensure_ascii=False), ex=TTL)

    mapping = {"A": "task", "B": "faq", "C": "chat", "D": "chat"}
    result = mapping[code]
    LOGGER.info("sender_id=%s query=%s arbitration=%s", payload.sender_id, payload.query, result)
    return {"data": result, "raw": code}


if __name__ == "__main__":
    endpoint = SETTINGS.service("arbitration")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
