from __future__ import annotations

import json
import re

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uvicorn

from test_backend.shared.langchain_utils import build_chat_model, message_text
from test_backend.shared.legacy_loader import load_prompts_module
from test_backend.shared.logging import get_logger
from test_backend.shared.redis_store import RedisStateStore
from test_backend.shared.schemas import ChatRequest, HealthResponse
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.chat")
PROMPTS = load_prompts_module()
SETTINGS = get_settings()
STORE = RedisStateStore()
HISTORY_KEY = SETTINGS.runtime.redis_keys.chat_history
MAX_HISTORY = SETTINGS.runtime.max_history
TTL = SETTINGS.runtime.chat_history_ttl
SENTENCE_PATTERN = re.compile(SETTINGS.runtime.chat_sentence_pattern)
FLUSH_EVERY = SETTINGS.runtime.chat_flush_every
HISTORY_FACT_PROMPT = (
    "回答当前问题时，请优先参考上面的多轮对话历史。"
    "如果用户之前已经明确提供过事实信息，就直接基于历史回答，"
    "不要说自己无法感知、无法看到，除非历史里确实没有答案。"
)

app = FastAPI(title="chat-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _history_messages(sender_id: str, multiturn: bool):
    if not multiturn:
        return [], []
    history_raw = STORE.get(HISTORY_KEY.format(sender_id=sender_id))
    if not history_raw:
        return [], []
    history = json.loads(history_raw)
    messages = []
    for item in history:
        if item.get("role") == "assistant":
            messages.append(AIMessage(content=item.get("content", "")))
        else:
            messages.append(HumanMessage(content=item.get("content", "")))
    return history, messages


async def _frame_stream(payload: ChatRequest):
    serialized_history, history_messages = _history_messages(payload.sender_id, payload.multiturn)
    messages = [SystemMessage(content=PROMPTS.BOT_CHAT_SYSTEM_PROMPT)]
    if history_messages:
        messages.append(SystemMessage(content=HISTORY_FACT_PROMPT))
    messages.extend(history_messages)
    messages.append(HumanMessage(content=payload.query))

    answer = ""
    frame = ""
    counter = 1

    try:
        async for chunk in build_chat_model("chat", streaming=True).astream(messages):
            text = message_text(chunk)
            if not text:
                continue
            answer += text
            frame += text
            flush = bool(SENTENCE_PATTERN.search(text)) or counter % FLUSH_EVERY == 0
            if flush and frame.strip():
                yield (json.dumps({"delta": frame}, ensure_ascii=False) + "\n").encode("utf-8")
                frame = ""
            counter += 1
    except Exception as exc:
        LOGGER.warning("Chat streaming failed: %s", exc)
        fallback = "抱歉，网络有点问题，请您再试一下。"
        yield (json.dumps({"delta": fallback}, ensure_ascii=False) + "\n").encode("utf-8")
        answer = fallback
        frame = ""

    if frame.strip():
        yield (json.dumps({"delta": frame}, ensure_ascii=False) + "\n").encode("utf-8")

    serialized_history.append({"role": "user", "content": payload.query})
    serialized_history.append({"role": "assistant", "content": answer})
    STORE.set(
        HISTORY_KEY.format(sender_id=payload.sender_id),
        json.dumps(serialized_history[-MAX_HISTORY:], ensure_ascii=False),
        ex=TTL,
    )
    yield (json.dumps({"done": True, "answer": answer}, ensure_ascii=False) + "\n").encode("utf-8")


@app.post(SETTINGS.service("chat").route)
async def stream_chat(payload: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_frame_stream(payload), media_type="application/x-ndjson")


if __name__ == "__main__":
    endpoint = SETTINGS.service("chat")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
