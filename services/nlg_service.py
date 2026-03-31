from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
import uvicorn

from test_backend.shared.langchain_utils import build_chat_model, message_text
from test_backend.shared.legacy_loader import load_prompts_module
from test_backend.shared.logging import get_logger
from test_backend.shared.schemas import HealthResponse, NlgRequest
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.nlg")
PROMPTS = load_prompts_module()
SETTINGS = get_settings()

app = FastAPI(title="nlg-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _action_phrase(query: str) -> str:
    text = str(query or "").strip()
    prefixes = ("请帮我", "麻烦你", "帮我", "请给我", "给我", "请")
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip() or text
    return text


def _deterministic_error_reply(query: str, tool_response: Any) -> str:
    if not isinstance(tool_response, dict):
        return ""

    error = str(tool_response.get("error", "") or "").strip()
    if not error:
        return ""

    service = str(tool_response.get("service", "") or "").strip()
    action = _action_phrase(query) or "处理这个请求"

    if error == "AMAP_MAPS_API_KEY is not set.":
        if service == "maps":
            return f"抱歉呀，暂时没办法帮你{action}啦，目前还没配置好高德地图服务哦～"
        if service == "weather":
            return "抱歉呀，暂时还没办法帮你查天气哦，目前还没配置好高德天气服务～"

    if service == "music":
        return "抱歉呀，暂时还没办法帮你调用音乐服务哦，相关配置还没准备好～"
    return error


def _deterministic_success_reply(query: str, tool_response: Any) -> str:
    if isinstance(tool_response, dict):
        message = str(tool_response.get("message", "") or "").strip()
        if message:
            return message

        pois = tool_response.get("pois")
        if isinstance(pois, list) and pois:
            first = pois[0]
            name = str(first.get("name", "") or "").strip()
            address = str(first.get("address", "") or "").strip()
            if name and address:
                return f"已为您找到{name}，地址是{address}，现在就可以开始导航。"
            if name:
                return f"已为您找到{name}，现在就可以开始导航。"

        weather = str(tool_response.get("天气", "") or "").strip()
        temperature = str(tool_response.get("温度", "") or "").strip()
        city = str(tool_response.get("城市", "") or "").strip()
        if weather and temperature:
            city_prefix = f"{city}" if city else "当前城市"
            return f"{city_prefix}天气{weather}，温度大约{temperature}度。"

    if isinstance(tool_response, list) and tool_response:
        first = tool_response[0] if isinstance(tool_response[0], dict) else {}
        title = str(first.get("title", "") or first.get("name", "") or "").strip()
        if title:
            return f"好呀～这就为你播放《{title}》。"

    return ""


@app.post(SETTINGS.service("nlg").route)
async def inference(payload: NlgRequest) -> dict[str, str]:
    answer = _deterministic_error_reply(payload.query, payload.tool_response)
    if answer:
        return {"data": answer}

    try:
        prompt = PROMPTS.NLG_PROMPT.format(payload.query, payload.tool_response)
        response = await build_chat_model("nlg").ainvoke([HumanMessage(content=prompt)])
        answer = message_text(response).strip()
    except Exception as exc:
        LOGGER.warning("NLG failed: %s", exc)
        answer = ""

    if not answer:
        answer = _deterministic_success_reply(payload.query, payload.tool_response)
    return {"data": answer}


if __name__ == "__main__":
    endpoint = SETTINGS.service("nlg")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
