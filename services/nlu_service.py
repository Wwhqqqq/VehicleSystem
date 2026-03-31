from __future__ import annotations

import asyncio
import copy
import json

import httpx
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage
import uvicorn

from test_backend.shared.langchain_utils import build_chat_model
from test_backend.shared.legacy_loader import (
    load_class_mappings,
    load_prompts_module,
    load_slot_intent_map,
    load_slot_process_module,
    load_tool_specs,
)
from test_backend.shared.legacy_models import get_intent_predictor
from test_backend.shared.logging import get_logger
from test_backend.shared.schemas import HealthResponse, NluRequest
from test_backend.shared.service_client import post_service_json
from test_backend.shared.settings import get_settings


LOGGER = get_logger("test_backend.nlu")
PROMPTS = load_prompts_module()
SLOT_PROCESS = load_slot_process_module()
SETTINGS = get_settings()
ID_TO_FUNC, FUNC_TO_NAME, NAME_TO_ID = load_class_mappings()
SLOT_MAP = load_slot_intent_map()
TOPK = 5
UNKNOWN_NLU_VALUE = "未知-无"

TOOL_MAP: dict[str, list[dict]] = {}
for tool in load_tool_specs():
    name = tool.get("function", {}).get("name")
    TOOL_MAP.setdefault(name, []).append(tool)

WINDOW_ACTION_SYNONYMS = {
    "open": ("\u6253\u5f00", "\u5f00\u542f", "\u5f00", "\u964d\u4e0b", "\u964d\u4e0b\u6765"),
    "close": ("\u5173\u95ed", "\u5173\u4e0a", "\u5173\u6389", "\u5173", "\u5347\u8d77", "\u5347\u8d77\u6765"),
    "set": (
        "\u8bbe\u7f6e",
        "\u8c03\u5230",
        "\u8c03\u6574",
        "\u5347\u4e00\u70b9",
        "\u964d\u4e00\u70b9",
        "\u5f00\u4e00\u70b9",
        "\u7559\u4e2a\u7f1d",
        "\u4e00\u534a",
        "\u534a\u5f00",
    ),
}
WINDOW_TARGET_TERMS = (
    "\u8f66\u7a97",
    "\u7a97\u6237",
    "\u7a97\u5b50",
    "\u5f00\u7a97",
    "\u5173\u7a97",
    "\u5347\u7a97",
    "\u964d\u7a97",
)
WINDOW_POSITION_TERMS = (
    ("\u4e3b\u9a7e\u9a76", "\u4e3b\u9a7e"),
    ("\u526f\u9a7e\u9a76", "\u526f\u9a7e"),
    ("\u5de6\u540e", "\u5de6\u540e"),
    ("\u53f3\u540e", "\u53f3\u540e"),
    ("\u524d\u6392", "\u524d\u6392"),
    ("\u540e\u6392", "\u540e\u6392"),
    ("\u5168\u90e8", "\u5168\u90e8"),
)
WINDOW_CONTROL_FUNCTIONS = {"Open_Window", "Close_Window", "Set_Window", "Open_Window_Diagonal", "Close_Window_Diagonal"}

app = FastAPI(title="nlu-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _intent_recall(query: str) -> tuple[list[int], list[float]]:
    predictor = get_intent_predictor()
    return predictor.predict_topk(query, TOPK)


def _merge_tool_specs(tool_specs: list[dict]) -> list[dict]:
    merged_by_name: dict[str, dict] = {}
    ordered_names: list[str] = []

    for tool in tool_specs:
        function_name = tool.get("function", {}).get("name")
        if not function_name:
            continue
        if function_name not in merged_by_name:
            merged_by_name[function_name] = copy.deepcopy(tool)
            ordered_names.append(function_name)
            continue

        merged = merged_by_name[function_name]
        merged_function = merged.setdefault("function", {})
        incoming_function = tool.get("function", {})

        existing_description = str(merged_function.get("description", "")).strip()
        incoming_description = str(incoming_function.get("description", "")).strip()
        if incoming_description and incoming_description not in existing_description:
            merged_function["description"] = "\n".join(
                part for part in (existing_description, incoming_description) if part
            )

        merged_parameters = merged_function.setdefault(
            "parameters",
            {"type": "object", "properties": {}, "required": []},
        )
        incoming_parameters = incoming_function.get("parameters", {})

        merged_properties = merged_parameters.setdefault("properties", {})
        for key, value in incoming_parameters.get("properties", {}).items():
            merged_properties.setdefault(key, value)

        merged_required = list(merged_parameters.get("required", []))
        for key in incoming_parameters.get("required", []):
            if key not in merged_required:
                merged_required.append(key)
        if merged_required:
            merged_parameters["required"] = merged_required

    return [merged_by_name[name] for name in ordered_names]


def _build_candidate_tools(result_ids: list[int]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    raw_tools: list[dict] = []
    top1_raw_tools: list[dict] = []
    ranked_raw_tools: list[dict] = []
    seen_function_names: set[str] = set()

    for index, intent_id in enumerate(result_ids):
        function_name = ID_TO_FUNC.get(str(intent_id))
        if not function_name:
            continue
        tool_specs = TOOL_MAP.get(function_name, [])
        if not tool_specs:
            continue
        raw_tools.extend(tool_specs)
        if index == 0:
            top1_raw_tools.extend(copy.deepcopy(tool_specs))
        if function_name not in seen_function_names:
            ranked_raw_tools.extend(copy.deepcopy(tool_specs))
            seen_function_names.add(function_name)

    return _merge_tool_specs(raw_tools), _merge_tool_specs(top1_raw_tools), top1_raw_tools, ranked_raw_tools




def _extract_window_position(query: str) -> str | None:
    compact = query.replace(" ", "")
    for value, keyword in WINDOW_POSITION_TERMS:
        if keyword in compact:
            return value
    return None


def _extract_window_ratio(query: str) -> str | None:
    compact = query.replace(" ", "")
    ratio_aliases = {
        "\u4e00\u534a": "50",
        "\u534a\u5f00": "50",
        "\u4e00\u70b9": "20",
        "\u7559\u4e2a\u7f1d": "10",
    }
    for token, ratio in ratio_aliases.items():
        if token in compact:
            return ratio

    import re

    match = re.search(r"(\d{1,3})\s*[%?]", compact)
    if match:
        value = match.group(1)
        return value if 0 <= int(value) <= 100 else None
    return None


def _heuristic_control_nlu(query: str) -> str | None:
    compact = query.replace(" ", "")

    if any(token in compact for token in ("\u901a\u98ce", "\u6362\u6c14")):
        intent_name = FUNC_TO_NAME.get("Open_Window_Diagonal", "\u6253\u5f00\u901a\u98ce\u6a21\u5f0f")
        return f"{intent_name}-\u65e0"

    has_window_target = any(token in compact for token in WINDOW_TARGET_TERMS)
    has_generic_window = "\u7a97" in compact and "\u5929\u7a97" not in compact
    if not has_window_target and not has_generic_window:
        return None

    ratio = _extract_window_ratio(compact)
    action_key: str | None = None
    if any(token in compact for token in WINDOW_ACTION_SYNONYMS["close"]):
        action_key = "close"
    elif ratio is not None or any(token in compact for token in WINDOW_ACTION_SYNONYMS["set"]):
        action_key = "set"
    elif any(token in compact for token in WINDOW_ACTION_SYNONYMS["open"]):
        action_key = "open"

    if action_key is None:
        return None

    function_name = {
        "open": "Open_Window",
        "close": "Close_Window",
        "set": "Set_Window",
    }[action_key]
    intent_name = FUNC_TO_NAME.get(
        function_name,
        {
            "open": "\u6253\u5f00\u8f66\u7a97",
            "close": "\u5173\u95ed\u8f66\u7a97",
            "set": "\u8bbe\u7f6e\u8f66\u7a97",
        }[action_key],
    )

    slot_pairs: list[str] = []
    position = _extract_window_position(compact)
    if position:
        slot_pairs.append(f"\u4f4d\u7f6e:{position}")
    if action_key == "set" and ratio is not None:
        slot_pairs.append(f"ratio:{ratio}")
    return f"{intent_name}-" + (",".join(slot_pairs) if slot_pairs else "\u65e0")

async def _call_bound_tools(query: str, tools: list[dict]):
    model = build_chat_model("nlu_tool")
    return await model.bind_tools(tools).ainvoke(
        [
            SystemMessage(content=PROMPTS.NLU_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )


async def _dm_tool_result(namespace: str, function_name: str, query: str, slots: dict) -> dict | None:
    endpoint = SETTINGS.service("dm_tool")
    url = f"{endpoint.base_url}{endpoint.route}/{namespace}"
    async with httpx.AsyncClient(timeout=SETTINGS.runtime.request_timeout_seconds, trust_env=False) as client:
        response = await client.post(url, json={"function": function_name, "query": query, "slots": slots})
        response.raise_for_status()
        data = response.json()
    if data.get("handled"):
        return data.get("tool_response")
    return None


def _normalize_task_slots(function_name: str, query: str, slots: dict[str, str]) -> dict[str, str]:
    normalized = dict(slots)
    if function_name == "Go_POI":
        has_nearby_intent = any(token in query for token in ("最近", "附近", "离我最近", "就近"))
        if has_nearby_intent and not normalized.get("landmark"):
            normalized["landmark"] = "当前位置"
    return normalized


async def _tool_call_to_nlu(query: str) -> str:
    heuristic_nlu = _heuristic_control_nlu(query)
    try:
        result_ids, scores = await asyncio.to_thread(_intent_recall, query)
        if not result_ids:
            return heuristic_nlu or UNKNOWN_NLU_VALUE

        max_score = max(scores) if scores else 0.0
        top1_function = ID_TO_FUNC.get(str(result_ids[0]), "")
        if heuristic_nlu and str(result_ids[0]) == "3" and max_score > SETTINGS.runtime.nlu_unknown_confidence:
            LOGGER.info("NLU control heuristic activated query=%s top1_intent=%s score=%.4f", query, result_ids[0], max_score)
            return heuristic_nlu
        if heuristic_nlu and top1_function in WINDOW_CONTROL_FUNCTIONS and max_score >= 0.9:
            LOGGER.info("NLU control heuristic shortcut query=%s top1_function=%s score=%.4f", query, top1_function, max_score)
            return heuristic_nlu
        if str(result_ids[0]) == "3" and max_score > SETTINGS.runtime.nlu_unknown_confidence:
            return UNKNOWN_NLU_VALUE

        current_tools, top1_tools, top1_raw_tools, ranked_raw_tools = _build_candidate_tools(result_ids)
        if not current_tools:
            return UNKNOWN_NLU_VALUE

        response = await _call_bound_tools(query, current_tools)
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls and len(current_tools) > 1 and top1_tools and max_score >= 0.9:
            LOGGER.info(
                "NLU tool selection retry with merged top1 candidate query=%s top1_intent=%s score=%.4f",
                query,
                result_ids[0],
                max_score,
            )
            response = await _call_bound_tools(query, top1_tools)
            tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls and top1_raw_tools and max_score >= 0.9:
            LOGGER.info(
                "NLU tool selection retry with raw top1 variants query=%s top1_intent=%s variants=%s",
                query,
                result_ids[0],
                len(top1_raw_tools),
            )
            for raw_tool in top1_raw_tools:
                response = await _call_bound_tools(query, [raw_tool])
                tool_calls = getattr(response, "tool_calls", None) or []
                if tool_calls:
                    break

        if not tool_calls and ranked_raw_tools and max_score >= 0.9:
            LOGGER.info(
                "NLU tool selection retry with ranked raw candidates query=%s intent_ids=%s variants=%s",
                query,
                result_ids,
                len(ranked_raw_tools),
            )
            for raw_tool in ranked_raw_tools:
                response = await _call_bound_tools(query, [raw_tool])
                tool_calls = getattr(response, "tool_calls", None) or []
                if tool_calls:
                    break

        if not tool_calls:
            LOGGER.warning(
                "NLU tool selection returned no tool_calls query=%s intent_ids=%s candidate_tools=%s",
                query,
                result_ids,
                [tool.get("function", {}).get("name") for tool in current_tools],
            )
            return heuristic_nlu or UNKNOWN_NLU_VALUE

        first_call = tool_calls[0]
        legacy_tool_call = [
            {
                "function": {
                    "name": first_call.get("name", "Unknown"),
                    "arguments": json.dumps(first_call.get("args", {}), ensure_ascii=False),
                }
            }
        ]
        return SLOT_PROCESS.intent_slot(legacy_tool_call, FUNC_TO_NAME, SLOT_MAP)
    except Exception as exc:
        LOGGER.warning("NLU tool selection failed: %s", exc)
        return heuristic_nlu or UNKNOWN_NLU_VALUE


@app.post(SETTINGS.service("nlu").route)
async def inference(payload: NluRequest) -> dict:
    nlu_value = await _tool_call_to_nlu(payload.query)
    items = nlu_value.split("-") if nlu_value else ["未知", "无"]
    intent = items[0]
    slots_raw = "-".join(items[1:]) if len(items) > 2 else (items[1] if len(items) > 1 else "无")

    slots: dict[str, str] = {}
    if slots_raw and slots_raw != "无":
        for item in slots_raw.split(","):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            slots[key] = value

    intent_id = NAME_TO_ID.get(intent)
    function_name = ID_TO_FUNC.get(intent_id, "Unknown") if intent_id else "Unknown"
    slots = _normalize_task_slots(function_name, payload.query, slots)
    response = {
        "query": payload.query,
        "tarce_id": payload.trace_id,
        "intent": intent,
        "intent_id": intent_id,
        "function": function_name,
        "slots": slots,
    }

    if payload.enable_dm:
        for namespace in ("weather", "music", "maps", "control"):
            try:
                tool_response = await _dm_tool_result(namespace, function_name, payload.query, slots)
            except Exception as exc:
                LOGGER.warning(
                    "DM tool invocation failed namespace=%s function=%s query=%s: %s",
                    namespace,
                    function_name,
                    payload.query,
                    exc,
                )
                continue
            if tool_response is None:
                continue
            response["tool"] = tool_response
            try:
                response["nlg"] = (
                    await post_service_json("nlg", {"query": payload.query, "tool_response": tool_response})
                ).get("data", "")
            except Exception as exc:
                LOGGER.warning("NLG generation failed function=%s query=%s: %s", function_name, payload.query, exc)
                response["nlg"] = ""
            break

    return response


if __name__ == "__main__":
    endpoint = SETTINGS.service("nlu")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
