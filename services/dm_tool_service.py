from __future__ import annotations

import asyncio
import inspect
from datetime import datetime

import httpx
from fastapi import FastAPI
import uvicorn

from test_backend.shared.date_utils import normalize_date
from test_backend.shared.legacy_loader import load_class_mappings
from test_backend.shared.schemas import DmToolRequest, HealthResponse
from test_backend.shared.settings import get_settings

try:
    from qqmusic_api import search as qqmusic_search
except ImportError:
    qqmusic_search = None


SETTINGS = get_settings()
_, FUNC_TO_NAME, _ = load_class_mappings()
CONTROL_EXCLUDED = {"Unknown", "Go_POI", "Search_Music", "Query_Weather", "Query_Timely_Weather"}
ACTION_PREFIXES = (
    ("Open_", "打开"),
    ("Close_", "关闭"),
    ("Set_", "设置"),
    ("Inc_", "调高"),
    ("Dec_", "调低"),
    ("Display_", "查看"),
    ("View_", "查看"),
    ("Ask_", "查询"),
    ("Check_", "检查"),
    ("Reserve_", "预约"),
    ("Cancel_", "取消"),
    ("Play_", "播放"),
    ("Pause_", "暂停"),
    ("Stop_", "停止"),
    ("Search_", "搜索"),
)
ACTION_PREFIX_WORDS = {
    "打开": ("打开", "开启"),
    "关闭": ("关闭",),
    "设置": ("设置",),
    "调高": ("调高", "升高", "升温", "增大", "增加"),
    "调低": ("调低", "降低", "降温", "减小", "减少"),
    "查看": ("查看", "显示"),
    "查询": ("查询",),
    "检查": ("检查",),
    "预约": ("预约",),
    "取消": ("取消",),
    "播放": ("播放",),
    "暂停": ("暂停",),
    "停止": ("停止",),
    "搜索": ("搜索",),
}

app = FastAPI(title="dm-tool-service")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


def _amap_key() -> str:
    return str(SETTINGS.runtime.amap_api_key or "").strip()


async def _resolve_weather_city(raw_location: str, client: httpx.AsyncClient) -> tuple[str, dict]:
    location = str(raw_location or "").strip()
    if not location:
        return str(SETTINGS.runtime.default_city or "").strip(), {}

    response = await client.get(
        "https://restapi.amap.com/v3/place/text",
        params={
            "key": _amap_key(),
            "keywords": location,
            "citylimit": "false",
            "offset": 1,
        },
    )
    response.raise_for_status()
    data = response.json()
    pois = data.get("pois") or []
    if not pois:
        return location, {}

    top_poi = pois[0]
    resolved = str(top_poi.get("adcode") or "").strip() or str(top_poi.get("cityname") or "").strip() or location
    return resolved, {
        "query": location,
        "name": top_poi.get("name", location),
        "cityname": top_poi.get("cityname") or "",
        "adcode": top_poi.get("adcode") or "",
        "address": top_poi.get("address") or "",
    }


def _infer_control_action(function_name: str) -> str:
    for prefix, action in ACTION_PREFIXES:
        if function_name.startswith(prefix):
            return action
    return "执行"


def _normalize_target_label(label: str, action: str) -> str:
    target = label.strip()
    for prefix in ACTION_PREFIX_WORDS.get(action, ()): 
        if target.startswith(prefix):
            target = target[len(prefix):].strip()
            break
    return target or label


def _infer_control_domain(function_name: str) -> str:
    lowered = function_name.lower()
    if any(token in lowered for token in ("air_condition", "cooling", "heating", "wind", "circulation", "defog", "ac")):
        return "climate"
    if any(token in lowered for token in ("window", "sunroof", "door", "trunk", "tailgate", "lock", "unlock")):
        return "body"
    if any(token in lowered for token in ("seat", "massage", "ventilation", "steering")):
        return "seat"
    if any(token in lowered for token in ("light", "lamp", "beam", "fog")):
        return "lighting"
    if any(token in lowered for token in ("player", "media", "radio", "news", "lyrics", "video", "album", "frequency", "music", "k_")):
        return "media"
    if any(token in lowered for token in ("calendar", "flow", "wifi", "bluetooth", "app", "training_camp", "hud", "camera")):
        return "system"
    return "vehicle"


def _control_message(action: str, target: str) -> str:
    if action == "打开":
        return f"已为您开启{target}"
    if action == "关闭":
        return f"已为您关闭{target}"
    if action == "设置":
        return f"已为您设置{target}"
    if action == "调高":
        return f"已为您调高{target}"
    if action == "调低":
        return f"已为您调低{target}"
    if action == "查看":
        return f"已为您打开{target}界面"
    if action == "查询":
        return f"已为您查询{target}"
    if action == "检查":
        return f"已为您检查{target}"
    if action == "预约":
        return f"已为您发起{target}预约"
    if action == "取消":
        return f"已为您取消{target}"
    if action == "播放":
        return f"已为您播放{target}"
    if action == "暂停":
        return f"已为您暂停{target}"
    if action == "停止":
        return f"已为您停止{target}"
    if action == "搜索":
        return f"已为您搜索{target}"
    return f"已为您执行{target}"


async def _weather_tool(payload: DmToolRequest) -> dict:
    if payload.function not in {"Query_Weather", "Query_Timely_Weather"}:
        return {"handled": False}
    if not _amap_key():
        return {"handled": True, "tool_response": {"service": "weather", "error": "AMAP_MAPS_API_KEY is not set."}}

    location = payload.slots.get("city") or SETTINGS.runtime.default_city
    target_date = normalize_date(payload.slots.get("date"))
    async with httpx.AsyncClient(timeout=SETTINGS.runtime.request_timeout_seconds, trust_env=False) as client:
        city, location_meta = await _resolve_weather_city(str(location), client)
        response = await client.get(
            "https://restapi.amap.com/v3/weather/weatherInfo",
            params={"key": _amap_key(), "city": city, "extensions": "all"},
        )
        response.raise_for_status()
        data = response.json()

    forecasts = data.get("forecasts") or []
    if not forecasts:
        return {"handled": True, "tool_response": {"service": "weather", "error": "No forecast data available."}}

    selected = forecasts[0].get("casts", [{}])[0]
    for forecast in forecasts[0].get("casts", []):
        if forecast.get("date") == target_date:
            selected = forecast
            break

    tool_response = {
        "service": "weather",
        "城市": forecasts[0].get("city", location_meta.get("cityname") or location),
        "位置": location_meta.get("name") or location,
        "日期": selected.get("date", target_date),
        "天气": selected.get("dayweather", ""),
        "温度": selected.get("daytemp", ""),
        "风向": selected.get("daywind", ""),
        "风力": selected.get("daypower", ""),
    }
    return {"handled": True, "tool_response": tool_response}


async def _music_tool(payload: DmToolRequest) -> dict:
    if payload.function != "Search_Music":
        return {"handled": False}
    if qqmusic_search is None:
        return {"handled": True, "tool_response": {"service": "music", "error": "qqmusic-api-python is not installed."}}

    keyword_parts = [str(value) for value in payload.slots.values() if value]
    keyword = " ".join(keyword_parts) if keyword_parts else payload.query
    search_func = getattr(qqmusic_search, "search_by_type", None)
    if search_func is None:
        return {"handled": True, "tool_response": {"service": "music", "error": "qqmusic search api is unavailable."}}

    if inspect.iscoroutinefunction(search_func):
        result = await search_func(keyword=keyword, page=1, num=3)
    else:
        result = await asyncio.to_thread(search_func, keyword=keyword, page=1, num=3)
        if inspect.isawaitable(result):
            result = await result

    filtered = []
    for item in result or []:
        filtered.append(
            {
                "id": item.get("id"),
                "mid": item.get("mid"),
                "name": item.get("name"),
                "pmid": item.get("pmid", ""),
                "icon_url": item.get("icon_url", ""),
                "subtitle": item.get("subtitle", ""),
                "time_public": item.get("time_public", ""),
                "title": item.get("title", item.get("name", "")),
            }
        )
    return {"handled": True, "tool_response": filtered}


async def _maps_tool(payload: DmToolRequest) -> dict:
    if payload.function != "Go_POI":
        return {"handled": False}
    if not _amap_key():
        return {"handled": True, "tool_response": {"service": "maps", "error": "AMAP_MAPS_API_KEY is not set."}}

    city = payload.slots.get("city", "")
    keyword = "".join(
        [
            str(payload.slots.get("city", "")),
            str(payload.slots.get("landmark", "")),
            str(payload.slots.get("POI", "")),
        ]
    ) or payload.query

    async with httpx.AsyncClient(timeout=SETTINGS.runtime.request_timeout_seconds, trust_env=False) as client:
        response = await client.get(
            "https://restapi.amap.com/v3/place/text",
            params={
                "key": _amap_key(),
                "keywords": keyword,
                "city": city,
                "citylimit": "false",
            },
        )
        response.raise_for_status()
        data = response.json()

    pois = []
    for item in data.get("pois", [])[:3]:
        pois.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "address": item.get("address"),
                "typecode": item.get("typecode"),
            }
        )
    return {"handled": True, "tool_response": {"service": "maps", "pois": pois}}


async def _control_tool(payload: DmToolRequest) -> dict:
    function_name = payload.function or ""
    if not function_name or function_name in CONTROL_EXCLUDED:
        return {"handled": False}

    if function_name == "Ask_Date":
        now = datetime.now()
        return {
            "handled": True,
            "tool_response": {
                "domain": "system",
                "function": function_name,
                "intent": FUNC_TO_NAME.get(function_name, "日期查询"),
                "日期": now.strftime("%Y-%m-%d"),
                "message": f"今天是{now.strftime('%Y年%m月%d日')}",
            },
        }

    if function_name == "Ask_Weekday":
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday = weekdays[datetime.now().weekday()]
        return {
            "handled": True,
            "tool_response": {
                "domain": "system",
                "function": function_name,
                "intent": FUNC_TO_NAME.get(function_name, "星期查询"),
                "星期": weekday,
                "message": f"今天是{weekday}",
            },
        }

    label = FUNC_TO_NAME.get(function_name, function_name.replace("_", " "))
    action = _infer_control_action(function_name)
    target = _normalize_target_label(label, action)
    tool_response = {
        "domain": _infer_control_domain(function_name),
        "function": function_name,
        "intent": label,
        "action": action,
        "target": target,
        "slots": payload.slots,
        "status": "success",
        "message": _control_message(action, target),
    }
    return {"handled": True, "tool_response": tool_response}


@app.post(SETTINGS.service("dm_tool").route + "/weather")
async def weather_tool(payload: DmToolRequest) -> dict:
    return await _weather_tool(payload)


@app.post(SETTINGS.service("dm_tool").route + "/music")
async def music_tool(payload: DmToolRequest) -> dict:
    return await _music_tool(payload)


@app.post(SETTINGS.service("dm_tool").route + "/maps")
async def maps_tool(payload: DmToolRequest) -> dict:
    return await _maps_tool(payload)


@app.post(SETTINGS.service("dm_tool").route + "/control")
async def control_tool(payload: DmToolRequest) -> dict:
    return await _control_tool(payload)


if __name__ == "__main__":
    endpoint = SETTINGS.service("dm_tool")
    uvicorn.run(app, host=endpoint.host, port=endpoint.port)
