from __future__ import annotations

from typing import Any

import httpx

from .settings import get_settings


async def post_service_json(service_name: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
    settings = get_settings()
    endpoint = settings.service(service_name)
    request_timeout = timeout or settings.runtime.request_timeout_seconds
    async with httpx.AsyncClient(timeout=request_timeout, trust_env=False) as client:
        response = await client.post(endpoint.url, json=payload)
        response.raise_for_status()
        return response.json()

