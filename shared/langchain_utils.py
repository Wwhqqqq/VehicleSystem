from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

from .settings import ModelConfig, get_settings


def _normalize_api_key(raw_value: str | None) -> str:
    if not raw_value:
        return "EMPTY_API_KEY"
    if raw_value.startswith("Bearer "):
        return raw_value.split(" ", 1)[1]
    return raw_value


def _normalize_base_url(raw_value: str, model_id: str) -> str:
    normalized = raw_value.rstrip("/")
    if normalized.endswith("/bots/chat/completions"):
        return normalized[: -len("/chat/completions")]
    if normalized.endswith("/chat/completions"):
        return normalized[: -len("/chat/completions")]
    if model_id.startswith("bot-") and not normalized.endswith("/bots"):
        return f"{normalized}/bots"
    return normalized


def _looks_like_literal_secret(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("sk-", "Bearer ")):
        return True
    if any(char in stripped for char in ("-", "/", "+", "=")) and len(stripped) >= 16:
        return True
    return len(stripped) >= 24 and not stripped.replace("_", "").isalnum()


def _resolve_secret(secret_or_env: str) -> str | None:
    env_value = os.getenv(secret_or_env)
    if env_value:
        return env_value
    if _looks_like_literal_secret(secret_or_env):
        return secret_or_env
    return None


def _build_model_config(name: str) -> tuple[ModelConfig, str, str]:
    settings = get_settings()
    config = settings.model(name)
    api_key = _normalize_api_key(_resolve_secret(config.api_key_env))
    base_url = _normalize_base_url(os.getenv(f"{name.upper()}_BASE_URL", config.base_url), config.model)
    return config, api_key, base_url


@lru_cache(maxsize=16)
def build_chat_model(name: str, streaming: bool | None = None) -> ChatOpenAI:
    config, api_key, base_url = _build_model_config(name)
    return ChatOpenAI(
        model=config.model,
        api_key=api_key,
        base_url=base_url,
        timeout=config.timeout,
        temperature=config.temperature,
        streaming=config.stream if streaming is None else streaming,
        max_tokens=config.max_tokens,
    )


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)
