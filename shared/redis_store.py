from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .logging import get_logger
from .settings import get_settings

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None


LOGGER = get_logger("test_backend.redis")


@dataclass
class _MemoryEntry:
    value: Any
    expires_at: float | None


class _MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, _MemoryEntry] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        expires_at = time.time() + ex if ex else None
        with self._lock:
            self._data[key] = _MemoryEntry(value=value, expires_at=expires_at)
        return True

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at and time.time() > entry.expires_at:
                del self._data[key]
                return None
            return entry.value


class RedisStateStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._memory = _MemoryStore()
        self._client = None
        if redis_lib is None:
            LOGGER.warning("redis package not installed, using in-memory state store.")
            return
        try:
            self._client = redis_lib.Redis.from_url(settings.runtime.redis_url, decode_responses=True)
            self._client.ping()
        except Exception as exc:
            LOGGER.warning("Redis unavailable, using in-memory state store: %s", exc)
            self._client = None

    def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        if self._client is None:
            return self._memory.set(key, value, ex=ex)
        try:
            return bool(self._client.set(key, value, ex=ex))
        except Exception as exc:
            LOGGER.warning("Redis set failed, falling back to memory: %s", exc)
            return self._memory.set(key, value, ex=ex)

    def get(self, key: str) -> Any:
        if self._client is None:
            return self._memory.get(key)
        try:
            return self._client.get(key)
        except Exception as exc:
            LOGGER.warning("Redis get failed, falling back to memory: %s", exc)
            return self._memory.get(key)
