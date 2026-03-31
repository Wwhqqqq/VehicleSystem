from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


TEST_BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TEST_BACKEND_ROOT.parent


class ServiceEndpoint(BaseModel):
    host: str
    port: int
    route: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.route}"


class LegacyPaths(BaseModel):
    legacy_root: str = "test_backend/assets/legacy"
    train_root: str = "test_backend/assets/legacy/train"


class RedisKeys(BaseModel):
    last_service: str
    arbitration_history: str
    rewrite_history: str
    chat_history: str


class RuntimeSettings(BaseModel):
    log_level: str = "INFO"
    redis_url: str = "redis://127.0.0.1:6379/0"
    request_timeout_seconds: float = 15.0
    max_workers: int = 10
    max_history: int = 6
    max_token: int = 2048
    rewrite_history_ttl: int = 40
    arbitration_history_ttl: int = 60
    chat_history_ttl: int = 45
    last_service_ttl: int = 40
    reject_threshold: float = 0.5
    nlu_unknown_confidence: float = 0.98
    default_city: str = "上海"
    chat_flush_every: int = 5
    chat_sentence_pattern: str = r"[，。！？；,.!?;]"
    amap_api_key: str = ""
    legacy_paths: LegacyPaths = Field(default_factory=LegacyPaths)
    redis_keys: RedisKeys


class ModelConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_env: str
    timeout: float
    temperature: float = 0.0
    stream: bool = False
    max_tokens: int | None = None


class Settings(BaseModel):
    runtime: RuntimeSettings
    models: dict[str, ModelConfig]
    services: dict[str, ServiceEndpoint]

    def service(self, name: str) -> ServiceEndpoint:
        return self.services[name]

    def model(self, name: str) -> ModelConfig:
        return self.models[name]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_root = TEST_BACKEND_ROOT / "config"
    runtime_payload = _read_yaml(config_root / "app_settings.yaml")
    models_payload = _read_yaml(config_root / "model_registry.yaml")
    services_payload = _read_yaml(config_root / "service_endpoints.yaml")
    return Settings(
        runtime=RuntimeSettings(**runtime_payload["runtime"]),
        models={name: ModelConfig(**cfg) for name, cfg in models_payload.items()},
        services={name: ServiceEndpoint(**cfg) for name, cfg in services_payload.items()},
    )
