from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from shared.settings import TEST_BACKEND_ROOT, get_settings


BENCHMARK_ROOT = TEST_BACKEND_ROOT / "benchmarks"
DATA_ROOT = BENCHMARK_ROOT / "data"
RESULT_ROOT = BENCHMARK_ROOT / "result"
ASSETS_ROOT = TEST_BACKEND_ROOT / "assets" / "legacy"
SETTINGS = get_settings()
LOCAL_NO_PROXY_ITEMS = ("127.0.0.1", "localhost")


def _merge_no_proxy(existing: str | None) -> str:
    current = [item.strip() for item in (existing or "").split(",") if item.strip()]
    for item in LOCAL_NO_PROXY_ITEMS:
        if item not in current:
            current.append(item)
    return ",".join(current)


def ensure_local_no_proxy() -> None:
    os.environ["NO_PROXY"] = _merge_no_proxy(os.environ.get("NO_PROXY"))
    os.environ["no_proxy"] = _merge_no_proxy(os.environ.get("no_proxy"))


def result_path(name: str) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULT_ROOT / name


def benchmark_data_path(name: str) -> Path:
    return DATA_ROOT / name


def asset_data_path(*parts: str) -> Path:
    path = ASSETS_ROOT
    for item in parts:
        path = path / item
    return path


def service_url(name: str) -> str:
    return SETTINGS.service(name).url


def service_base_url(name: str) -> str:
    return SETTINGS.service(name).base_url


def service_route(name: str) -> str:
    return SETTINGS.service(name).route


def random_trace_id(prefix: str = "bench") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def load_tab_samples(path: Path) -> list[tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(tuple(item.strip() for item in line.split("\t")))
        return rows


def load_json_lines(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
