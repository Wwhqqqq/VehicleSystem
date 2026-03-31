from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from .settings import PROJECT_ROOT, get_settings


def _module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_legacy_paths() -> None:
    settings = get_settings()
    legacy_root = PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root
    train_root = PROJECT_ROOT / settings.runtime.legacy_paths.train_root
    for path in (PROJECT_ROOT, legacy_root, train_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    import os

    original = Path.cwd()
    try:
        os.chdir(str(path))
        yield
    finally:
        os.chdir(str(original))


@lru_cache(maxsize=1)
def load_prompts_module():
    settings = get_settings()
    return _module_from_path(
        "legacy_prompts",
        PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root / "prompts.py",
    )


@lru_cache(maxsize=1)
def load_slot_process_module():
    settings = get_settings()
    return _module_from_path(
        "legacy_slot_process",
        PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root / "function_call" / "slot_process.py",
    )


@lru_cache(maxsize=1)
def load_tool_module():
    settings = get_settings()
    return _module_from_path(
        "legacy_function_tools",
        PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root / "function_call" / "function.py",
    )


@lru_cache(maxsize=1)
def load_class_mappings() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    settings = get_settings()
    id_to_func: dict[str, str] = {}
    func_to_name: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    class_file = PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root / "config" / "class.txt"
    with class_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            intent_id, intent_name, function_name = line.split(":")
            id_to_func[intent_id] = function_name
            func_to_name[function_name] = intent_name
            name_to_id[intent_name] = intent_id
    return id_to_func, func_to_name, name_to_id


@lru_cache(maxsize=1)
def load_slot_intent_map() -> dict:
    settings = get_settings()
    slot_path = PROJECT_ROOT / settings.runtime.legacy_paths.legacy_root / "config" / "slot_intent.json"
    with slot_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tool_specs() -> list[dict]:
    return list(load_tool_module().tools1)
