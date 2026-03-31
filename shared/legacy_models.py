from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from test_backend.shared.legacy_loader import ensure_legacy_paths, working_directory
from test_backend.shared.settings import PROJECT_ROOT, get_settings


CLS = "[CLS]"


def _ensure_legacy_optional_dependencies() -> None:
    if "boto3" not in sys.modules and importlib.util.find_spec("boto3") is None:
        sys.modules["boto3"] = types.ModuleType("boto3")

    if "botocore" not in sys.modules and importlib.util.find_spec("botocore") is None:
        botocore_module = types.ModuleType("botocore")
        exceptions_module = types.ModuleType("botocore.exceptions")

        class ClientError(Exception):
            pass

        exceptions_module.ClientError = ClientError
        botocore_module.exceptions = exceptions_module
        sys.modules["botocore"] = botocore_module
        sys.modules["botocore.exceptions"] = exceptions_module


@contextmanager
def _force_utf8_open():
    original_open = builtins.open

    def patched_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = patched_open
    try:
        yield
    finally:
        builtins.open = original_open


def _load_module(module_name: str, path: Path):
    _ensure_legacy_optional_dependencies()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyBertPredictor:
    def __init__(self, dataset: str, model_file: str) -> None:
        ensure_legacy_paths()
        settings = get_settings()
        train_root = PROJECT_ROOT / settings.runtime.legacy_paths.train_root
        module_path = train_root / "models" / model_file
        module = _load_module(f"{dataset}_{module_path.stem}_module", module_path)
        with working_directory(train_root), _force_utf8_open():
            config = module.Config(dataset)
            model = module.Model(config).to(config.device)
            state_dict = torch.load(config.save_path, map_location=config.device)
            model.load_state_dict(state_dict)
        model.eval()
        self.config = config
        self.model = model

    def _encode(self, query: str):
        token = self.config.tokenizer.tokenize(query)
        token = [CLS] + token
        seq_len = len(token)
        token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
        if len(token) < self.config.pad_size:
            mask = [1] * len(token_ids) + [0] * (self.config.pad_size - len(token))
            token_ids += [0] * (self.config.pad_size - len(token))
        else:
            mask = [1] * self.config.pad_size
            token_ids = token_ids[: self.config.pad_size]
            seq_len = self.config.pad_size
        x = torch.LongTensor([token_ids]).to(self.config.device)
        seq = torch.LongTensor([seq_len]).to(self.config.device)
        mask_tensor = torch.LongTensor([mask]).to(self.config.device)
        return x, seq, mask_tensor

    def predict_binary(self, query: str, threshold: float) -> tuple[int, float]:
        with torch.no_grad():
            encoded = self._encode(query)
            output = self.model(encoded)
            prob = float(F.softmax(output, dim=-1).cpu().numpy()[0][1])
            return (1 if prob > threshold else 0), prob

    def predict_topk(self, query: str, topk: int) -> tuple[list[int], list[float]]:
        with torch.no_grad():
            encoded = self._encode(query)
            output = self.model(encoded)
            prob = F.softmax(output, dim=-1).cpu().numpy()[0]
            indices = np.argsort(-prob)[:topk]
            return indices.tolist(), prob[indices].tolist()


@lru_cache(maxsize=1)
def get_reject_predictor() -> LegacyBertPredictor:
    return LegacyBertPredictor(dataset="reject", model_file="bert_tiny.py")


@lru_cache(maxsize=1)
def get_intent_predictor() -> LegacyBertPredictor:
    return LegacyBertPredictor(dataset="intent", model_file="bert.py")


