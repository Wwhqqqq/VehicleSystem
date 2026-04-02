from __future__ import annotations

import asyncio
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from shared.langchain_utils import build_chat_model, message_text
from shared.legacy_models import get_reject_predictor

REJECT_LLM_SYSTEM = (
    "你是车载语音助手的「拒识」判别器。"
    "若用户语句是有效的车控指令、导航/媒体/天气等车机能力范围内的请求、或正常的知识/闲聊问题，输出1。"
    "若语句无意义、乱码、过短无效、纯语气词、或明显不适合车机应答的胡言乱语，输出0。"
    "只输出一个字符：0 或 1，不要输出其它文字。"
)


def _parse_binary(text: str) -> int | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    match = re.search(r"[01]", stripped)
    if not match:
        return None
    return int(match.group(0))


def _est_tokens_zh(text: str) -> int:
    return max(1, len(text) // 2)


@dataclass
class StepTrace:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    used_llm_override: bool = False


class RejectStrategy(ABC):
    name: str

    @abstractmethod
    async def predict(self, query: str, *, threshold: float, history: str = "") -> tuple[int, StepTrace]:
        raise NotImplementedError


class BertOnlyRejectStrategy(RejectStrategy):
    name = "bert_only"

    async def predict(self, query: str, *, threshold: float, history: str = "") -> tuple[int, StepTrace]:
        del history
        trace = StepTrace()

        def _run():
            return get_reject_predictor().predict_binary(query, threshold)

        pred, _prob = await asyncio.to_thread(_run)
        return int(pred), trace


class FullLlmRejectStrategy(RejectStrategy):
    name = "full_llm"

    def __init__(self, model_profile: str) -> None:
        self._model_profile = model_profile

    async def predict(self, query: str, *, threshold: float, history: str = "") -> tuple[int, StepTrace]:
        del threshold
        trace = StepTrace()
        if history.strip():
            user = f"对话上文（仅供参考）：\n{history.strip()}\n当前用户说：{query}\n请结合上下文输出 0 或 1。"
        else:
            user = f"用户说：{query}\n请输出 0 或 1。"
        trace.prompt_tokens = _est_tokens_zh(REJECT_LLM_SYSTEM) + _est_tokens_zh(user)
        trace.llm_calls = 1
        trace.completion_tokens = 2
        try:
            response = await build_chat_model(self._model_profile).ainvoke(
                [SystemMessage(content=REJECT_LLM_SYSTEM), HumanMessage(content=user)]
            )
            parsed = _parse_binary(message_text(response))
            pred = parsed if parsed is not None else 1
        except Exception:
            pred = 1
            trace.used_llm_override = True
        return pred, trace


class BertPrescreenLlmDeepRejectStrategy(RejectStrategy):
    """BERT front filter; LLM only when confidence is near the threshold (deep disambiguation)."""

    name = "bert_prescreen_llm_deep"

    def __init__(self, model_profile: str, margin: float) -> None:
        self._model_profile = model_profile
        self._margin = margin

    async def predict(self, query: str, *, threshold: float, history: str = "") -> tuple[int, StepTrace]:
        trace = StepTrace()

        def _bert():
            return get_reject_predictor().predict_binary(query, threshold)

        pred_bert, prob = await asyncio.to_thread(_bert)
        if abs(float(prob) - float(threshold)) <= self._margin:
            if history.strip():
                user = (
                    f"对话上文：\n{history.strip()}\n当前用户说：{query}\n"
                    "BERT 分数接近阈值，请结合上下文做最终裁决。只输出 0 或 1。"
                )
            else:
                user = f"用户说：{query}\nBERT 分数接近阈值，请你做最终裁决。只输出 0 或 1。"
            trace.prompt_tokens = _est_tokens_zh(REJECT_LLM_SYSTEM) + _est_tokens_zh(user)
            trace.llm_calls = 1
            trace.completion_tokens = 2
            try:
                response = await build_chat_model(self._model_profile).ainvoke(
                    [SystemMessage(content=REJECT_LLM_SYSTEM), HumanMessage(content=user)]
                )
                parsed = _parse_binary(message_text(response))
                pred = parsed if parsed is not None else int(pred_bert)
            except Exception:
                pred = int(pred_bert)
                trace.used_llm_override = True
            return pred, trace

        return int(pred_bert), trace


def default_llm_profile() -> str:
    return os.getenv("BERT_VALUE_LLM_PROFILE", "correlation")
