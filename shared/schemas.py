from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    health: str = "healthy"


class ArbitrationRequest(BaseModel):
    query: str
    sender_id: str = "test"


class RewriteRequest(BaseModel):
    query: str
    last_answer: str = ""
    sender_id: str = "test"


class RejectRequest(BaseModel):
    query: str
    thres: float = 0.5
    trace_id: str = "1"


class IntentRequest(BaseModel):
    query: str
    trace_id: str = "1"


class CorrelationRequest(BaseModel):
    query: str
    sender_id: str = "test"


class ChatRequest(BaseModel):
    query: str
    sender_id: str = "test"
    multiturn: bool = True


class NlgRequest(BaseModel):
    query: str
    tool_response: Any


class DmToolRequest(BaseModel):
    function: str
    query: str
    slots: dict[str, Any] = Field(default_factory=dict)


class NluRequest(BaseModel):
    query: str
    trace_id: str = "1"
    enable_dm: bool = True
