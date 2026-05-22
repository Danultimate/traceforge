from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import ulid


class SpanType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    BRANCH = "branch"
    JOIN = "join"
    ERROR = "error"
    CUSTOM = "custom"


class LLMCallData(BaseModel):
    provider: str
    model: str
    system_prompt: Optional[str] = None
    messages: list[dict]
    response: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    cost_usd: Optional[float] = None


class ToolCallData(BaseModel):
    tool_name: str
    tool_input: Any
    tool_output: Optional[Any] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(ulid.ULID()))
    parent_id: Optional[str] = None
    run_id: str
    span_type: SpanType
    name: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    latency_ms: Optional[int] = None
    state_before: Optional[dict] = None
    state_after: Optional[dict] = None
    llm_data: Optional[LLMCallData] = None
    tool_data: Optional[ToolCallData] = None
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None

    def end(self) -> None:
        self.ended_at = datetime.utcnow()
        if self.started_at:
            self.latency_ms = int(
                (self.ended_at - self.started_at).total_seconds() * 1000
            )
