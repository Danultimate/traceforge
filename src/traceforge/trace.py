from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from traceforge.span import Span, SpanType


class TraceManifest(BaseModel):
    run_id: str
    run_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    total_spans: int
    llm_calls: int
    tool_calls: int
    errors: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float = 0.0
    traceforge_version: str
    schema_version: str = "v1"


class Trace(BaseModel):
    manifest: TraceManifest
    spans: list[Span] = Field(default_factory=list)

    @property
    def llm_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == SpanType.LLM_CALL]

    @property
    def tool_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == SpanType.TOOL_CALL]

    @property
    def error_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == SpanType.ERROR]

    def has_span(
        self,
        *,
        name: Optional[str] = None,
        span_type: Optional[SpanType] = None,
    ) -> bool:
        """Return True iff at least one span matches the given filter."""
        return self.find_span(name=name, span_type=span_type) is not None

    def find_span(
        self,
        *,
        name: Optional[str] = None,
        span_type: Optional[SpanType] = None,
    ) -> Optional[Span]:
        """Return the first span matching the filter, or None."""
        for span in self.spans:
            if name is not None and span.name != name:
                continue
            if span_type is not None and span.span_type != span_type:
                continue
            return span
        return None

    def find_spans(
        self,
        *,
        name: Optional[str] = None,
        span_type: Optional[SpanType] = None,
    ) -> list[Span]:
        """Return every span matching the filter."""
        out = []
        for span in self.spans:
            if name is not None and span.name != name:
                continue
            if span_type is not None and span.span_type != span_type:
                continue
            out.append(span)
        return out

    def print_summary(self) -> None:
        from traceforge.report.terminal import print_summary
        print_summary(self)

    def to_html(self) -> str:
        from traceforge.report.html_report import to_html
        return to_html(self)

    def to_jsonl(self) -> str:
        from traceforge.report.jsonl import to_jsonl
        return to_jsonl(self)

    def save(self, path: str = None):
        from traceforge.storage.file_store import save_trace
        return save_trace(self, path)
