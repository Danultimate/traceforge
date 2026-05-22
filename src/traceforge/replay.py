"""TraceForge replay engine.

Two reproducibility modes:

* `llm-mock` — LLM responses are served from the trace cache; tool calls run
  live (may have side effects).
* `dry-run`  — both LLM responses and tool outputs come from cache; no
  external calls. Tools that validate response format or timestamps may
  still raise.

These are *reproducibility modes*, not "deterministic" guarantees. The agent
function must consult `_mock_llm` (and `_mock_tool` in dry-run) before
calling out, otherwise responses will diverge.
"""
import hashlib
import json
from typing import Any, Callable, Optional

from traceforge.trace import Trace
from traceforge.span import SpanType


class LLMMockInterceptor:
    """Returns LLM responses from the trace cache keyed by message hash."""

    def __init__(self, trace: Trace):
        self._cache: dict[str, str] = {}
        for span in trace.spans:
            if span.span_type == SpanType.LLM_CALL and span.llm_data:
                key = self._make_key(span.llm_data.messages)
                if span.llm_data.response is not None:
                    self._cache[key] = span.llm_data.response

    @staticmethod
    def _make_key(messages: list[dict]) -> str:
        return hashlib.sha256(
            json.dumps(messages, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def get(self, messages: list[dict]) -> Optional[str]:
        return self._cache.get(self._make_key(messages))


class ToolMockInterceptor:
    """Returns tool outputs from the trace cache keyed by (tool, input) hash."""

    def __init__(self, trace: Trace):
        self._cache: dict[str, Any] = {}
        for span in trace.spans:
            if span.span_type == SpanType.TOOL_CALL and span.tool_data:
                key = self._make_key(
                    span.tool_data.tool_name, span.tool_data.tool_input
                )
                self._cache[key] = span.tool_data.tool_output

    @staticmethod
    def _make_key(tool_name: str, tool_input: Any) -> str:
        payload = json.dumps(
            {"tool": tool_name, "input": tool_input},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, tool_name: str, tool_input: Any) -> Any:
        return self._cache.get(self._make_key(tool_name, tool_input))


class ReplayResult:
    def __init__(self, original: Trace, replayed: Trace):
        self.original = original
        self.replayed = replayed

    @property
    def similarity_score(self) -> float:
        """Structural similarity 0.0-1.0 based on span-type sequence."""
        orig_seq = [s.span_type for s in self.original.spans]
        rep_seq = [s.span_type for s in self.replayed.spans]
        if not orig_seq and not rep_seq:
            return 1.0
        matches = sum(a == b for a, b in zip(orig_seq, rep_seq))
        return matches / max(len(orig_seq), len(rep_seq))

    @property
    def diverged(self) -> bool:
        """True if the replay took a different execution path."""
        return self.similarity_score < 0.4

    def print(self) -> None:
        from traceforge.report.terminal import print_replay_result
        print_replay_result(self)


class ReplayEngine:
    def __init__(self, trace: Trace, mode: str = "llm-mock"):
        if mode not in ("llm-mock", "dry-run"):
            raise ValueError(
                f"Unknown replay mode: {mode!r}. Use 'llm-mock' or 'dry-run'."
            )
        self.trace = trace
        self.mode = mode
        self.llm_mock = LLMMockInterceptor(trace)
        self.tool_mock = ToolMockInterceptor(trace) if mode == "dry-run" else None

    async def run(self, agent_fn: Callable) -> ReplayResult:
        from traceforge.tracer import Tracer

        tracer = Tracer(auto_save=False)
        captured: Optional[Any] = None
        async with tracer.run() as run:
            await agent_fn(
                _mock_llm=self.llm_mock,
                _mock_tool=self.tool_mock,
                _run=run,
            )
            captured = run
        replayed = captured.trace if captured and captured.trace else tracer.last()
        return ReplayResult(original=self.trace, replayed=replayed)
