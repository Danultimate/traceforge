"""TraceForge primary API: async context manager + decorator sugar."""
import functools
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import AsyncGenerator, Generator, Optional, Any
import ulid

from traceforge.span import Span, SpanType, LLMCallData, ToolCallData
from traceforge.trace import Trace, TraceManifest
from traceforge.naming import generate_run_name
from traceforge.pricing import estimate_cost, ModelPrice

__version__ = "0.2.0"


class RunContext:
    """Active run context. Returned by `tracer.run()` and `tracer.run_sync()`.

    Provides the span-recording API for manual instrumentation.
    """

    def __init__(
        self,
        run_id: str,
        run_name: str,
        pricing: Optional[dict[str, ModelPrice]] = None,
    ):
        self.run_id = run_id
        self.run_name = run_name
        self._spans: list[Span] = []
        self._started_at = datetime.utcnow()
        self._pricing = pricing
        self.trace: Optional[Trace] = None

    def record_llm_call(
        self,
        provider: str,
        model: str,
        messages: list[dict],
        response: str,
        system_prompt: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        parent_id: Optional[str] = None,
        state_before: Optional[dict] = None,
        state_after: Optional[dict] = None,
        cost_usd: Optional[float] = None,
    ) -> Span:
        if cost_usd is None:
            cost_usd = estimate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=self._pricing,
            )
        span = Span(
            run_id=self.run_id,
            span_type=SpanType.LLM_CALL,
            name=f"{provider}/{model}",
            parent_id=parent_id,
            state_before=state_before,
            state_after=state_after,
            llm_data=LLMCallData(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                response=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                temperature=temperature,
                seed=seed,
                cost_usd=cost_usd,
            ),
            latency_ms=latency_ms,
        )
        self._spans.append(span)
        return span

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        parent_id: Optional[str] = None,
        state_before: Optional[dict] = None,
        state_after: Optional[dict] = None,
    ) -> Span:
        span = Span(
            run_id=self.run_id,
            span_type=SpanType.TOOL_CALL,
            name=tool_name,
            parent_id=parent_id,
            state_before=state_before,
            state_after=state_after,
            tool_data=ToolCallData(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                latency_ms=latency_ms,
                error=error,
            ),
            latency_ms=latency_ms,
            error=error,
        )
        self._spans.append(span)
        return span

    def record_error(
        self,
        error: str,
        span_name: str = "error",
        parent_id: Optional[str] = None,
    ) -> Span:
        span = Span(
            run_id=self.run_id,
            span_type=SpanType.ERROR,
            name=span_name,
            parent_id=parent_id,
            error=error,
        )
        self._spans.append(span)
        return span

    def custom(
        self,
        name: str,
        metadata: Optional[dict] = None,
        parent_id: Optional[str] = None,
    ) -> Span:
        span = Span(
            run_id=self.run_id,
            span_type=SpanType.CUSTOM,
            name=name,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._spans.append(span)
        return span

    def _finalise(self) -> Trace:
        completed_at = datetime.utcnow()
        duration_ms = int(
            (completed_at - self._started_at).total_seconds() * 1000
        )
        llm_spans = [s for s in self._spans if s.span_type == SpanType.LLM_CALL]
        tool_spans = [s for s in self._spans if s.span_type == SpanType.TOOL_CALL]
        error_spans = [s for s in self._spans if s.span_type == SpanType.ERROR]

        manifest = TraceManifest(
            run_id=self.run_id,
            run_name=self.run_name,
            started_at=self._started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            total_spans=len(self._spans),
            llm_calls=len(llm_spans),
            tool_calls=len(tool_spans),
            errors=len(error_spans),
            total_input_tokens=sum(
                (s.llm_data.input_tokens or 0) for s in llm_spans if s.llm_data
            ),
            total_output_tokens=sum(
                (s.llm_data.output_tokens or 0) for s in llm_spans if s.llm_data
            ),
            total_cost_usd=round(
                sum(
                    (s.llm_data.cost_usd or 0.0)
                    for s in llm_spans if s.llm_data
                ),
                6,
            ),
            traceforge_version=__version__,
        )

        self.trace = Trace(manifest=manifest, spans=self._spans)
        return self.trace


class Tracer:
    """TraceForge tracer.

    Async usage:

        async with tracer.run() as run:
            result = await my_agent(query, _run=run)
        trace = run.trace

    Sync usage:

        with tracer.run_sync() as run:
            result = my_agent(query, _run=run)
        trace = run.trace

    Decorator sugar:

        @tracer.trace
        async def my_agent(query, _run=None):
            ...

    Instrumentors (manual LLM interception):

        from traceforge.integrations.anthropic import AnthropicInstrumentor
        async with tracer.run() as run:
            client = AnthropicInstrumentor(run).instrument(AsyncAnthropic())
    """

    def __init__(
        self,
        auto_save: bool = True,
        pricing: Optional[dict[str, ModelPrice]] = None,
    ):
        self.auto_save = auto_save
        self.pricing = pricing
        self._last_trace: Optional[Trace] = None

    @asynccontextmanager
    async def run(self) -> AsyncGenerator[RunContext, None]:
        run_id = str(ulid.ULID())
        run_name = generate_run_name()
        ctx = RunContext(run_id=run_id, run_name=run_name, pricing=self.pricing)
        try:
            yield ctx
        except Exception as e:
            ctx.record_error(str(e))
            raise
        finally:
            trace = ctx._finalise()
            self._last_trace = trace
            if self.auto_save:
                trace.save()

    @contextmanager
    def run_sync(self) -> Generator[RunContext, None, None]:
        """Synchronous context manager for non-async agents."""
        run_id = str(ulid.ULID())
        run_name = generate_run_name()
        ctx = RunContext(run_id=run_id, run_name=run_name, pricing=self.pricing)
        try:
            yield ctx
        except Exception as e:
            ctx.record_error(str(e))
            raise
        finally:
            trace = ctx._finalise()
            self._last_trace = trace
            if self.auto_save:
                trace.save()

    def trace(self, func):
        """Decorator sugar over the async context manager."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.run() as run:
                return await func(*args, **kwargs, _run=run)
        return wrapper

    def last(self) -> Optional[Trace]:
        return self._last_trace

    async def replay(self, trace: Trace, agent_fn, mode: str = "llm-mock"):
        from traceforge.replay import ReplayEngine
        engine = ReplayEngine(trace=trace, mode=mode)
        return await engine.run(agent_fn)
