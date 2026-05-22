"""LangChain instrumentation — manual.

Auto-patching LangChain runnables / chains is fragile across versions, so
TraceForge ships a *manual* helper: you call `record_chain_step` from your
LangChain callback handler (or anywhere you have a `RunContext`).

Example, inside a `BaseCallbackHandler.on_llm_end`:

    from traceforge.integrations.langchain import LangChainInstrumentor

    instrumentor = LangChainInstrumentor(run)
    instrumentor.record_chain_step(
        step_name="my_chain.llm_step",
        inputs={"prompt": prompt},
        outputs={"text": llm_result.generations[0][0].text},
    )

No `langchain` import is required at module load — keeping this file safe to
import even when the optional `langchain` dependency is missing.
"""
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from traceforge.tracer import RunContext


class LangChainInstrumentor:
    """Manual LangChain bridge.

    Constructor signature matches the other instrumentors so users can swap
    them without learning a new API.
    """

    def __init__(self, run: "RunContext", mock_interceptor=None):
        self._run = run
        self._mock = mock_interceptor

    def record_chain_step(
        self,
        step_name: str,
        inputs: Any,
        outputs: Any = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """Record one LangChain step as a tool-call span.

        We model chain steps as TOOL_CALL spans rather than LLM_CALL because
        a single LangChain chain step may aggregate multiple LLM calls plus
        local logic.
        """
        return self._run.record_tool_call(
            tool_name=step_name,
            tool_input=inputs,
            tool_output=outputs,
            latency_ms=latency_ms,
            error=error,
        )

    def record_llm_step(
        self,
        model: str,
        messages: list[dict],
        response: str,
        provider: str = "langchain",
        **kwargs,
    ):
        """Record one underlying LLM call from inside a LangChain callback."""
        return self._run.record_llm_call(
            provider=provider,
            model=model,
            messages=messages,
            response=response,
            **kwargs,
        )
