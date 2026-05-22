"""Anthropic AsyncAnthropic instrumentor."""
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from traceforge.tracer import RunContext


class _MockAnthropicResponse:
    """Minimal mock Anthropic response for replay mode."""

    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text, "type": "text"})()]
        self.usage = type("Usage", (), {"input_tokens": 0, "output_tokens": 0})()
        self.stop_reason = "end_turn"


class AnthropicInstrumentor:
    """Wraps `client.messages.create` on an Anthropic async client.

    Usage:
        async with tracer.run() as run:
            instrumentor = AnthropicInstrumentor(run)
            client = instrumentor.instrument(AsyncAnthropic())
            # use client normally — every call is traced
    """

    def __init__(self, run: "RunContext", mock_interceptor=None):
        self._run = run
        self._mock = mock_interceptor

    def instrument(self, client):
        original_create = client.messages.create

        async def traced_create(**kwargs):
            messages = kwargs.get("messages", [])

            if self._mock is not None:
                cached = self._mock.get(messages)
                if cached is not None:
                    self._run.record_llm_call(
                        provider="anthropic",
                        model=kwargs.get("model", "unknown"),
                        messages=messages,
                        response=cached,
                        system_prompt=kwargs.get("system"),
                        latency_ms=0,
                        temperature=kwargs.get("temperature"),
                    )
                    return _MockAnthropicResponse(cached)

            start = time.time()
            response = await original_create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            response_text: Optional[str] = None
            try:
                response_text = response.content[0].text
            except Exception:
                response_text = str(response)

            input_tokens = getattr(getattr(response, "usage", None), "input_tokens", None)
            output_tokens = getattr(getattr(response, "usage", None), "output_tokens", None)

            self._run.record_llm_call(
                provider="anthropic",
                model=kwargs.get("model", "unknown"),
                messages=messages,
                response=response_text,
                system_prompt=kwargs.get("system"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                temperature=kwargs.get("temperature"),
            )
            return response

        client.messages.create = traced_create
        return client
