"""OpenAI AsyncOpenAI instrumentor."""
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from traceforge.tracer import RunContext


class _MockOpenAIResponse:
    def __init__(self, text: str):
        self.choices = [
            type("Choice", (), {
                "message": type("Message", (), {"content": text, "role": "assistant"})(),
                "finish_reason": "stop",
                "index": 0,
            })()
        ]
        self.usage = type("Usage", (), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })()


class OpenAIInstrumentor:
    """Wraps `client.chat.completions.create` on an OpenAI async client."""

    def __init__(self, run: "RunContext", mock_interceptor=None):
        self._run = run
        self._mock = mock_interceptor

    def instrument(self, client):
        original_create = client.chat.completions.create

        async def traced_create(**kwargs):
            messages = kwargs.get("messages", [])

            if self._mock is not None:
                cached = self._mock.get(messages)
                if cached is not None:
                    self._run.record_llm_call(
                        provider="openai",
                        model=kwargs.get("model", "unknown"),
                        messages=messages,
                        response=cached,
                        latency_ms=0,
                        temperature=kwargs.get("temperature"),
                    )
                    return _MockOpenAIResponse(cached)

            start = time.time()
            response = await original_create(**kwargs)
            latency_ms = int((time.time() - start) * 1000)

            response_text: Optional[str] = None
            try:
                response_text = response.choices[0].message.content
            except Exception:
                response_text = str(response)

            prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
            completion_tokens = getattr(
                getattr(response, "usage", None), "completion_tokens", None
            )

            self._run.record_llm_call(
                provider="openai",
                model=kwargs.get("model", "unknown"),
                messages=messages,
                response=response_text,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                latency_ms=latency_ms,
                temperature=kwargs.get("temperature"),
            )
            return response

        client.chat.completions.create = traced_create
        return client
