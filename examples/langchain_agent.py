"""TraceForge + LangChain — manual instrumentation pattern.

LangChain instrumentation is *manual* in TraceForge: you record each chain
step from your callback handler. See `src/traceforge/integrations/langchain.py`
for the bridge helper.

Run:
    pip install "traceforge-llm[langchain]"
    python examples/langchain_agent.py
"""
import asyncio

from traceforge import Tracer
from traceforge.integrations.langchain import LangChainInstrumentor


async def main():
    tracer = Tracer()

    async with tracer.run() as run:
        instrumentor = LangChainInstrumentor(run)

        # In a real app, plug `instrumentor` into a LangChain BaseCallbackHandler
        # and call these from `on_llm_end`, `on_tool_end`, etc.
        instrumentor.record_llm_step(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            response="4",
            input_tokens=12,
            output_tokens=1,
        )
        instrumentor.record_chain_step(
            step_name="answer_formatter",
            inputs={"raw": "4"},
            outputs={"formatted": "The answer is 4."},
        )

    run.trace.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
