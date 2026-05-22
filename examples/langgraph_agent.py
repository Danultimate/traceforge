"""TraceForge + LangGraph — manual instrumentation pattern.

LangGraph nodes are async functions — wrap each node body in a
`run.record_tool_call(...)` / `run.record_llm_call(...)` call. The graph
runner itself is not instrumented; you record each node's execution.

Run:
    pip install "agentrace-llm[langgraph]"
    python examples/langgraph_agent.py
"""
import asyncio

from traceforge import Tracer


async def main():
    tracer = Tracer()

    async with tracer.run() as run:
        # node 1: a fake LLM call
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "classify: cat"}],
            response="animal",
            input_tokens=8,
            output_tokens=1,
            latency_ms=240,
        )
        # node 2: a routing decision modeled as a branch span
        run.custom("router.branch", metadata={"taken": "animal_handler"})
        # node 3: tool that handles the chosen branch
        run.record_tool_call(
            tool_name="animal_handler",
            tool_input={"label": "animal"},
            tool_output={"reply": "I see an animal."},
            latency_ms=12,
        )

    run.trace.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
