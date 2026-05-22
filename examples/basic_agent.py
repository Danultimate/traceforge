"""TraceForge quickstart — raw Anthropic client.

Install:
    pip install "agentrace-llm[anthropic]"
    export ANTHROPIC_API_KEY=sk-ant-...
Run:
    python examples/basic_agent.py
"""
import asyncio

from anthropic import AsyncAnthropic

from traceforge import Tracer
from traceforge.integrations.anthropic import AnthropicInstrumentor

tracer = Tracer()


async def research_agent(query: str, _run=None, _mock_llm=None, _mock_tool=None):
    """Two-step research agent: plan, then answer."""
    client = AnthropicInstrumentor(_run, mock_interceptor=_mock_llm).instrument(
        AsyncAnthropic()
    )

    plan_response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are a research planner. Output a 3-step research plan as JSON.",
        messages=[{"role": "user", "content": f"Plan research for: {query}"}],
    )
    plan = plan_response.content[0].text
    _run.record_tool_call("parse_plan", {"raw": plan}, {"steps": 3})

    result_response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="You are a research analyst. Answer concisely with citations.",
        messages=[
            {"role": "user", "content": f"Research: {query}\nPlan: {plan}"}
        ],
    )

    return result_response.content[0].text


async def main():
    query = "What are the main approaches to LLM agent observability?"

    async with tracer.run() as run:
        result = await research_agent(query, _run=run)
        print(result)

    trace = run.trace
    trace.print_summary()

    print("\n--- Replaying with mocked LLM responses ---")
    replay_result = await tracer.replay(
        trace=trace,
        agent_fn=lambda **kwargs: research_agent(query, **kwargs),
        mode="llm-mock",
    )
    replay_result.print()


if __name__ == "__main__":
    asyncio.run(main())
