"""Render a representative terminal summary as an inline SVG.

Run: .venv/bin/python scripts/make_terminal_svg.py
Output: docs/terminal-report.svg
"""
import asyncio
import os

from rich.console import Console

from traceforge import Tracer
import traceforge.report.terminal as tt

# Replace the module-level console with a recording one
rec = Console(record=True, width=96, file=open(os.devnull, "w"))
tt.console = rec


async def main():
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "plan multi-step research"}],
            response="ok",
            input_tokens=1240,
            output_tokens=380,
            latency_ms=842,
        )
        run.record_tool_call(
            "parse_plan_json",
            tool_input={"raw": "..."},
            tool_output={"steps": 3},
            latency_ms=1,
        )
        run.custom("phase.research", metadata={"step": 1, "parallel": True})
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "list framework strengths"}],
            response="LangSmith has strong UI but vendor lock-in...",
            input_tokens=620,
            output_tokens=240,
            latency_ms=190,
        )
        run.record_tool_call(
            "web_search",
            tool_input={"query": "OpenLLMetry trace schema"},
            tool_output=None,
            latency_ms=2400,
            error="Connection timeout after 2.4s",
        )
        run.record_llm_call(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "synthesise findings"}],
            response="Across the four frameworks, replay is the common gap...",
            input_tokens=420,
            output_tokens=140,
            latency_ms=520,
        )
    run.trace.print_summary()


asyncio.run(main())
rec.save_svg("docs/terminal-report.svg", title="traceforge show true-elk")
print(f"Wrote docs/terminal-report.svg ({os.path.getsize('docs/terminal-report.svg')} bytes)")
