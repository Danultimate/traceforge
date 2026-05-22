from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

from traceforge.trace import Trace
from traceforge.span import SpanType

console = Console()

SPAN_ICONS = {
    SpanType.LLM_CALL: "[LLM]",
    SpanType.TOOL_CALL: "[TOOL]",
    SpanType.MEMORY_READ: "[MEM-R]",
    SpanType.MEMORY_WRITE: "[MEM-W]",
    SpanType.BRANCH: "[BRANCH]",
    SpanType.JOIN: "[JOIN]",
    SpanType.ERROR: "[ERROR]",
    SpanType.CUSTOM: "[CUSTOM]",
}


def _fmt_cost(cost: float) -> str:
    if cost <= 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def print_summary(trace: Trace) -> None:
    m = trace.manifest
    console.print()
    body = (
        f"[bold]TraceForge Run Report[/bold]\n"
        f"[dim]{m.run_id}[/dim]  [cyan]{m.run_name}[/cyan]\n"
        f"Duration: {m.duration_ms}ms  ·  "
        f"Spans: {m.total_spans}  ·  "
        f"LLM calls: {m.llm_calls}  ·  "
        f"Tool calls: {m.tool_calls}\n"
        f"Tokens: {m.total_input_tokens}in / {m.total_output_tokens}out  ·  "
        f"Cost: [green]{_fmt_cost(m.total_cost_usd)}[/green]"
    )
    if m.errors:
        body += f"\n[red]Errors: {m.errors}[/red]"
    console.print(Panel.fit(body, border_style="dim"))
    console.print()

    tree = Tree("[bold]Execution trace[/bold]")
    for span in trace.spans:
        icon = SPAN_ICONS.get(span.span_type, "•")
        label = f"{icon} [bold]{span.name}[/bold]"
        if span.latency_ms is not None:
            label += f" [dim]({span.latency_ms}ms)[/dim]"
        if span.llm_data and span.llm_data.cost_usd:
            label += f" [green]{_fmt_cost(span.llm_data.cost_usd)}[/green]"
        if span.error:
            label += f" [red]ERROR: {span.error[:60]}[/red]"
        tree.add(label)
    console.print(tree)
    console.print()


def print_replay_result(result) -> None:
    score = result.similarity_score
    color = "green" if score >= 0.8 else "yellow" if score >= 0.4 else "red"
    status = "ALIGNED" if not result.diverged else "DIVERGED"

    console.print()
    console.print(Panel.fit(
        f"[bold]Replay Result[/bold]\n"
        f"Similarity: [{color}]{score:.0%}[/{color}]  ·  "
        f"Status: [{color}]{status}[/{color}]\n"
        f"Original spans: {len(result.original.spans)}  ·  "
        f"Replayed spans: {len(result.replayed.spans)}",
        border_style="dim",
    ))

    if result.diverged:
        console.print(
            "[yellow]Traces diverged significantly. "
            "See docs/replay-faq.md for common causes.[/yellow]"
        )
    console.print()
