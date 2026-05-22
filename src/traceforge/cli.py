"""TraceForge CLI: init, list, open, show."""
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="agentrace-llm", prog_name="traceforge")
def cli():
    """TraceForge — agent runtime tracing and replay."""


@cli.command()
def init():
    """Scaffold traceforge.yaml, agent.py example, .gitignore entry."""
    Path("traceforge.yaml").write_text(
        "auto_save: true\n"
        "store_dir: .traceforge/runs\n"
        "slim: false\n"
    )

    Path("agent.py").write_text(
        '"""TraceForge example agent.\nRun: python agent.py\n"""\n'
        "import asyncio\n"
        "from anthropic import AsyncAnthropic\n"
        "from traceforge import Tracer\n"
        "from traceforge.integrations.anthropic import AnthropicInstrumentor\n"
        "\n"
        "tracer = Tracer()\n"
        "\n"
        "\n"
        "async def main():\n"
        "    async with tracer.run() as run:\n"
        "        client = AnthropicInstrumentor(run).instrument(AsyncAnthropic())\n"
        "        response = await client.messages.create(\n"
        '            model="claude-haiku-4-5-20251001",\n'
        "            max_tokens=256,\n"
        '            system="You are a helpful assistant.",\n'
        '            messages=[{"role": "user", "content": "What is 2 + 2?"}],\n'
        "        )\n"
        "        print(response.content[0].text)\n"
        "\n"
        "    trace = run.trace\n"
        "    trace.print_summary()\n"
        '    print(f"\\nReport saved: .traceforge/runs/<run-id>-<run-name>/report.html")\n'
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    asyncio.run(main())\n"
    )

    gitignore = Path(".gitignore")
    existing = gitignore.read_text() if gitignore.exists() else ""
    if ".traceforge/" not in existing:
        with gitignore.open("a") as f:
            f.write("\n.traceforge/\n")

    click.echo("Created traceforge.yaml")
    click.echo("Created agent.py (example)")
    click.echo("Updated .gitignore")
    click.echo("\nNext: python agent.py")


@cli.command(name="list")
def list_runs():
    """List all local traces."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    from traceforge.storage.file_store import list_traces

    console = Console()
    runs = list_traces()

    if not runs:
        console.print("[dim]No traces found. Run your agent first.[/dim]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Run name")
    table.add_column("Run ID")
    table.add_column("Started")
    table.add_column("Duration")
    table.add_column("Spans", justify="right")
    table.add_column("Errors", justify="right")

    for run in runs:
        table.add_row(
            f"[cyan]{run['run_name']}[/cyan]",
            f"[dim]{run['run_id'][:8]}...[/dim]",
            str(run.get("started_at", ""))[:19],
            f"{run.get('duration_ms', '?')}ms",
            str(run.get("total_spans", "?")),
            f"[red]{run.get('errors', 0)}[/red]" if run.get("errors") else "0",
        )

    console.print(table)


@cli.command(name="open")
@click.argument("run_id_or_name")
def open_cmd(run_id_or_name: str):
    """Open a trace HTML report in the browser."""
    import webbrowser

    from traceforge.storage.file_store import STORE_DIR

    if not STORE_DIR.exists():
        click.echo(f"Trace store {STORE_DIR} does not exist")
        raise SystemExit(1)

    matches = [
        d for d in STORE_DIR.iterdir()
        if d.is_dir() and run_id_or_name in d.name
    ]
    if not matches:
        click.echo(f"No trace found matching {run_id_or_name!r}")
        raise SystemExit(1)

    report = matches[0] / "report.html"
    if not report.exists():
        click.echo(f"Report HTML not found for {matches[0].name}")
        raise SystemExit(1)
    webbrowser.open(f"file://{report.resolve()}")
    click.echo(f"Opening {report}")


@cli.command()
@click.argument("run_id_or_name")
def show(run_id_or_name: str):
    """Print a trace summary to the terminal."""
    from traceforge.storage.file_store import load_trace
    trace = load_trace(run_id_or_name)
    trace.print_summary()


if __name__ == "__main__":
    cli()
