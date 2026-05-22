import json
from pathlib import Path
from typing import Optional, Union

STORE_DIR = Path(".traceforge/runs")


def save_trace(trace, base_path: Optional[Union[str, Path]] = None) -> Path:
    store = Path(base_path) if base_path else STORE_DIR
    run_dir = store / f"{trace.manifest.run_id}-{trace.manifest.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    (run_dir / "manifest.json").write_text(
        trace.manifest.model_dump_json(indent=2)
    )

    # run.jsonl
    jsonl_path = run_dir / "run.jsonl"
    with jsonl_path.open("w") as f:
        for span in trace.spans:
            f.write(span.model_dump_json() + "\n")

    # report.html (self-contained)
    try:
        html = trace.to_html()
        (run_dir / "report.html").write_text(html)
    except Exception:
        # HTML rendering should never block saving the trace.
        pass

    return run_dir


def load_trace(run_id_or_name: str, base_path: Optional[Union[str, Path]] = None):
    from traceforge.span import Span
    from traceforge.trace import Trace, TraceManifest

    store = Path(base_path) if base_path else STORE_DIR
    if not store.exists():
        raise FileNotFoundError(f"Trace store {store} does not exist")

    matches = [
        d for d in store.iterdir()
        if d.is_dir() and (
            d.name.startswith(run_id_or_name) or run_id_or_name in d.name
        )
    ]
    if not matches:
        raise FileNotFoundError(
            f"No trace found matching {run_id_or_name!r} in {store}"
        )
    if len(matches) > 1:
        names = [d.name for d in matches]
        raise ValueError(
            f"Ambiguous trace ID {run_id_or_name!r}. Matches: {names}"
        )

    run_dir = matches[0]
    manifest = TraceManifest.model_validate_json(
        (run_dir / "manifest.json").read_text()
    )
    spans = []
    for line in (run_dir / "run.jsonl").read_text().splitlines():
        if line.strip():
            spans.append(Span.model_validate_json(line))

    return Trace(manifest=manifest, spans=spans)


def list_traces(base_path: Optional[Union[str, Path]] = None) -> list[dict]:
    store = Path(base_path) if base_path else STORE_DIR
    if not store.exists():
        return []
    runs = []
    for run_dir in sorted(store.iterdir(), reverse=True):
        if (run_dir / "manifest.json").exists():
            manifest = json.loads((run_dir / "manifest.json").read_text())
            runs.append(manifest)
    return runs
