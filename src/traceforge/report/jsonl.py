"""Replayable JSONL writer.

Line 0: manifest (with `__type__: manifest`).
Line 1+: spans (with `__type__: span`).

Round-trips through `traceforge.storage.file_store.load_trace`-compatible
parsing if you split lines yourself.
"""
import json

from traceforge.trace import Trace


def to_jsonl(trace: Trace) -> str:
    lines: list[str] = []

    manifest_obj = json.loads(trace.manifest.model_dump_json())
    manifest_obj["__type__"] = "manifest"
    lines.append(json.dumps(manifest_obj, default=str))

    for span in trace.spans:
        span_obj = json.loads(span.model_dump_json())
        span_obj["__type__"] = "span"
        lines.append(json.dumps(span_obj, default=str))

    return "\n".join(lines) + "\n"
