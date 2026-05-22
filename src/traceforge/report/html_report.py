"""Self-contained HTML report. No CDN, no JS frameworks, no server."""
import html as _html
import json

from traceforge.trace import Trace
from traceforge.span import SpanType


SPAN_ICONS = {
    SpanType.LLM_CALL: "LLM",
    SpanType.TOOL_CALL: "TOOL",
    SpanType.MEMORY_READ: "MEM-R",
    SpanType.MEMORY_WRITE: "MEM-W",
    SpanType.BRANCH: "BRANCH",
    SpanType.JOIN: "JOIN",
    SpanType.ERROR: "ERROR",
    SpanType.CUSTOM: "CUSTOM",
}

SPAN_COLORS = {
    SpanType.LLM_CALL: "#4f46e5",
    SpanType.TOOL_CALL: "#0891b2",
    SpanType.MEMORY_READ: "#65a30d",
    SpanType.MEMORY_WRITE: "#65a30d",
    SpanType.BRANCH: "#a855f7",
    SpanType.JOIN: "#a855f7",
    SpanType.ERROR: "#dc2626",
    SpanType.CUSTOM: "#64748b",
}


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0b1020; color: #e2e8f0; margin: 0; padding: 32px; }
.container { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
.stats { display: flex; gap: 16px; flex-wrap: wrap; background: #111a35;
         border: 1px solid #1e293b; border-radius: 8px;
         padding: 16px; margin-bottom: 24px; }
.stat { display: flex; flex-direction: column; min-width: 110px; }
.stat .label { color: #94a3b8; font-size: 11px;
               text-transform: uppercase; letter-spacing: 0.05em; }
.stat .value { font-size: 18px; font-weight: 600; margin-top: 2px; }
.error-banner { background: #450a0a; border: 1px solid #991b1b;
                color: #fecaca; padding: 12px 16px; border-radius: 6px;
                margin-bottom: 16px; }
.span { background: #111a35; border: 1px solid #1e293b;
        border-left: 4px solid #4f46e5; border-radius: 6px;
        padding: 14px 16px; margin-bottom: 10px; }
.span.has-error { border-left-color: #dc2626; }
.span-header { display: flex; align-items: center; gap: 8px; }
.tag { font-size: 10px; padding: 2px 6px; border-radius: 3px;
       background: #1e293b; color: #94a3b8;
       text-transform: uppercase; letter-spacing: 0.05em; }
.name { font-weight: 600; color: #f1f5f9; }
.latency { color: #94a3b8; font-size: 12px; margin-left: auto; }
.detail { margin-top: 10px; font-size: 13px; color: #cbd5e1; }
.detail .row { margin: 4px 0; }
.detail .row .k { color: #94a3b8; display: inline-block; min-width: 92px; }
pre { background: #050816; border: 1px solid #1e293b; border-radius: 4px;
      padding: 10px 12px; color: #cbd5e1; font-size: 12px;
      overflow-x: auto; white-space: pre-wrap; word-break: break-word;
      max-height: 280px; overflow-y: auto; margin: 6px 0 0; }
.err-text { color: #fca5a5; }
details > summary { cursor: pointer; color: #94a3b8; font-size: 12px;
                    margin-top: 8px; }
footer { color: #64748b; font-size: 11px; text-align: center;
         margin-top: 32px; }
"""


def _esc(value) -> str:
    if value is None:
        return ""
    return _html.escape(str(value))


def _fmt_cost(cost) -> str:
    if not cost or cost <= 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _truncate(text, limit: int = 1500) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n… [truncated, {len(s) - limit} more chars]"


def _format_messages(messages) -> str:
    try:
        return json.dumps(messages, indent=2, default=str)
    except Exception:
        return str(messages)


def _render_span(span) -> str:
    icon = SPAN_ICONS.get(span.span_type, "•")
    color = SPAN_COLORS.get(span.span_type, "#4f46e5")
    has_error = bool(span.error) or span.span_type == SpanType.ERROR
    latency = f"{span.latency_ms}ms" if span.latency_ms is not None else "—"

    detail_parts: list[str] = []

    if span.error:
        detail_parts.append(
            f"<div class='row err-text'><span class='k'>error</span>{_esc(span.error)}</div>"
        )

    if span.llm_data is not None:
        d = span.llm_data
        detail_parts.append(
            f"<div class='row'><span class='k'>provider</span>{_esc(d.provider)}</div>"
            f"<div class='row'><span class='k'>model</span>{_esc(d.model)}</div>"
            f"<div class='row'><span class='k'>tokens</span>"
            f"{_esc(d.input_tokens)} in / {_esc(d.output_tokens)} out</div>"
            f"<div class='row'><span class='k'>cost</span>{_fmt_cost(d.cost_usd)}</div>"
        )
        if d.system_prompt:
            detail_parts.append(
                f"<details><summary>system prompt</summary>"
                f"<pre>{_esc(_truncate(d.system_prompt))}</pre></details>"
            )
        detail_parts.append(
            f"<details><summary>messages</summary>"
            f"<pre>{_esc(_truncate(_format_messages(d.messages)))}</pre></details>"
        )
        detail_parts.append(
            f"<details open><summary>response</summary>"
            f"<pre>{_esc(_truncate(d.response or ''))}</pre></details>"
        )

    if span.tool_data is not None:
        d = span.tool_data
        detail_parts.append(
            f"<div class='row'><span class='k'>tool</span>{_esc(d.tool_name)}</div>"
        )
        detail_parts.append(
            f"<details><summary>input</summary>"
            f"<pre>{_esc(_truncate(_format_messages(d.tool_input)))}</pre></details>"
        )
        if d.tool_output is not None:
            detail_parts.append(
                f"<details><summary>output</summary>"
                f"<pre>{_esc(_truncate(_format_messages(d.tool_output)))}</pre></details>"
            )

    if span.metadata:
        detail_parts.append(
            f"<details><summary>metadata</summary>"
            f"<pre>{_esc(_format_messages(span.metadata))}</pre></details>"
        )

    return (
        f"<div class='span{ ' has-error' if has_error else ''}' "
        f"style='border-left-color:{color}'>"
        f"<div class='span-header'>"
        f"<span class='tag' style='background:{color};color:white'>{icon}</span>"
        f"<span class='name'>{_esc(span.name)}</span>"
        f"<span class='latency'>{latency}</span>"
        f"</div>"
        f"<div class='detail'>{''.join(detail_parts)}</div>"
        f"</div>"
    )


def to_html(trace: Trace) -> str:
    m = trace.manifest
    spans_html = "\n".join(_render_span(s) for s in trace.spans)

    error_banner = ""
    if m.errors:
        error_banner = (
            f"<div class='error-banner'>"
            f"{m.errors} error span{'s' if m.errors != 1 else ''} recorded in this run."
            f"</div>"
        )

    return (
        "<!doctype html><html><head>"
        f"<meta charset='utf-8'>"
        f"<title>TraceForge — {_esc(m.run_name)}</title>"
        f"<style>{_CSS}</style>"
        "</head><body><div class='container'>"
        f"<h1>{_esc(m.run_name)}</h1>"
        f"<div class='meta'>"
        f"{_esc(m.run_id)} · started {_esc(m.started_at)} · "
        f"TraceForge {_esc(m.traceforge_version)} (schema {_esc(m.schema_version)})"
        "</div>"
        f"{error_banner}"
        "<div class='stats'>"
        f"<div class='stat'><span class='label'>Duration</span>"
        f"<span class='value'>{_esc(m.duration_ms)}ms</span></div>"
        f"<div class='stat'><span class='label'>Spans</span>"
        f"<span class='value'>{_esc(m.total_spans)}</span></div>"
        f"<div class='stat'><span class='label'>LLM calls</span>"
        f"<span class='value'>{_esc(m.llm_calls)}</span></div>"
        f"<div class='stat'><span class='label'>Tool calls</span>"
        f"<span class='value'>{_esc(m.tool_calls)}</span></div>"
        f"<div class='stat'><span class='label'>Tokens in</span>"
        f"<span class='value'>{_esc(m.total_input_tokens)}</span></div>"
        f"<div class='stat'><span class='label'>Tokens out</span>"
        f"<span class='value'>{_esc(m.total_output_tokens)}</span></div>"
        f"<div class='stat'><span class='label'>Cost</span>"
        f"<span class='value'>{_fmt_cost(m.total_cost_usd)}</span></div>"
        f"<div class='stat'><span class='label'>Errors</span>"
        f"<span class='value'>{_esc(m.errors)}</span></div>"
        "</div>"
        f"{spans_html}"
        f"<footer>Generated by TraceForge {_esc(m.traceforge_version)}</footer>"
        "</div></body></html>"
    )
