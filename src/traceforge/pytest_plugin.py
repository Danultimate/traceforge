"""TraceForge pytest plugin.

Provides three things:

1. A `tracer` fixture — a fresh `Tracer(auto_save=False)` per test, so test
   runs don't pollute `.traceforge/runs/`.
2. A `tf_assert` fixture — a callable that takes a `Trace` and a set of
   keyword assertions (`has_span=...`, `no_errors=True`,
   `max_cost_usd=...`, `llm_calls=...`). One-liner for the common cases.
3. A `tf_snapshot` fixture — golden-trace snapshot testing. First run saves
   the trace as a JSONL fixture; subsequent runs assert
   span-type-sequence similarity >= a threshold. Refresh snapshots with
   `pytest --tf-update-snapshots`.

Registered automatically via the `pytest11` entry point in pyproject, so
users just `pip install traceforge-llm` and the fixtures appear.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from traceforge import Tracer
from traceforge.span import Span, SpanType
from traceforge.trace import Trace, TraceManifest


# ---------- CLI options -------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("traceforge")
    group.addoption(
        "--tf-update-snapshots",
        action="store_true",
        default=False,
        help="Re-record TraceForge trace snapshots used by tf_snapshot.",
    )
    group.addoption(
        "--tf-snapshot-dir",
        default="tests/__tf_snapshots__",
        help="Directory holding tf_snapshot JSONL fixtures.",
    )


# ---------- fixtures ----------------------------------------------------------

@pytest.fixture
def tracer() -> Tracer:
    """A non-auto-saving Tracer scoped to the current test."""
    return Tracer(auto_save=False)


@pytest.fixture
def tf_assert():
    """Return a callable that runs common assertions against a Trace."""
    return _assert_trace


@pytest.fixture
def tf_snapshot(request, pytestconfig) -> "TFSnapshot":
    """Golden-trace snapshot fixture.

    Usage:
        async def test_agent(tracer, tf_snapshot):
            async with tracer.run() as run:
                await my_agent(query, _run=run)
            tf_snapshot.assert_match(run.trace, "agent_v1")
    """
    rootdir = Path(str(pytestconfig.rootpath))
    snap_dir = rootdir / pytestconfig.getoption("--tf-snapshot-dir")
    update = pytestconfig.getoption("--tf-update-snapshots")
    return TFSnapshot(snapshot_dir=snap_dir, update=update)


# ---------- assertion helper --------------------------------------------------

def _assert_trace(
    trace: Trace,
    *,
    has_span: Optional[str] = None,
    has_span_type: Optional[SpanType] = None,
    no_errors: bool = True,
    llm_calls: Optional[int] = None,
    tool_calls: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
    max_tokens: Optional[int] = None,
    min_spans: Optional[int] = None,
) -> None:
    """One-shot trace assertions. Raises AssertionError on first failure."""
    m = trace.manifest

    if no_errors and m.errors:
        errs = [s.error for s in trace.error_spans]
        raise AssertionError(
            f"Trace recorded {m.errors} error span(s): {errs}"
        )

    if has_span is not None and not trace.has_span(name=has_span):
        names = [s.name for s in trace.spans]
        raise AssertionError(
            f"Trace missing span named {has_span!r}. Saw: {names}"
        )

    if has_span_type is not None and not trace.has_span(span_type=has_span_type):
        raise AssertionError(
            f"Trace missing span of type {has_span_type!r}."
        )

    if llm_calls is not None and m.llm_calls != llm_calls:
        raise AssertionError(
            f"Expected {llm_calls} LLM call(s), got {m.llm_calls}."
        )

    if tool_calls is not None and m.tool_calls != tool_calls:
        raise AssertionError(
            f"Expected {tool_calls} tool call(s), got {m.tool_calls}."
        )

    if max_cost_usd is not None and m.total_cost_usd > max_cost_usd:
        raise AssertionError(
            f"Cost budget exceeded: ${m.total_cost_usd:.4f} > ${max_cost_usd:.4f}."
        )

    if max_tokens is not None:
        total = m.total_input_tokens + m.total_output_tokens
        if total > max_tokens:
            raise AssertionError(
                f"Token budget exceeded: {total} > {max_tokens}."
            )

    if min_spans is not None and m.total_spans < min_spans:
        raise AssertionError(
            f"Expected at least {min_spans} span(s), got {m.total_spans}."
        )


# ---------- snapshot helper ---------------------------------------------------

class TFSnapshot:
    def __init__(self, snapshot_dir: Path, update: bool):
        self.dir = Path(snapshot_dir)
        self.update = update

    def assert_match(
        self,
        trace: Trace,
        name: str,
        min_similarity: float = 0.8,
    ) -> None:
        """Snapshot-assert a trace.

        First run (or with `--tf-update-snapshots`): saves trace to fixture.
        Subsequent runs: loads snapshot, compares span-type sequence,
        asserts similarity >= min_similarity.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{name}.jsonl"

        if self.update or not path.exists():
            path.write_text(trace.to_jsonl())
            return

        snapshot = _load_snapshot(path)
        score = _span_type_similarity(snapshot, trace)
        if score < min_similarity:
            raise AssertionError(
                f"TraceForge snapshot mismatch for {name!r}: "
                f"similarity={score:.0%} (min={min_similarity:.0%}).\n"
                f"  Snapshot spans:  {[s.span_type.value for s in snapshot.spans]}\n"
                f"  Recorded spans:  {[s.span_type.value for s in trace.spans]}\n"
                "Run `pytest --tf-update-snapshots` to refresh the fixture."
            )


def _span_type_similarity(a: Trace, b: Trace) -> float:
    seq_a = [s.span_type for s in a.spans]
    seq_b = [s.span_type for s in b.spans]
    if not seq_a and not seq_b:
        return 1.0
    matches = sum(x == y for x, y in zip(seq_a, seq_b))
    return matches / max(len(seq_a), len(seq_b))


def _load_snapshot(path: Path) -> Trace:
    manifest: Optional[TraceManifest] = None
    spans: list[Span] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.pop("__type__", None)
        if kind == "manifest":
            manifest = TraceManifest.model_validate(obj)
        elif kind == "span":
            spans.append(Span.model_validate(obj))
        elif "total_spans" in obj:
            # Untagged manifest (legacy JSONL written via run.jsonl).
            manifest = TraceManifest.model_validate(obj)
        else:
            spans.append(Span.model_validate(obj))
    if manifest is None:
        raise ValueError(f"Snapshot {path} has no manifest line")
    return Trace(manifest=manifest, spans=spans)
