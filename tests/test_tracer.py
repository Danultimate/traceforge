import pytest

from traceforge import Tracer
from traceforge.span import SpanType


@pytest.mark.asyncio
async def test_async_run_captures_llm_and_tool_spans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            input_tokens=2,
            output_tokens=1,
            latency_ms=50,
        )
        run.record_tool_call(
            tool_name="add", tool_input={"a": 1, "b": 2}, tool_output=3, latency_ms=1
        )

    trace = run.trace
    assert trace is not None
    assert trace.manifest.total_spans == 2
    assert trace.manifest.llm_calls == 1
    assert trace.manifest.tool_calls == 1
    assert trace.manifest.total_input_tokens == 2
    assert trace.manifest.total_output_tokens == 1
    assert trace.manifest.run_name and "-" in trace.manifest.run_name


def test_sync_run_records_errors_and_finalises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    with pytest.raises(RuntimeError):
        with tracer.run_sync() as run:
            run.record_tool_call("noop", tool_input={}, tool_output=None)
            raise RuntimeError("boom")

    trace = tracer.last()
    assert trace is not None
    assert trace.manifest.errors == 1
    error_spans = [s for s in trace.spans if s.span_type == SpanType.ERROR]
    assert error_spans[0].error == "boom"


@pytest.mark.asyncio
async def test_auto_save_writes_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=True)

    async with tracer.run() as run:
        run.custom("phase", metadata={"step": 1})

    runs = list((tmp_path / ".traceforge" / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "run.jsonl").exists()
    assert (run_dir / "report.html").exists()


@pytest.mark.asyncio
async def test_decorator_sugar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    @tracer.trace
    async def agent(x, _run=None):
        _run.record_tool_call("double", tool_input={"x": x}, tool_output=x * 2)
        return x * 2

    result = await agent(5)
    assert result == 10
    assert tracer.last().manifest.tool_calls == 1
