import pytest

from traceforge import Tracer


@pytest.mark.asyncio
async def test_replay_llm_mock_hits_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            messages=messages,
            response="4",
            input_tokens=8,
            output_tokens=1,
            latency_ms=12,
        )
    original = run.trace

    async def replayed_agent(_run, _mock_llm, _mock_tool=None, **_):
        cached = _mock_llm.get(messages)
        assert cached == "4"
        _run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            messages=messages,
            response=cached,
            latency_ms=0,
        )

    result = await tracer.replay(original, replayed_agent, mode="llm-mock")
    assert result.similarity_score >= 0.99
    assert not result.diverged


@pytest.mark.asyncio
async def test_replay_dry_run_serves_tool_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run:
        run.record_tool_call("lookup", tool_input={"k": "x"}, tool_output={"v": 42})
    original = run.trace

    async def agent(_run, _mock_llm, _mock_tool, **_):
        cached = _mock_tool.get("lookup", {"k": "x"})
        assert cached == {"v": 42}
        _run.record_tool_call("lookup", tool_input={"k": "x"}, tool_output=cached)

    result = await tracer.replay(original, agent, mode="dry-run")
    assert result.similarity_score == 1.0


@pytest.mark.asyncio
async def test_replay_diverged_when_paths_differ(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            response="y",
        )
        run.record_tool_call("t1", tool_input={}, tool_output=None)
        run.record_tool_call("t2", tool_input={}, tool_output=None)
        run.record_tool_call("t3", tool_input={}, tool_output=None)
    original = run.trace

    async def divergent_agent(_run, _mock_llm, _mock_tool=None, **_):
        _run.record_error("totally different path")

    result = await tracer.replay(original, divergent_agent, mode="llm-mock")
    assert result.diverged
    assert result.similarity_score < 0.4


def test_replay_engine_rejects_unknown_mode():
    from traceforge.replay import ReplayEngine
    from traceforge.trace import Trace, TraceManifest
    from datetime import datetime

    manifest = TraceManifest(
        run_id="01H", run_name="x", started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(), duration_ms=0, total_spans=0,
        llm_calls=0, tool_calls=0, errors=0,
        total_input_tokens=0, total_output_tokens=0,
        traceforge_version="0.1.0",
    )
    trace = Trace(manifest=manifest, spans=[])
    with pytest.raises(ValueError):
        ReplayEngine(trace, mode="time-travel")
