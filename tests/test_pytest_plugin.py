"""Tests for the pytest plugin (fixtures, assertions, snapshot)."""
import pytest

from traceforge import Tracer
from traceforge.pytest_plugin import TFSnapshot, _assert_trace


# ---------- _assert_trace -----------------------------------------------------

@pytest.mark.asyncio
async def test_assert_trace_passes_on_clean_run():
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[],
            response="ok",
            input_tokens=10,
            output_tokens=5,
        )
        run.record_tool_call("noop", tool_input={}, tool_output=None)

    _assert_trace(
        run.trace,
        has_span="anthropic/claude-haiku-4-5",
        llm_calls=1,
        tool_calls=1,
        max_cost_usd=1.0,
        min_spans=2,
    )


@pytest.mark.asyncio
async def test_assert_trace_fails_when_span_missing():
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run:
        run.record_tool_call("a", tool_input={}, tool_output=None)

    with pytest.raises(AssertionError, match="missing span"):
        _assert_trace(run.trace, has_span="never-recorded")


@pytest.mark.asyncio
async def test_assert_trace_fails_on_cost_budget():
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-opus-4-7",
            messages=[],
            response="",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )

    with pytest.raises(AssertionError, match="Cost budget"):
        _assert_trace(run.trace, max_cost_usd=0.01)


@pytest.mark.asyncio
async def test_assert_trace_fails_on_errors_by_default():
    tracer = Tracer(auto_save=False)
    with pytest.raises(RuntimeError):
        async with tracer.run() as run:
            raise RuntimeError("boom")

    with pytest.raises(AssertionError, match="error span"):
        _assert_trace(tracer.last())


# ---------- TFSnapshot --------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_records_then_matches(tmp_path):
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[],
            response="hi",
            input_tokens=5,
            output_tokens=1,
        )
        run.record_tool_call("t", tool_input={}, tool_output=None)

    snap = TFSnapshot(snapshot_dir=tmp_path / "snaps", update=False)
    # First call: writes the fixture and returns silently.
    snap.assert_match(run.trace, name="agent_v1")
    assert (tmp_path / "snaps" / "agent_v1.jsonl").exists()

    # Second call against the same shape: should match.
    snap.assert_match(run.trace, name="agent_v1")


@pytest.mark.asyncio
async def test_snapshot_mismatch_raises(tmp_path):
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run_a:
        run_a.record_llm_call(
            provider="x", model="claude-haiku-4-5",
            messages=[], response="", input_tokens=1, output_tokens=1,
        )
        run_a.record_tool_call("a", tool_input={}, tool_output=None)
        run_a.record_tool_call("b", tool_input={}, tool_output=None)
        run_a.record_tool_call("c", tool_input={}, tool_output=None)

    snap = TFSnapshot(snapshot_dir=tmp_path / "snaps", update=False)
    snap.assert_match(run_a.trace, name="shape")

    # Now run an agent that takes a totally different path.
    async with tracer.run() as run_b:
        run_b.record_error("divergent")

    with pytest.raises(AssertionError, match="snapshot mismatch"):
        snap.assert_match(run_b.trace, name="shape")


@pytest.mark.asyncio
async def test_snapshot_update_flag_overwrites(tmp_path):
    tracer = Tracer(auto_save=False)
    async with tracer.run() as run_a:
        run_a.record_tool_call("a", tool_input={}, tool_output=None)

    snap = TFSnapshot(snapshot_dir=tmp_path / "snaps", update=False)
    snap.assert_match(run_a.trace, name="evolving")

    # New shape, run with update=True → silently overwrites.
    async with tracer.run() as run_b:
        run_b.record_tool_call("a", tool_input={}, tool_output=None)
        run_b.record_tool_call("b", tool_input={}, tool_output=None)

    snap_update = TFSnapshot(snapshot_dir=tmp_path / "snaps", update=True)
    snap_update.assert_match(run_b.trace, name="evolving")

    # Subsequent non-update run matches the new shape.
    snap.assert_match(run_b.trace, name="evolving")


# ---------- end-to-end fixture discovery via pytester ------------------------

def test_plugin_fixtures_discovered(pytester):
    """Verify the pytest11 entry point wires up `tracer` / `tf_assert`
    fixtures for end users.
    """
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.asyncio
        async def test_uses_tracer_fixture(tracer, tf_assert):
            async with tracer.run() as run:
                run.record_tool_call("x", tool_input={}, tool_output=None)
            tf_assert(run.trace, tool_calls=1)
        """
    )
    pytester.makefile(".ini", pytest="[pytest]\nasyncio_mode = auto\n")
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)
