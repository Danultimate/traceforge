# TraceForge Replay FAQ

## "Why is my replay different from the original run?"

### LLM response changed
**Cause:** The model was updated or returned a different response at temperature > 0.
**Fix:** Set `temperature=0` in your agent for reproducible outputs.
Use `--pin-model` to lock the model version in the trace.

### Tool output changed
**Cause:** Your tool has side effects or returns dynamic data (timestamps, random IDs).
**Fix:** Use `mode="dry-run"` to mock tool calls from cache.
Or mark the dynamic field: `_run.record_tool_call(..., metadata={"note": "dynamic_output"})`

### State mismatch
**Cause:** A non-serialisable field was excluded from state snapshots.
**Fix:** Check for `__excluded__` entries in your trace JSONL.
Implement `__traceforge_serialise__` on the excluded class.

### Different branch taken
**Cause:** The upstream LLM mock returned a different response than expected.
**Fix:** Check if the message hash matches using `traceforge diff` (planned).
The heuristic cache key uses message content — if messages changed, cache misses.

## "What does similarity score < 0.4 mean?"

The traces diverged significantly — the agent took a different execution path.
This is expected when:
- The agent is non-deterministic by design
- Tool outputs changed between runs
- The LLM mock cache had a miss

Use `traceforge diff <before> <after>` (planned) to see exactly where the paths diverged.

## "Can I replay without calling any external APIs?"

Use `mode="dry-run"`:

```python
replayed = await tracer.replay(trace, agent_fn, mode="dry-run")
```

Both LLM responses and tool outputs are served from cache.
Warning: tools that validate response format or timestamps may still fail.

## "Replay modes — what's the contract?"

TraceForge calls these *reproducibility modes*, not "deterministic". The
agent function is what consumes the cached responses; if your agent skips
the `_mock_llm` / `_mock_tool` arguments, traces will diverge silently. The
replay engine never patches your SDK clients automatically.
