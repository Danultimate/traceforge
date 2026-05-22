import warnings

import pytest

from traceforge import Tracer
from traceforge.pricing import (
    DEFAULT_PRICING,
    ModelPrice,
    _WARNED_MODELS,
    estimate_cost,
)


def test_estimate_cost_exact_match():
    cost = estimate_cost(
        "claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    expected = DEFAULT_PRICING["claude-haiku-4-5"]
    assert cost == pytest.approx(
        expected.input_per_million + expected.output_per_million
    )


def test_estimate_cost_prefix_match_longest_wins():
    cost = estimate_cost(
        "claude-haiku-4-5-20251001", input_tokens=1_000, output_tokens=500
    )
    price = DEFAULT_PRICING["claude-haiku-4-5"]
    expected = (
        1_000 / 1_000_000 * price.input_per_million
        + 500 / 1_000_000 * price.output_per_million
    )
    assert cost == pytest.approx(expected)


def test_estimate_cost_unknown_model_warns_once():
    _WARNED_MODELS.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert estimate_cost("totally-fake-model", 100, 50) == 0.0
        assert estimate_cost("totally-fake-model", 100, 50) == 0.0
    # Warned only once for the same model.
    relevant = [w for w in caught if "totally-fake-model" in str(w.message)]
    assert len(relevant) == 1


def test_estimate_cost_handles_none_tokens():
    assert estimate_cost("claude-haiku-4-5", None, None) == 0.0


def test_custom_pricing_table_overrides_default():
    table = {"my-model": ModelPrice(input_per_million=100.0, output_per_million=200.0)}
    cost = estimate_cost("my-model", 1_000_000, 1_000_000, pricing=table)
    assert cost == pytest.approx(300.0)


@pytest.mark.asyncio
async def test_tracer_attaches_cost_to_llm_span(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "x"}],
            response="y",
            input_tokens=1_000,
            output_tokens=500,
        )

    trace = run.trace
    span = trace.llm_spans[0]
    expected = estimate_cost("claude-haiku-4-5", 1_000, 500)
    assert span.llm_data.cost_usd == pytest.approx(expected)
    assert trace.manifest.total_cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_explicit_cost_overrides_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracer = Tracer(auto_save=False)

    async with tracer.run() as run:
        run.record_llm_call(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[],
            response="",
            input_tokens=999,
            output_tokens=999,
            cost_usd=0.42,
        )

    assert run.trace.llm_spans[0].llm_data.cost_usd == 0.42
    assert run.trace.manifest.total_cost_usd == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_tracer_accepts_custom_pricing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = {"local-llm": ModelPrice(input_per_million=0.5, output_per_million=1.5)}
    tracer = Tracer(auto_save=False, pricing=custom)

    async with tracer.run() as run:
        run.record_llm_call(
            provider="local",
            model="local-llm",
            messages=[],
            response="",
            input_tokens=2_000_000,
            output_tokens=1_000_000,
        )

    # 2M * 0.5 + 1M * 1.5 = 1.0 + 1.5 = 2.5
    assert run.trace.manifest.total_cost_usd == pytest.approx(2.5)
