"""Token pricing table for cost estimation.

Prices are USD per 1M tokens. Vendors change pricing every few months, so
the table here is *best effort* — override per-Tracer for production use:

    from traceforge.pricing import ModelPrice
    tracer = Tracer(pricing={"my-model": ModelPrice(input_per_million=2.0,
                                                    output_per_million=8.0)})

Unknown models cost 0.0 (with a one-shot warning), so cost never blocks a
trace from being saved.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


# Best-effort published list prices as of mid-2026. Override per-Tracer for
# real production accounting.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-4-7":           ModelPrice(15.00, 75.00),
    "claude-opus-4-6":           ModelPrice(15.00, 75.00),
    "claude-sonnet-4-6":         ModelPrice(3.00, 15.00),
    "claude-sonnet-4-5":         ModelPrice(3.00, 15.00),
    "claude-haiku-4-5":          ModelPrice(1.00, 5.00),
    # OpenAI
    "gpt-4o":                    ModelPrice(2.50, 10.00),
    "gpt-4o-mini":               ModelPrice(0.15, 0.60),
    "gpt-4-turbo":               ModelPrice(10.00, 30.00),
    "o1":                        ModelPrice(15.00, 60.00),
    "o1-mini":                   ModelPrice(3.00, 12.00),
    # Local / free
    "ollama":                    ModelPrice(0.0, 0.0),
    "local":                     ModelPrice(0.0, 0.0),
}

_WARNED_MODELS: set[str] = set()


def _lookup(model: str, table: dict[str, ModelPrice]) -> Optional[ModelPrice]:
    if model in table:
        return table[model]
    # Prefix match: "claude-haiku-4-5-20251001" → "claude-haiku-4-5"
    # Longest matching prefix wins so "claude-opus-4-7" beats "claude-opus".
    candidates = [k for k in table if model.startswith(k)]
    if candidates:
        return table[max(candidates, key=len)]
    return None


def estimate_cost(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    pricing: Optional[dict[str, ModelPrice]] = None,
) -> float:
    """Return USD cost for a single LLM call. Returns 0.0 if model unknown."""
    table = pricing if pricing is not None else DEFAULT_PRICING
    price = _lookup(model, table)
    if price is None:
        if model not in _WARNED_MODELS:
            _WARNED_MODELS.add(model)
            warnings.warn(
                f"TraceForge: no pricing for model {model!r}; cost will be 0. "
                "Pass `pricing=` to `Tracer(...)` to override.",
                stacklevel=2,
            )
        return 0.0
    return (
        (input_tokens or 0) / 1_000_000 * price.input_per_million
        + (output_tokens or 0) / 1_000_000 * price.output_per_million
    )
