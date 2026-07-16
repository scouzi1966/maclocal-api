"""Per-model token pricing.

Computes the USD cost of a run from its prompt/completion token counts. Prices
are a built-in table (USD per 1M tokens) that can be overridden via the
``VULCANBENCH_PRICING`` env var (path to a JSON file merged over the defaults).

Honesty: unknown models return ``None`` (cost unknown) rather than a guessed
number; ``mock`` models are free. Built-in prices are a point-in-time snapshot —
override them for anything that must be exact.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

# USD per 1,000,000 tokens, as of 2026-06. Keys are exact "provider:model" specs;
# lookup also falls back to a "provider:" prefix default. Override with a JSON
# file at $VULCANBENCH_PRICING ({"openai:gpt-4o": {"input": .., "output": ..}}).
# These are a point-in-time snapshot — verify against the provider's pricing page
# before publishing numbers, and use the override file for anything that must be
# exact.
PRICES: dict[str, dict[str, float]] = {
    "openai:gpt-5.5": {"input": 5.00, "output": 30.00},
    "openai:gpt-5.5-pro": {"input": 30.00, "output": 180.00},
    "openai:gpt-5.4": {"input": 2.50, "output": 15.00},
    "openai:gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "openai:gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "openai:gpt-5": {"input": 1.25, "output": 10.00},
    "openai:gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai:gpt-5-nano": {"input": 0.05, "output": 0.40},
    "openai:gpt-4o": {"input": 2.50, "output": 10.00},
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-4.1": {"input": 2.00, "output": 8.00},
    "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai:o3": {"input": 2.00, "output": 8.00},
    "openai:o4-mini": {"input": 1.10, "output": 4.40},
    "anthropic:claude-fable-5": {"input": 10.00, "output": 50.00},
    "anthropic:claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "anthropic:claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "anthropic:claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "anthropic:claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "zai:glm-5.2": {"input": 1.40, "output": 4.40},
    "zai:glm-5.1": {"input": 1.40, "output": 4.40},
    "zai:glm-5": {"input": 1.00, "output": 3.20},
    "zai:glm-5-turbo": {"input": 1.20, "output": 4.00},
    # Free / offline.
    "mock:": {"input": 0.0, "output": 0.0},
}

_PER_MILLION = 1_000_000.0


@lru_cache(maxsize=1)
def _prices() -> dict[str, dict[str, float]]:
    prices = dict(PRICES)
    override = os.environ.get("VULCANBENCH_PRICING")
    if override and os.path.exists(override):
        try:
            with open(override, encoding="utf-8") as f:
                custom = json.load(f)
            if isinstance(custom, dict):
                prices.update(custom)
        except (OSError, json.JSONDecodeError):
            pass
    return prices


def _rate(model: str) -> dict[str, float] | None:
    prices = _prices()
    if model in prices:
        return prices[model]
    provider = model.split(":", 1)[0] + ":" if ":" in model else ""
    return prices.get(provider)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD cost for a model call, or ``None`` if the model is not priced."""
    rate = _rate(model)
    if rate is None:
        return None
    cost = (prompt_tokens * rate["input"] + completion_tokens * rate["output"]) / _PER_MILLION
    return round(cost, 6)


def is_priced(model: str) -> bool:
    """True if ``model`` has a known price (so a real cost can be computed)."""
    return _rate(model) is not None


def reset_cache() -> None:
    """Clear the memoized price table (used by tests after setting the env var)."""
    _prices.cache_clear()


def merged_prices() -> dict[str, Any]:
    """Return the effective price table (defaults + override). For inspection."""
    return dict(_prices())
