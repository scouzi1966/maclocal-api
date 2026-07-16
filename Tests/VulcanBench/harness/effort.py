"""Normalized reasoning-effort helpers for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EFFORT_LEVELS = frozenset({"low", "medium", "high", "extra-high"})
DEFAULT_SWEEP_EFFORTS = ("low", "medium", "high")

_OPENAI_EFFORT_VALUES = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra-high": "xhigh",
}

# Anthropic Messages API `output_config.effort`. `xhigh` is model-dependent
# (Opus 4.7+); unsupported combinations are rejected by the API with a clear
# error, mirroring how OpenAI handles model-dependent effort values.
_ANTHROPIC_EFFORT_VALUES = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra-high": "xhigh",
}


class EffortNotSupportedError(ValueError):
    """Raised when a provider cannot run a requested effort level."""


@dataclass(frozen=True)
class EffortConfig:
    requested: str
    provider: str
    provider_value: str | None
    supported: bool

    def as_summary(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "provider": self.provider,
            "provider_value": self.provider_value,
            "supported": self.supported,
        }


def normalize_effort(value: str | None) -> str | None:
    """Normalize a user-facing effort string, returning ``None`` when unset."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in EFFORT_LEVELS:
        allowed = ", ".join(sorted(EFFORT_LEVELS))
        raise ValueError(f"effort must be one of {allowed}, got {value!r}")
    return normalized


def parse_efforts(raw: str | None) -> list[str]:
    """Parse a comma-separated effort list for sweep runs."""
    if raw is None or not raw.strip():
        return list(DEFAULT_SWEEP_EFFORTS)
    values = [normalize_effort(part) for part in raw.split(",") if part.strip()]
    efforts = [v for v in values if v is not None]
    if not efforts:
        raise ValueError("at least one effort level is required")
    seen: set[str] = set()
    deduped: list[str] = []
    for effort in efforts:
        if effort not in seen:
            seen.add(effort)
            deduped.append(effort)
    return deduped


def effort_config(provider: str, requested: str | None) -> EffortConfig | None:
    """Resolve benchmark effort metadata for a provider."""
    effort = normalize_effort(requested)
    if effort is None:
        return None

    provider_name = provider.strip().lower()
    if provider_name == "openai":
        return EffortConfig(
            requested=effort,
            provider="openai",
            provider_value=_OPENAI_EFFORT_VALUES[effort],
            supported=True,
        )
    if provider_name in {"mock", "zai"}:
        return EffortConfig(
            requested=effort,
            provider=provider_name,
            provider_value=None,
            supported=False,
        )
    if provider_name == "anthropic":
        return EffortConfig(
            requested=effort,
            provider="anthropic",
            provider_value=_ANTHROPIC_EFFORT_VALUES[effort],
            supported=True,
        )
    raise EffortNotSupportedError(f"reasoning effort is not supported for provider {provider!r}")
