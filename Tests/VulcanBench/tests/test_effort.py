"""Tests for normalized reasoning-effort helpers."""

from __future__ import annotations

import pytest

from harness.effort import effort_config, parse_efforts


def test_parse_efforts_defaults_and_dedupes() -> None:
    assert parse_efforts(None) == ["low", "medium", "high"]
    assert parse_efforts("low, medium, low") == ["low", "medium"]


def test_openai_extra_high_maps_to_xhigh() -> None:
    cfg = effort_config("openai", "extra-high")
    assert cfg is not None
    assert cfg.as_summary() == {
        "requested": "extra-high",
        "provider": "openai",
        "provider_value": "xhigh",
        "supported": True,
    }


def test_mock_effort_is_noop_metadata() -> None:
    cfg = effort_config("mock", "low")
    assert cfg is not None
    assert cfg.as_summary() == {
        "requested": "low",
        "provider": "mock",
        "provider_value": None,
        "supported": False,
    }


def test_zai_effort_is_noop_metadata() -> None:
    cfg = effort_config("zai", "low")
    assert cfg is not None
    assert cfg.as_summary() == {
        "requested": "low",
        "provider": "zai",
        "provider_value": None,
        "supported": False,
    }


def test_anthropic_effort_maps_to_output_config_values() -> None:
    cfg = effort_config("anthropic", "medium")
    assert cfg is not None
    assert cfg.as_summary() == {
        "requested": "medium",
        "provider": "anthropic",
        "provider_value": "medium",
        "supported": True,
    }
    xhigh = effort_config("anthropic", "extra-high")
    assert xhigh is not None
    assert xhigh.provider_value == "xhigh"


def test_unknown_provider_effort_rejected() -> None:
    with pytest.raises(ValueError, match="not supported for provider"):
        effort_config("acme", "medium")
