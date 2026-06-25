"""Behavior that must hold before and after the fix (regression guard)."""
import pytest

import ratecache.clock as clock
from ratecache import TTLCache


def test_set_get_within_ttl(monkeypatch):
    monkeypatch.setattr(clock, "now_seconds", lambda: 100.0)
    c = TTLCache(ttl_seconds=10)
    c.set("a", 1)
    assert c.get("a") == 1


def test_missing_key_returns_none():
    c = TTLCache(ttl_seconds=10)
    assert c.get("nope") is None


def test_invalid_ttl_rejected():
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=0)
