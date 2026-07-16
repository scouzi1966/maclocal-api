"""Expiry behavior — these fail on the buggy starting code and pass after the fix."""
import ratecache.clock as clock
from ratecache import TTLCache


def test_get_returns_none_after_expiry(monkeypatch):
    t = {"v": 1000.0}
    monkeypatch.setattr(clock, "now_seconds", lambda: t["v"])
    c = TTLCache(ttl_seconds=10)
    c.set("k", "val")
    assert c.get("k") == "val"  # still fresh
    t["v"] = 1011.0  # 11s later, ttl=10 -> expired
    assert c.get("k") is None


def test_expired_entry_is_evicted(monkeypatch):
    t = {"v": 0.0}
    monkeypatch.setattr(clock, "now_seconds", lambda: t["v"])
    c = TTLCache(ttl_seconds=5)
    c.set("k", "val")
    t["v"] = 100.0
    c.get("k")  # touching an expired entry should drop it
    assert len(c) == 0
