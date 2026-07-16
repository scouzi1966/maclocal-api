"""An in-memory cache where each entry expires after a fixed TTL."""
from __future__ import annotations

from typing import Any

from ratecache import clock


class TTLCache:
    """Maps keys to values; entries expire ``ttl_seconds`` after they are set."""

    def __init__(self, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl = ttl_seconds
        self._store: dict[Any, tuple[Any, float]] = {}

    def set(self, key: Any, value: Any) -> None:
        self._store[key] = (value, clock.now_seconds() + self.ttl)

    def get(self, key: Any) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, _expires_at = entry
        return value

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None
