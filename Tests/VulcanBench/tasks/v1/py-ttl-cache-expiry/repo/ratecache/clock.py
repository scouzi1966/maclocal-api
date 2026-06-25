"""Monotonic clock, isolated so it can be controlled in tests."""
from __future__ import annotations

import time


def now_seconds() -> float:
    """Seconds from a monotonic source (never goes backwards)."""
    return time.monotonic()
