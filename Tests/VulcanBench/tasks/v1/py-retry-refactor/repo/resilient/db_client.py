"""DB client with its own (slightly different) copy of the retry loop."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


class DbClient:
    """Calls an operation, retrying with exponential backoff on failure."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep

    def query_with_retry(self, operation: Callable[[], T]) -> T:
        """Call ``operation`` up to ``max_attempts`` times; return its result.

        Re-raises the last exception if every attempt fails.
        """
        last_exc: BaseException | None = None
        for attempt in range(self.max_attempts - 1):
            if attempt > 0:
                self._sleep(min(self.base_delay * 2 ** (attempt - 1), self.max_delay))
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 — retry on any operation failure
                last_exc = exc
        assert last_exc is not None
        raise last_exc
