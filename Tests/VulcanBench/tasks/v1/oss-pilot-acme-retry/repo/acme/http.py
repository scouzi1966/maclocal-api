from acme.config import DEFAULT_BASE_DELAY, DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_DELAY
from acme.retry import retry_call

class HttpClient:
    def __init__(self, sleep):
        self._sleep = sleep
    def request_with_retry(self, fn, max_attempts=DEFAULT_MAX_ATTEMPTS,
                           base_delay=DEFAULT_BASE_DELAY, max_delay=DEFAULT_MAX_DELAY):
        return retry_call(fn, max_attempts=max_attempts, base_delay=base_delay,
                          max_delay=max_delay, sleep=self._sleep)
