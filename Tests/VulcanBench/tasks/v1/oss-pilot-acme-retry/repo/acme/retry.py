import time


def retry_call(fn, *, max_attempts, base_delay, max_delay, sleep=time.sleep):
    last = None
    for attempt in range(max_attempts - 1):
        try:
            return fn()
        except Exception as e:
            last = e
            delay = base_delay * (2 ** attempt)
            sleep(delay)
    raise last
