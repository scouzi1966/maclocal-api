# Consolidate the duplicated retry logic (and fix the two copies that disagree)

`HttpClient.request_with_retry` and `DbClient.query_with_retry` both implement
the same retry-with-exponential-backoff loop, copy-pasted into two modules. The
copies have drifted apart and now behave differently, which is a bug:

- `HttpClient` runs the full number of attempts but **forgets to cap the backoff
  delay** at `max_delay` — delays grow unboundedly (`1, 2, 4, 8, …`).
- `DbClient` caps the delay correctly but **runs one fewer attempt than
  `max_attempts`** (an off-by-one in the loop bound), so an operation that would
  succeed on its final attempt is reported as a failure.

Refactor the retry loop into a single shared helper and have both clients use
it, so there is exactly one implementation and the two clients behave
identically. The unified, correct contract is:

- The operation is attempted up to `max_attempts` times (the first call plus
  retries). The result of the first successful call is returned.
- Before the k-th retry (k = 1, 2, …) the client sleeps
  `min(base_delay * 2**(k-1), max_delay)` seconds — i.e. exponential backoff,
  capped at `max_delay`.
- If every attempt fails, the last exception is re-raised.

The clients live in `resilient/http_client.py` and `resilient/db_client.py`.
Both take an injectable `sleep` callable (the tests pass a recorder instead of
`time.sleep`), so behavior is observable without real delays.
