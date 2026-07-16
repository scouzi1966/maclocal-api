# TTLCache returns expired entries

`ratecache.TTLCache` is supposed to expire each entry `ttl_seconds` after it was
set. In practice, `get()` keeps returning values long after their TTL has passed,
and expired entries are never removed from the cache, so memory grows without
bound.

Expected behavior:

- `get(key)` returns the value only while the entry is still within its TTL.
- Once an entry has expired (current time is at or past its expiry), `get(key)`
  returns `None`.
- Accessing an expired entry should drop it from the cache (so `len(cache)`
  reflects only live entries).

The clock lives in `ratecache/clock.py` (`now_seconds()`); tests control time by
patching it, so use it rather than calling `time` directly.
