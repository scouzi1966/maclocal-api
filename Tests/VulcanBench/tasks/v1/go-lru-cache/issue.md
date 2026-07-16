# Implement LRU eviction for the cache

The `lru.Cache` type (`example.com/lrucache/lru`) is supposed to be a
fixed-capacity **least-recently-used** cache, but right now it is just a map: it
keeps growing past its capacity and never evicts anything.

Make `Cache` behave like a real LRU cache with capacity `New(capacity)`:

- It holds at most `capacity` entries. Inserting a new key when the cache is
  full must first evict the **least-recently-used** entry.
- Both `Get` (on a hit) and `Put` (insert or update) count as *using* a key and
  make it the most-recently-used. A `Get` miss returns `(0, false)` and does not
  change anything.
- `Len()` returns the current number of stored entries (never more than
  `capacity`).

Example:

```go
c := lru.New(2)
c.Put("a", 1)
c.Put("b", 2)
c.Get("a")     // touches "a", so "b" is now least-recently-used
c.Put("c", 3)  // evicts "b"
c.Get("b")     // (0, false)
```

The cache lives in `lru/lru.go`. The standard library `container/list` is a
convenient way to track recency, but any correct approach is fine.
