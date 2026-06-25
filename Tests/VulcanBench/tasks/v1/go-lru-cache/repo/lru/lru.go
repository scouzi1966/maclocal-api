// Package lru is meant to provide a fixed-capacity least-recently-used cache.
//
// NOTE: the current implementation is just a map — it never evicts and does not
// track recency. See issue.md.
package lru

// Cache maps string keys to int values with a bounded capacity.
type Cache struct {
	capacity int
	data     map[string]int
}

// New returns an empty cache that should hold at most `capacity` entries.
func New(capacity int) *Cache {
	return &Cache{capacity: capacity, data: make(map[string]int)}
}

// Get returns the value for key and whether it was present.
func (c *Cache) Get(key string) (int, bool) {
	v, ok := c.data[key]
	return v, ok
}

// Put inserts or updates the value for key.
func (c *Cache) Put(key string, value int) {
	c.data[key] = value
}

// Len reports how many entries are currently stored.
func (c *Cache) Len() int {
	return len(c.data)
}
