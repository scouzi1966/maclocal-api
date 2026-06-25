package lru

import "testing"

// --- fail_to_pass: eviction + recency (absent in the starting repo) ---

func TestEvictsLeastRecentlyUsed(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	c.Put("b", 2)
	c.Put("c", 3) // exceeds capacity -> "a" (LRU) is evicted
	if _, ok := c.Get("a"); ok {
		t.Fatalf("expected a to be evicted")
	}
	if v, ok := c.Get("b"); !ok || v != 2 {
		t.Fatalf("b = (%d,%v), want (2,true)", v, ok)
	}
	if v, ok := c.Get("c"); !ok || v != 3 {
		t.Fatalf("c = (%d,%v), want (3,true)", v, ok)
	}
}

func TestGetMarksRecentlyUsed(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	c.Put("b", 2)
	if _, ok := c.Get("a"); !ok { // touch "a" so "b" becomes the LRU
		t.Fatalf("expected a present")
	}
	c.Put("c", 3) // evicts "b", not "a"
	if _, ok := c.Get("b"); ok {
		t.Fatalf("expected b to be evicted")
	}
	if v, ok := c.Get("a"); !ok || v != 1 {
		t.Fatalf("a = (%d,%v), want (1,true)", v, ok)
	}
}

func TestUpdateMarksRecentlyUsed(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	c.Put("b", 2)
	c.Put("a", 9) // update "a" -> now most-recently-used
	c.Put("c", 3) // evicts "b"
	if _, ok := c.Get("b"); ok {
		t.Fatalf("expected b to be evicted")
	}
	if v, ok := c.Get("a"); !ok || v != 9 {
		t.Fatalf("a = (%d,%v), want (9,true)", v, ok)
	}
}

func TestLenStaysWithinCapacity(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	c.Put("b", 2)
	c.Put("c", 3)
	if c.Len() != 2 {
		t.Fatalf("Len = %d, want 2", c.Len())
	}
}

// --- pass_to_pass: basic map behavior (holds before and after) ---

func TestGetMissReturnsFalse(t *testing.T) {
	c := New(2)
	if v, ok := c.Get("x"); ok || v != 0 {
		t.Fatalf("miss = (%d,%v), want (0,false)", v, ok)
	}
}

func TestPutThenGet(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	if v, ok := c.Get("a"); !ok || v != 1 {
		t.Fatalf("a = (%d,%v), want (1,true)", v, ok)
	}
}

func TestUpdateValueInPlace(t *testing.T) {
	c := New(2)
	c.Put("a", 1)
	c.Put("a", 2)
	if v, ok := c.Get("a"); !ok || v != 2 {
		t.Fatalf("a = (%d,%v), want (2,true)", v, ok)
	}
}
