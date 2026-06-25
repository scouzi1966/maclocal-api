package stack

import "testing"

// TestPushLen exercises Push and Len, which are independent of the Pop bug.
func TestPushLen(t *testing.T) {
	s := New()
	if s.Len() != 0 {
		t.Fatalf("new stack: Len = %d, want 0", s.Len())
	}

	s.Push(7)
	s.Push(8)
	if s.Len() != 2 {
		t.Fatalf("after two pushes: Len = %d, want 2", s.Len())
	}

	s.Push(9)
	if s.Len() != 3 {
		t.Fatalf("after three pushes: Len = %d, want 3", s.Len())
	}
}
