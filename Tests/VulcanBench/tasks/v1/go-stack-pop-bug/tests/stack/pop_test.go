package stack

import "testing"

// TestPopLIFO verifies that Pop returns items in last-in-first-out order and
// that an empty stack reports (0, false).
func TestPopLIFO(t *testing.T) {
	s := New()
	s.Push(1)
	s.Push(2)
	s.Push(3)

	want := []int{3, 2, 1}
	for i, w := range want {
		got, ok := s.Pop()
		if !ok {
			t.Fatalf("pop %d: ok = false, want true", i)
		}
		if got != w {
			t.Fatalf("pop %d: got %d, want %d (LIFO order)", i, got, w)
		}
	}

	if got, ok := s.Pop(); ok || got != 0 {
		t.Fatalf("pop on empty stack: got (%d, %v), want (0, false)", got, ok)
	}
}

// TestPopShrinks checks that popping reduces Len and that elements do not
// resurface after being removed.
func TestPopShrinks(t *testing.T) {
	s := New()
	s.Push(10)
	s.Push(20)

	if _, ok := s.Pop(); !ok {
		t.Fatal("first pop: ok = false, want true")
	}
	if s.Len() != 1 {
		t.Fatalf("after one pop: Len = %d, want 1", s.Len())
	}

	got, ok := s.Pop()
	if !ok || got != 10 {
		t.Fatalf("second pop: got (%d, %v), want (10, true)", got, ok)
	}
	if s.Len() != 0 {
		t.Fatalf("after draining: Len = %d, want 0", s.Len())
	}
}
