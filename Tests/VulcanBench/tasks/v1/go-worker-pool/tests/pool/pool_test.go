package pool

import (
	"reflect"
	"testing"
)

func sq(x int) int { return x * x }

// --- fail_to_pass: run under -race so a fix that introduces a data race also fails ---

func TestMapReturnsAllResults(t *testing.T) {
	got := Map([]int{1, 2, 3, 4, 5, 6}, 3, sq)
	want := []int{1, 4, 9, 16, 25, 36}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Map = %v, want %v", got, want)
	}
}

func TestMapConcurrentStress(t *testing.T) {
	const n = 200
	in := make([]int, n)
	for i := range in {
		in[i] = i
	}
	got := Map(in, 8, func(x int) int { return x + 1 })
	if len(got) != n {
		t.Fatalf("len(got) = %d, want %d", len(got), n)
	}
	for i := 0; i < n; i++ {
		if got[i] != i+1 {
			t.Fatalf("got[%d] = %d, want %d", i, got[i], i+1)
		}
	}
}

// --- pass_to_pass: invariants that hold before and after the fix ---

func TestMapLengthMatchesInput(t *testing.T) {
	if got := Map([]int{1, 2, 3}, 2, sq); len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
}

func TestMapEmpty(t *testing.T) {
	if got := Map([]int{}, 4, sq); len(got) != 0 {
		t.Fatalf("len(got) = %d, want 0", len(got))
	}
}

func TestMapFirstElement(t *testing.T) {
	if got := Map([]int{7, 8, 9}, 2, sq); got[0] != 49 {
		t.Fatalf("got[0] = %d, want 49", got[0])
	}
}
