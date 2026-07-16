package money

import "testing"

func sum(parts []Money) int64 {
	var total int64
	for _, p := range parts {
		total += p.Cents()
	}
	return total
}

func cents(parts []Money) []int64 {
	out := make([]int64, len(parts))
	for i, p := range parts {
		out[i] = p.Cents()
	}
	return out
}

func equal(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// $0.10 into 3 parts: 10 = 4 + 3 + 3; the extra cent goes to the first part.
func TestAllocateUnevenRemainder(t *testing.T) {
	got := cents(money10().Allocate(3))
	want := []int64{4, 3, 3}
	if !equal(got, want) {
		t.Fatalf("Allocate(3) = %v, want %v", got, want)
	}
}

// When n divides the amount evenly every part is identical.
func TestAllocateEvenSplit(t *testing.T) {
	got := cents(FromCents(100).Allocate(4))
	want := []int64{25, 25, 25, 25}
	if !equal(got, want) {
		t.Fatalf("Allocate(4) = %v, want %v", got, want)
	}
}

// The parts must always sum back to the original amount, never losing or
// inventing cents, across a range of amounts and divisors.
func TestAllocatePreservesTotal(t *testing.T) {
	for amount := int64(0); amount <= 101; amount++ {
		for n := 1; n <= 7; n++ {
			parts := FromCents(amount).Allocate(n)
			if len(parts) != n {
				t.Fatalf("amount=%d n=%d: got %d parts, want %d", amount, n, len(parts), n)
			}
			if s := sum(parts); s != amount {
				t.Fatalf("amount=%d n=%d: parts sum to %d, want %d (%v)", amount, n, s, amount, cents(parts))
			}
		}
	}
}

// The remainder is spread one extra cent to the first r parts only, so parts
// differ by at most one cent and the larger parts come first.
func TestAllocateLargestRemainderFirst(t *testing.T) {
	parts := cents(FromCents(7).Allocate(3)) // 7 = 3 + 2 + 2
	want := []int64{3, 2, 2}
	if !equal(parts, want) {
		t.Fatalf("Allocate(3) of 7c = %v, want %v", parts, want)
	}
	for i := 1; i < len(parts); i++ {
		if parts[i] > parts[i-1] {
			t.Fatalf("parts not in non-increasing order: %v", parts)
		}
	}
}

// A single part returns the whole amount unchanged.
func TestAllocateSinglePart(t *testing.T) {
	got := cents(money10().Allocate(1))
	want := []int64{10}
	if !equal(got, want) {
		t.Fatalf("Allocate(1) = %v, want %v", got, want)
	}
}

func money10() Money { return FromCents(10) }
