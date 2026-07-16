package money

import "testing"

func TestMoneyAdd(t *testing.T) {
	got := FromCents(150).Add(FromCents(75)).Cents()
	if got != 225 {
		t.Fatalf("Add = %d, want 225", got)
	}
	// Adding a debt subtracts.
	if got := FromCents(100).Add(FromCents(-30)).Cents(); got != 70 {
		t.Fatalf("Add(debt) = %d, want 70", got)
	}
}

func TestMoneyString(t *testing.T) {
	cases := []struct {
		cents int64
		want  string
	}{
		{105, "$1.05"},
		{100, "$1.00"},
		{9, "$0.09"},
		{-30, "-$0.30"},
		{0, "$0.00"},
	}
	for _, c := range cases {
		if got := FromCents(c.cents).String(); got != c.want {
			t.Fatalf("FromCents(%d).String() = %q, want %q", c.cents, got, c.want)
		}
	}
}
