// Package money represents monetary amounts as a whole number of cents,
// avoiding floating-point rounding error.
package money

import "fmt"

// Money is an amount of money measured in integer cents. A value of 100
// represents $1.00; negative values represent debts.
type Money struct {
	cents int64
}

// FromCents constructs a Money value from a whole number of cents.
func FromCents(cents int64) Money {
	return Money{cents: cents}
}

// Cents returns the amount as a whole number of cents.
func (m Money) Cents() int64 {
	return m.cents
}

// Add returns the sum of two amounts.
func (m Money) Add(other Money) Money {
	return Money{cents: m.cents + other.cents}
}

// String formats the amount as a signed dollar string, e.g. "$1.05" or "-$0.30".
func (m Money) String() string {
	c := m.cents
	sign := ""
	if c < 0 {
		sign = "-"
		c = -c
	}
	return fmt.Sprintf("%s$%d.%02d", sign, c/100, c%100)
}
