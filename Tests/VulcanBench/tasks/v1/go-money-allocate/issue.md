# Add Money.Allocate to split an amount into n parts

The `money` package (`example.com/moneydemo/money`) represents money as a whole
number of integer cents via the `Money` type. It already supports `Add` and
`String`, plus `FromCents` and `Cents` helpers.

We need a way to split a `Money` value into several parts — for example, to
divide a bill or a payout among several people — without ever losing or
inventing a cent. Float division (`amount / n`) is unacceptable because rounding
each share independently can make the parts fail to sum back to the original.

Please add a method:

```go
func (m Money) Allocate(n int) []Money
```

with the following semantics (assume `n >= 1`):

- It returns exactly `n` parts.
- The parts sum **exactly** to the original amount — no cent is lost or
  invented, for any amount and any `n`.
- The parts are as even as possible: every two parts differ by at most one cent.
- Let `r = cents mod n` be the leftover cents after an even integer division.
  The base share `cents / n` (integer division) goes to every part, and the `r`
  leftover cents are distributed one extra cent each to the **first `r` parts**.
  So the larger parts come first and the result is in non-increasing order.

Examples (amounts in cents):

- `FromCents(10).Allocate(3)` → parts of `4, 3, 3` (sum 10).
- `FromCents(100).Allocate(4)` → parts of `25, 25, 25, 25`.
- `FromCents(7).Allocate(3)` → parts of `3, 2, 2`.
- `FromCents(10).Allocate(1)` → a single part of `10`.

A stub already exists in `money/allocate.go`; replace its body with a working
implementation.
