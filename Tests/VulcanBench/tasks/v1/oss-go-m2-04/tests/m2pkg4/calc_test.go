package m2pkg4_test

import (
  "testing"
  "example.com/bench/m2pkg4"
)

func TestDouble(t *testing.T) {
  if m2pkg4.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if m2pkg4.ID(1) != 1 { t.Fatal() }
}
