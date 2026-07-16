package m2pkg7_test

import (
  "testing"
  "example.com/bench/m2pkg7"
)

func TestDouble(t *testing.T) {
  if m2pkg7.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if m2pkg7.ID(1) != 1 { t.Fatal() }
}
