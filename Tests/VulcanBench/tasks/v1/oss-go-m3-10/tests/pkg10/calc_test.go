package pkg10_test

import (
  "testing"
  "example.com/bench/pkg10"
)

func TestDouble(t *testing.T) {
  if pkg10.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if pkg10.ID(1) != 1 { t.Fatal() }
}
