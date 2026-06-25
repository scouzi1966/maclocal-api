package pkg13_test

import (
  "testing"
  "example.com/bench/pkg13"
)

func TestDouble(t *testing.T) {
  if pkg13.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if pkg13.ID(1) != 1 { t.Fatal() }
}
