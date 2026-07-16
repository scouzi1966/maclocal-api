package m2pkg1_test

import (
  "testing"
  "example.com/bench/m2pkg1"
)

func TestDouble(t *testing.T) {
  if m2pkg1.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if m2pkg1.ID(1) != 1 { t.Fatal() }
}
