package errwrap_test

import (
  "testing"
  "example.com/bench/errwrap"
)

func TestDouble(t *testing.T) {
  if errwrap.Double(2) != 4 { t.Fatal("want 4") }
}

func TestID(t *testing.T) {
  if errwrap.ID(1) != 1 { t.Fatal() }
}
