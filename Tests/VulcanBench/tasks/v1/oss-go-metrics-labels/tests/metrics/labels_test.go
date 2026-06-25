package metrics_test

import (
  "testing"
  "example.com/bench/metrics"
)

func TestInc(t *testing.T) { r := metrics.New(); r.Inc("x") }

func TestLabelKeyStable(t *testing.T) {
  labels := map[string]string{"e": "5", "b": "2", "d": "4", "a": "1", "c": "3", "f": "6"}
  want := "h,a=1,b=2,c=3,d=4,e=5,f=6"
  // Go randomizes map iteration per call, so unsorted output fails reliably
  // across repeated calls while correctly sorted output is identical every time.
  for i := 0; i < 8; i++ {
    if got := metrics.Key("h", labels); got != want {
      t.Fatalf("call %d: got %q want %q", i, got, want)
    }
  }
}
