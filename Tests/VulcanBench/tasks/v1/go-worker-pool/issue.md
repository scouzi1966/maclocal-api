# Worker pool drops the last input

`pool.Map(inputs, workers, fn)` is meant to apply `fn` to every element of
`inputs` concurrently, using up to `workers` goroutines, and return the results
in the **same order** as `inputs`:

```go
pool.Map([]int{1, 2, 3, 4, 5, 6}, 3, func(x int) int { return x * x })
// want: [1 4 9 16 25 36]
```

Right now the last element is never processed — its slot comes back as the zero
value (`0`) instead of `fn` applied to the last input. The result slice is the
right length, and every element except the last is correct.

Fix `Map` so that **all** inputs are processed and every result lands in the
correct position. The function must stay correct under concurrency — the tests
run with Go's race detector (`go test -race`), so a fix that processes all
inputs but introduces a data race will still fail.

The code is in `pool/pool.go`.
