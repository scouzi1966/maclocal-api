# Stack.Pop returns the wrong element

The `stack` package (`example.com/stackdemo/stack`) is documented as a
last-in-first-out (LIFO) stack: the most recently pushed value should be the
first one returned by `Pop`.

In practice it behaves like a queue. After pushing `1`, `2`, `3`, calling
`Pop` returns `1` first instead of `3`, so values come back in the order they
went in rather than the reverse.

Expected behavior:

- `Pop` removes and returns the value on the **top** of the stack — the most
  recently pushed one.
- Repeated `Pop` calls walk the stack in reverse insertion order.
- Each `Pop` shrinks the stack: `Len` decreases by one and a removed value
  never resurfaces.
- `Pop` on an empty stack returns `(0, false)`.

`Push` and `Len` live in `stack/stack.go`; `Pop` lives in `stack/pop.go`.
