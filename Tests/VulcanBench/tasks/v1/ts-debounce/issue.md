# Implement `debounce`

`src/debounce.ts` exports a `debounce(fn, wait, options?)` helper, but it is an
unimplemented skeleton — it returns a callable with a `cancel` method that never
actually invokes `fn`. Implement it.

`debounce` returns a wrapped version of `fn` that delays invocation until `wait`
milliseconds have passed without another call:

- **Trailing edge (default).** With the default options, calling the debounced
  function repeatedly only invokes `fn` once, `wait` ms after the *last* call,
  using the **most recent** arguments. It must not invoke `fn` before `wait` ms
  have elapsed.
- **Leading edge.** With `{ leading: true }`, `fn` is invoked immediately on the
  first call of a burst. With `{ leading: true, trailing: false }` it fires only
  on that leading edge and not at the end of the window.
- **cancel().** The returned function has a `.cancel()` method that discards any
  pending trailing invocation. After cancelling, the debouncer must still work
  for subsequent calls.

Options default to `{ leading: false, trailing: true }`.

Use `setTimeout`/`clearTimeout` for scheduling — the tests drive virtual time
with Node's mock timers, so no real waiting occurs. The function lives in
`src/debounce.ts`.
