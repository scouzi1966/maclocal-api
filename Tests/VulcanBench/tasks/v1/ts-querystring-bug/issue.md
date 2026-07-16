# parseQuery mangles encoded values and drops repeated keys

`parseQuery(qs: string)` in `src/parse.ts` turns a URL query string into a map
of keys to values. Two things are broken:

1. **Values are not URL-decoded.** A query like `q=hello%20world` should yield
   `{ q: "hello world" }`, but instead the value keeps the literal `%20` (and a
   `+`, which should also become a space, survives literally too). Keys are
   already decoded via the helper in `src/decode.ts`; values should be decoded
   the same way.

2. **Repeated keys overwrite each other.** When the same key appears more than
   once — e.g. `tag=a&tag=b&tag=c` — only the last value is kept. All values for
   a repeated key should instead be collected into an array, in the order they
   appear: `{ tag: ["a", "b", "c"] }`. A key that appears exactly once should
   still map to a plain string.

Expected behavior:

- `parseQuery("q=hello%20world")` -> `{ q: "hello world" }`
- `parseQuery("q=a+b")` -> `{ q: "a b" }`
- `parseQuery("tag=a&tag=b&tag=c")` -> `{ tag: ["a", "b", "c"] }`
- `parseQuery("a=1")` -> `{ a: "1" }` (single value stays a string)

There is a decoding helper in `src/parse.ts`'s sibling module `src/decode.ts`
(`decodeComponent`) that already handles `+` and percent-encoding; use it.
