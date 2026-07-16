# `deepMerge` merges arrays wrong, mutates its inputs, and is unsafe

`deepMerge(target, source)` in `src/merge.ts` recursively merges `source` into
`target`. It has three problems:

1. **Arrays are merged element-by-element** instead of replaced. Merging
   `{ list: [1, 2, 3] }` with `{ list: [9] }` should produce `{ list: [9] }`
   (the source array wins), but it currently produces `{ list: [9, 2, 3] }`.
2. **It mutates its inputs.** `deepMerge` should return a brand-new object and
   leave both `target` and `source` untouched.
3. **It is vulnerable to prototype pollution.** Keys such as `__proto__`,
   `constructor`, and `prototype` coming from `source` must never be copied,
   otherwise a crafted payload like `{"__proto__": {"polluted": "yes"}}` can
   pollute `Object.prototype`.

Rewrite `deepMerge` so that:

- only **plain objects** are merged recursively; arrays and primitive values
  from `source` replace the value in `target`,
- a new object is returned and the inputs are never mutated, and
- the dangerous keys `__proto__`, `constructor`, and `prototype` are skipped.

The function lives in `src/merge.ts`.
