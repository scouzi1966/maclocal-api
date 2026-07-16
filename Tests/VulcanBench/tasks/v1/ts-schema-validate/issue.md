# Implement object validation against a schema

The `validate(obj, schema)` function in `src/validate.ts` is currently a stub: it
returns an empty array for every input, so it treats every object as valid. We
need it to actually check objects against the schema.

The types live in `src/types.ts`:

- A `Schema` maps field names to a `FieldSpec`.
- A `FieldSpec` has a `type` (`"string" | "number" | "boolean"`) and an optional
  `required` flag (defaults to falsy/optional).
- A `ValidationError` has a `field`, a `kind` (`"missing"` or `"type"`), and a
  human-readable `message`.

Expected behavior of `validate(obj, schema)`:

- For each field declared in the schema:
  - If the field is `required` and is absent from `obj`, report a `ValidationError`
    for that field with `kind: "missing"`.
  - If the field is present in `obj`, its value must match the declared `type`
    (`typeof value === spec.type`). If it does not, report a `ValidationError`
    for that field with `kind: "type"`.
  - An optional field that is absent is fine — no error.
  - An absent required field should NOT additionally produce a type error.
- Return the full list of errors (one per failing field). The order does not
  matter. An empty array means the object is valid.

Fields present on `obj` but not declared in the schema should be ignored.
