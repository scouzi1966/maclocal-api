# Finish the RFC 6901 JSON Pointer resolver

`jsonptr.resolve(document, pointer)` resolves a [JSON Pointer (RFC 6901)](https://www.rfc-editor.org/rfc/rfc6901)
against an already-decoded JSON document (Python dicts / lists / scalars). The
current implementation only walks object members — it does not handle escaping,
arrays, or the error cases the RFC requires. Finish it.

A JSON Pointer is either the empty string (which refers to the whole document)
or a sequence of `/`-prefixed reference tokens. Implement the following:

- **Escaping.** Within a token, `~1` decodes to `/` and `~0` decodes to `~`.
  These must be applied in that order: first `~1` → `/`, then `~0` → `~` (so
  `~01` decodes to `~1`).
- **Arrays.** When the current value is a list, the token must be either a
  non-negative integer index with **no leading zeros** (`0`, `1`, `12`, …) or
  `-`. A valid in-range index selects that element.
- **Errors** (raise `jsonptr.JSONPointerError`):
  - a pointer that is non-empty and does not start with `/` (already handled),
  - a member that does not exist on an object,
  - an array index that is out of range,
  - an array token that is `-` (it refers to the nonexistent element just past
    the end), or a malformed index such as `01`,
  - a token applied to a scalar (you cannot descend into a number/string/bool/null).

The resolver lives in `jsonptr/pointer.py`; `JSONPointerError` is in
`jsonptr/errors.py`.
