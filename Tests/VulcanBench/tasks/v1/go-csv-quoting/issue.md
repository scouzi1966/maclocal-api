# CSV parser mangles escaped quotes inside quoted fields

`csvparse.Parse` (`example.com/csvparse/csvparse`) parses RFC 4180 CSV text. It
handles quoted fields that contain commas and newlines correctly, but it gets
**escaped quotes** wrong.

Per RFC 4180, inside a quoted field a doubled double-quote (`""`) represents a
single literal `"`. The current parser instead treats the first `"` as the end
of the quoted field, so the escaped quote — and everything around it — is parsed
incorrectly:

```go
csvparse.Parse(`"a""b"`)
// want: [][]string{{`a"b`}}
// got:  [][]string{{"ab"}}
```

Fix `Parse` so that a `""` inside a quoted field is emitted as one `"` and the
field continues, rather than ending the field. The existing behavior for
unescaped quoted fields (embedded commas, embedded newlines, multiple records)
must keep working.

The parser is in `csvparse/csv.go`.
