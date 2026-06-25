# Add a CSV exporter

The `report` library can already serialize a list of rows to JSON via
`report.exporter.to_json`. We need a matching CSV exporter, `to_csv`, so reports
can be opened in spreadsheet tools.

`report.exporter.to_csv(rows)` currently raises `NotImplementedError`. Implement
it so it returns an RFC 4180 CSV string.

Requirements:

- The column order is taken from the first row's insertion order (same as
  `to_json`). The first output line is the header row of column names; each row
  follows on its own line.
- Records are separated by CRLF (`\r\n`), including a trailing CRLF after the
  last record, per RFC 4180.
- A field must be quoted with double-quotes if it contains a comma, a
  double-quote, a carriage return, or a line feed. Any double-quote inside a
  quoted field is escaped by doubling it (`"` becomes `""`).
- Fields that need no quoting are written as-is.
- Values are stringified (e.g. an integer `7` becomes `7`); a `None` value is
  written as an empty field.
- An empty `rows` list produces an empty string (`""`).

The `Row` model lives in `report/models.py`; the exporters live in
`report/exporter.py`.
