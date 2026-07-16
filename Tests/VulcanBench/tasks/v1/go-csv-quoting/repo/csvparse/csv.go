// Package csvparse parses RFC 4180 CSV text into records of fields.
//
// NOTE: the handling of escaped quotes is buggy — see issue.md.
package csvparse

import "strings"

// Parse parses a CSV document into a slice of records, each a slice of fields.
//
// A field may be quoted with double-quotes; inside a quoted field, commas and
// newlines are literal. Records are separated by "\n" (a leading "\r" is
// stripped). The final record does not require a trailing newline.
func Parse(s string) [][]string {
	s = strings.ReplaceAll(s, "\r\n", "\n")

	var records [][]string
	var record []string
	var field strings.Builder
	inQuotes := false

	for i := 0; i < len(s); i++ {
		c := s[i]
		if inQuotes {
			if c == '"' {
				inQuotes = false
				continue
			}
			field.WriteByte(c)
			continue
		}
		switch c {
		case '"':
			inQuotes = true
		case ',':
			record = append(record, field.String())
			field.Reset()
		case '\n':
			record = append(record, field.String())
			records = append(records, record)
			record = nil
			field.Reset()
		default:
			field.WriteByte(c)
		}
	}
	record = append(record, field.String())
	records = append(records, record)
	return records
}
