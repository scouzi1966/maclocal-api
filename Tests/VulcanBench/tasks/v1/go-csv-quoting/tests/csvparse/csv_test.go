package csvparse

import (
	"reflect"
	"testing"
)

// --- fail_to_pass: escaped quotes ("" -> ") are mishandled in the starting repo ---

func TestEscapedQuote(t *testing.T) {
	got := Parse(`"a""b"`)
	want := [][]string{{`a"b`}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

func TestEscapedQuoteInContext(t *testing.T) {
	got := Parse(`a,"she said ""hi""",b`)
	want := [][]string{{"a", `she said "hi"`, "b"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

func TestOnlyEscapedQuotes(t *testing.T) {
	got := Parse(`""""`) // a quoted field containing a single double-quote
	want := [][]string{{`"`}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

// --- pass_to_pass: quoting behavior that already works (regression guard) ---

func TestSimpleRecord(t *testing.T) {
	got := Parse("a,b,c")
	want := [][]string{{"a", "b", "c"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

func TestQuotedFieldWithComma(t *testing.T) {
	got := Parse(`"x,y",z`)
	want := [][]string{{"x,y", "z"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

func TestQuotedFieldWithNewline(t *testing.T) {
	got := Parse("\"line1\nline2\",c")
	want := [][]string{{"line1\nline2", "c"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}

func TestMultipleRecords(t *testing.T) {
	got := Parse("a,b\nc,d")
	want := [][]string{{"a", "b"}, {"c", "d"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Parse = %#v, want %#v", got, want)
	}
}
