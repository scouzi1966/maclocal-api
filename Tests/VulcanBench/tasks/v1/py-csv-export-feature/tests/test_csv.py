"""CSV exporter behavior — fails on the starting repo, passes after the change."""
from report import Row, to_csv


def test_header_and_simple_rows():
    rows = [Row({"name": "Ada", "city": "London"}), Row({"name": "Linus", "city": "Helsinki"})]
    assert to_csv(rows) == "name,city\r\nAda,London\r\nLinus,Helsinki\r\n"


def test_empty_rows_is_empty_string():
    assert to_csv([]) == ""


def test_field_with_comma_is_quoted():
    rows = [Row({"item": "apples, oranges", "qty": 3})]
    assert to_csv(rows) == 'item,qty\r\n"apples, oranges",3\r\n'


def test_double_quote_is_doubled_and_quoted():
    rows = [Row({"text": 'she said "hi"'})]
    assert to_csv(rows) == 'text\r\n"she said ""hi"""\r\n'


def test_newline_in_field_is_quoted():
    rows = [Row({"note": "line1\nline2"})]
    assert to_csv(rows) == 'note\r\n"line1\nline2"\r\n'


def test_carriage_return_in_field_is_quoted():
    rows = [Row({"note": "a\rb"})]
    assert to_csv(rows) == 'note\r\n"a\rb"\r\n'


def test_none_is_empty_field():
    rows = [Row({"a": None, "b": "x"})]
    assert to_csv(rows) == "a,b\r\n,x\r\n"


def test_integer_value_is_stringified():
    rows = [Row({"n": 7})]
    assert to_csv(rows) == "n\r\n7\r\n"


def test_columns_follow_first_row_order():
    rows = [Row({"z": 1, "a": 2}), Row({"z": 3, "a": 4})]
    assert to_csv(rows) == "z,a\r\n1,2\r\n3,4\r\n"
