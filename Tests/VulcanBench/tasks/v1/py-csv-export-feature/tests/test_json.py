"""JSON exporter behavior — must hold before and after the change."""
import json

from report import Row, to_json


def test_to_json_basic():
    rows = [Row({"name": "Ada", "age": 36}), Row({"name": "Linus", "age": 54})]
    assert json.loads(to_json(rows)) == [
        {"name": "Ada", "age": 36},
        {"name": "Linus", "age": 54},
    ]


def test_to_json_uses_first_row_columns():
    rows = [Row({"a": 1, "b": 2}), Row({"a": 3, "b": 4})]
    assert json.loads(to_json(rows)) == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_to_json_empty():
    assert json.loads(to_json([])) == []
