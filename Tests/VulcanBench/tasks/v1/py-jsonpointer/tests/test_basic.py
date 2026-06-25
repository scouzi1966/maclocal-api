"""Pointer behavior that already works — must hold before and after (regression guard)."""
import pytest

from jsonptr import JSONPointerError, resolve


def test_empty_pointer_is_whole_document():
    doc = {"a": 1}
    assert resolve(doc, "") is doc


def test_simple_member():
    assert resolve({"a": 1, "b": 2}, "/a") == 1


def test_nested_members():
    assert resolve({"a": {"b": {"c": 42}}}, "/a/b/c") == 42


def test_non_slash_pointer_is_rejected():
    with pytest.raises(JSONPointerError):
        resolve({"a": 1}, "nope")
