"""RFC 6901 conformance — fails on the starting repo, passes after the feature lands."""
import pytest

from jsonptr import JSONPointerError, resolve


def test_escape_slash():
    # "~1" decodes to "/", so this addresses the member named "a/b".
    assert resolve({"a/b": 7}, "/a~1b") == 7


def test_escape_tilde():
    # "~0" decodes to "~".
    assert resolve({"m~n": 8}, "/m~0n") == 8


def test_escape_combined_order():
    # "~01" decodes to "~1" (apply ~1->/ first, then ~0->~).
    assert resolve({"~1": 9}, "/~01") == 9


def test_array_index():
    assert resolve([10, 20, 30], "/1") == 20


def test_array_nested_in_object():
    assert resolve({"xs": [{"y": 99}]}, "/xs/0/y") == 99


def test_missing_member_raises():
    with pytest.raises(JSONPointerError):
        resolve({"a": 1}, "/b")


def test_array_index_out_of_range_raises():
    with pytest.raises(JSONPointerError):
        resolve([1, 2], "/5")


def test_array_dash_raises():
    with pytest.raises(JSONPointerError):
        resolve([1, 2], "/-")


def test_array_leading_zero_raises():
    with pytest.raises(JSONPointerError):
        resolve([1, 2, 3], "/01")


def test_descend_into_scalar_raises():
    with pytest.raises(JSONPointerError):
        resolve({"a": 1}, "/a/b")
