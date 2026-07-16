"""Regression test for inflection.titleize() and accented words (issue #33).

The accented cases come straight from the upstream regression test; the ASCII
cases guard the behavior that already worked.
"""
import inflection


# --- fail_to_pass: words that START with a non-ASCII letter were not capitalized ---


def test_titleize_leading_accented_word():
    assert inflection.titleize("ana índia") == "Ana Índia"


def test_titleize_already_titlecased_accented_word():
    assert inflection.titleize("Ana Índia") == "Ana Índia"


# --- pass_to_pass: behavior that holds before and after the fix ---


def test_titleize_ascii_phrase():
    assert inflection.titleize("man from the boondocks") == "Man From The Boondocks"


def test_titleize_camelcase():
    assert inflection.titleize("TheManWithoutAPast") == "The Man Without A Past"


def test_titleize_midword_accent():
    # "café" starts with an ASCII letter, so it capitalized correctly even before.
    assert inflection.titleize("café del mar") == "Café Del Mar"
