# `titleize` does not capitalize words that start with an accented letter

`inflection.titleize(word)` is meant to capitalize the first letter of every
word to produce a pretty title. It works for ASCII text, but a word that begins
with a non-ASCII letter (e.g. an accented character) is left lowercased:

```python
>>> inflection.titleize("ana índia")
'Ana índia'      # want: 'Ana Índia'
```

The first letter of `índia` (`í`) is not uppercased because the capitalization
only recognizes words starting with a plain `a`–`z` letter. Titles containing
international characters therefore come out wrong.

Fix `titleize` so that the first letter of every word is capitalized regardless
of whether it is an ASCII letter, so that `titleize("ana índia")` returns
`"Ana Índia"`. Words that already title-case correctly (ASCII words, and words
whose accent is not the first letter such as `café`) must keep working.

The function lives in `inflection.py`.

> Upstream issue: https://github.com/jpvanhal/inflection/issues/33
