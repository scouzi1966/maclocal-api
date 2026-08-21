# Pinned Tree-sitter external scanners

SwiftPM evaluates dependency manifests with the root package as the current
working directory. Four official grammar manifests use a relative
`FileManager.fileExists` check before adding `src/scanner.c`; as transitive
dependencies, that check omits the scanner and leaves unresolved symbols at
the final executable link.

This target compiles exact copies of only those scanner sources. The generated
parsers remain supplied by their pinned upstream Swift packages. Their source
revisions match `Package.swift`:

- CSS: `tree-sitter/tree-sitter-css@dda5cfc5722c429eaba1c910ca32c2c0c5bb1a3f`
- JavaScript: `tree-sitter/tree-sitter-javascript@58404d8cf191d69f2674a8fd507bd5776f46cb11`
- Python: `tree-sitter/tree-sitter-python@26855eabccb19c6abf499fbc5b8dc7cc9ab8bc64`
- YAML: `tree-sitter-grammars/tree-sitter-yaml@a1c4812a73ec5e089de8e441fdea3a921e8d5079`

`schema.core.c` is a generated include required by the pinned YAML scanner.
Do not update these files without updating the corresponding package revision,
running the grammar ABI tests, and linking the release `afm` executable.
