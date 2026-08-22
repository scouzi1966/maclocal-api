#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT_DIR/Package.swift"

require_manifest_text() {
    if ! grep -Fq "$1" "$MANIFEST"; then
        echo "tree-sitter integrity: missing Package.swift pin: $1" >&2
        exit 1
    fi
}

check_digest() {
    local expected="$1"
    local relative="$2"
    local actual
    actual="$(shasum -a 256 "$ROOT_DIR/$relative" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        echo "tree-sitter integrity: $relative does not match its pinned upstream source" >&2
        exit 1
    fi
}

require_manifest_text '.package(url: "https://github.com/tree-sitter/tree-sitter.git", exact: "0.25.10")'
require_manifest_text '.package(url: "https://github.com/tree-sitter/swift-tree-sitter.git", exact: "0.25.0")'

check_digest 18dae8c8c4f515f28a3dc7ffb5bda259b06013a752921dc411a2fad8ecf78988 Sources/AFMTreeSitterScanners/css_scanner.c
check_digest b3d3f64284d97bf80749c026862427782cf7ecc0b7dc094e6698ab311c9a42c7 Sources/AFMTreeSitterScanners/javascript_scanner.c
check_digest 6db82134ac2d4c90a1a1475487a625cface02662ebda9b7478cad9c7147e9afe Sources/AFMTreeSitterScanners/python_scanner.c
check_digest a510c0ca699cf3853bd2192bf103e11132414bc002fe952a88a7106ffb5d44e9 Sources/AFMTreeSitterScanners/yaml_scanner.c
check_digest 91e56c3f3ae6fad1803b739ce4eb4568a782a9aec710f733db2f3e38d407a4d5 Sources/AFMTreeSitterScanners/schema.core.c
check_digest 180b893c8734778fd32f372dfbc27bd6ad1cd2221f26150b31256ff6716320d2 Sources/AFMTreeSitterScanners/tree_sitter/parser.h
check_digest 5bdf6ed1a78e3409fd443e085ca967a64c188a5d082aaf7f819bccd53a471c94 Sources/AFMTreeSitterScanners/tree_sitter/array.h
check_digest b29c1c9fb7cc82f58c84b376df1297d6e2737a1d655fd356db0859e3c29c2fea Sources/AFMTreeSitterScanners/tree_sitter/alloc.h

if [ "$#" -gt 0 ]; then
    BINARY="$1"
    if [ ! -x "$BINARY" ]; then
        echo "tree-sitter integrity: binary is not executable: $BINARY" >&2
        exit 1
    fi
    SYMBOLS="$(nm -gU "$BINARY")"
    for language in bash c cpp c_sharp css diff go html java javascript json kotlin markdown php python ruby rust sql swift toml tsx typescript yaml; do
        if ! grep -q "_tree_sitter_${language}$" <<<"$SYMBOLS"; then
            echo "tree-sitter integrity: linked binary is missing tree_sitter_${language}" >&2
            exit 1
        fi
    done
fi

echo "tree-sitter integrity: pinned sources and linked grammars verified"
