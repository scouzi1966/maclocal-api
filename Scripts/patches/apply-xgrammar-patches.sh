#!/bin/bash
# Fix xgrammar v0.1.32 C++17 ODR linker error on macOS.
#
# static const class members are not implicitly inline in C++17 —
# they cause "Undefined symbols" linker errors when ODR-used in lambdas.
# Changing to static constexpr makes them implicitly inline.
#
# Upstream bug: https://github.com/mlc-ai/xgrammar/issues/TBD
# Affects: vendor/xgrammar/cpp/grammar_functor.cc (GrammarFSMHasherImpl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="$ROOT_DIR/vendor/xgrammar/cpp/grammar_functor.cc"

if [ ! -f "$TARGET" ]; then
    echo "[SKIP] xgrammar not found (submodule not initialized?)"
    exit 0
fi

if grep -q 'static constexpr int16_t kNotEndStateFlag' "$TARGET"; then
    echo "[OK] xgrammar constexpr patch already applied"
    exit 0
fi

sed -i '' \
    's/static const int16_t kNotEndStateFlag/static constexpr int16_t kNotEndStateFlag/;
     s/static const int16_t kEndStateFlag/static constexpr int16_t kEndStateFlag/;
     s/static const int16_t kSelfRecursionFlag/static constexpr int16_t kSelfRecursionFlag/;
     s/static const int16_t kSimpleCycleFlag/static constexpr int16_t kSimpleCycleFlag/;
     s/static const int16_t kUnKnownFlag/static constexpr int16_t kUnKnownFlag/' \
    "$TARGET"

echo "[OK] xgrammar constexpr patch applied"
