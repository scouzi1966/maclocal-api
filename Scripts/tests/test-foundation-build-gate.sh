#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="$ROOT_DIR/.build/foundation-build-gate-tests"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

make_fixture() {
  local path="$1"
  local payload="$2"
  printf '#!/usr/bin/env bash\nprintf '\''%%s\\n'\'' '\''%s'\''\n' "$payload" > "$path"
  chmod 700 "$path"
}

make_fixture "$WORK_ROOT/complete-afm" \
  '{"build_capabilities":{"foundation_models_compiled":true,"minimum_swift_compiler":"6.4"}}'
make_fixture "$WORK_ROOT/degraded-afm" \
  '{"build_capabilities":{"foundation_models_compiled":false,"minimum_swift_compiler":"6.4"}}'
make_fixture "$WORK_ROOT/legacy-afm" \
  '{"version":"v0.9.18"}'

"$ROOT_DIR/Scripts/check-foundation-models-build.sh" "$WORK_ROOT/complete-afm" >/dev/null

for rejected in "$WORK_ROOT/degraded-afm" "$WORK_ROOT/legacy-afm"; do
  if "$ROOT_DIR/Scripts/check-foundation-models-build.sh" "$rejected" >"$WORK_ROOT/rejected.log" 2>&1; then
    echo "[foundation-build-test] accepted incomplete fixture: $rejected" >&2
    exit 1
  fi
  grep -Fq 'refusing to package a degraded MLX-only executable' "$WORK_ROOT/rejected.log" || {
    echo "[foundation-build-test] rejection was not actionable: $rejected" >&2
    exit 1
  }
done

cat > "$WORK_ROOT/swift" <<'SH'
#!/usr/bin/env bash
printf '%s\n' 'Apple Swift version 6.3 (swiftlang-6.3.0 clang-1700.0.0.0)'
SH
chmod 700 "$WORK_ROOT/swift"

if PATH="$WORK_ROOT:$PATH" \
   "$ROOT_DIR/Scripts/swiftpm-reliable.sh" build \
   >"$WORK_ROOT/old-compiler.log" 2>&1; then
  echo "[foundation-build-test] Swift 6.3 unexpectedly reached SwiftPM" >&2
  exit 1
fi
grep -Fq 'Apple Foundation Models require Swift 6.4 or newer' "$WORK_ROOT/old-compiler.log" || {
  echo "[foundation-build-test] old-compiler rejection was not actionable" >&2
  exit 1
}
grep -Fq 'Refusing to produce a degraded MLX-only build' "$WORK_ROOT/old-compiler.log" || {
  echo "[foundation-build-test] old compiler did not reject degraded output" >&2
  exit 1
}

echo "[foundation-build-test] complete build accepted; degraded binaries and Swift 6.3 rejected"
