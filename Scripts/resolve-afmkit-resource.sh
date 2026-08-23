#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="metallib"
SEARCH_ROOT=""

usage() {
  cat <<'USAGE'
Usage: Scripts/resolve-afmkit-resource.sh [--source|--metallib|--bundle-dir] [search-root]

Resolves AFMKit-owned MLX resources without relying on maclocal-api shadow
sources. A caller-supplied MACAFM_MLX_METALLIB always wins for metallib modes.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) MODE="source"; shift ;;
    --metallib) MODE="metallib"; shift ;;
    --bundle-dir) MODE="bundle-dir"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      [[ -z "$SEARCH_ROOT" ]] || { echo "Only one search root is supported" >&2; exit 2; }
      SEARCH_ROOT="$1"
      shift
      ;;
  esac
done

if [[ "$MODE" != "bundle-dir" && -n "${MACAFM_MLX_METALLIB:-}" && -f "$MACAFM_MLX_METALLIB" ]]; then
  printf '%s\n' "$MACAFM_MLX_METALLIB"
  exit 0
fi

afmkit_roots=()
[[ -n "${MACLOCAL_AFMKIT_PATH:-}" ]] && afmkit_roots+=("$MACLOCAL_AFMKIT_PATH")
afmkit_roots+=("$ROOT_DIR/.build/checkouts/AFMKit")

if [[ "$MODE" == "source" ]]; then
  for afmkit_root in "${afmkit_roots[@]}"; do
    for candidate in \
      "$afmkit_root/Packages/AFMKitMLX/Sources/AFMKitMLX/Resources/default.metallib" \
      "$afmkit_root/Sources/AFMKitMLX/Resources/default.metallib"; do
      if [[ -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        exit 0
      fi
    done
  done
  echo "AFMKit MLX source resource was not found. Run 'swift package resolve' first or set MACLOCAL_AFMKIT_PATH." >&2
  exit 1
fi

search_roots=()
[[ -n "$SEARCH_ROOT" ]] && search_roots+=("$SEARCH_ROOT")
search_roots+=(
  "$ROOT_DIR/.build/arm64-apple-macosx/release"
  "$ROOT_DIR/.build/arm64-apple-macosx/debug"
  "$ROOT_DIR/.build/release"
  "$ROOT_DIR/.build/debug"
  "$ROOT_DIR/.build/out/Products/Release"
  "$ROOT_DIR/.build/out/Products/Debug"
)

for root in "${search_roots[@]}"; do
  [[ -d "$root" ]] || continue
  for bundle_name in AFMKit_AFMKitMLX.bundle MacLocalAPI_AFMKitMLX.bundle; do
    bundle="$root/$bundle_name"
    for resource in "$bundle/default.metallib" "$bundle/Contents/Resources/default.metallib"; do
      if [[ -f "$resource" ]]; then
        if [[ "$MODE" == "bundle-dir" ]]; then
          printf '%s\n' "$bundle"
        else
          printf '%s\n' "$resource"
        fi
        exit 0
      fi
    done
  done
done

echo "Built AFMKit_AFMKitMLX.bundle was not found${SEARCH_ROOT:+ under $SEARCH_ROOT}." >&2
exit 1
