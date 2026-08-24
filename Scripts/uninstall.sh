#!/usr/bin/env bash
set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

usage() {
  cat <<'USAGE'
Usage: Scripts/uninstall.sh [--prefix PATH]

Remove files installed by ./build.sh --install. INSTALL_PREFIX defaults to
/usr/local and can also be supplied through the environment.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || { echo "--prefix requires a path" >&2; exit 2; }
      INSTALL_PREFIX="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$INSTALL_PREFIX" && "$INSTALL_PREFIX" != "/" ]] || {
  echo "[uninstall] Refusing unsafe INSTALL_PREFIX: ${INSTALL_PREFIX:-<empty>}" >&2
  exit 1
}

permission_probe="$INSTALL_PREFIX"
while [[ ! -e "$permission_probe" ]]; do
  parent="$(dirname "$permission_probe")"
  [[ "$parent" != "$permission_probe" ]] || break
  permission_probe="$parent"
done

use_sudo=false
if [[ ! -w "$permission_probe" ]]; then
  use_sudo=true
fi

run_remove() {
  if $use_sudo; then
    sudo "$@"
  else
    "$@"
  fi
}

remove_owned_path() {
  local path="$1"
  if [[ -L "$path" || -f "$path" ]]; then
    run_remove rm -f -- "$path"
  elif [[ -d "$path" ]]; then
    run_remove rm -rf -- "$path"
  fi
}

remove_owned_path "$INSTALL_PREFIX/bin/afm"
for bundle in MacLocalAPI_AFMKit.bundle MacLocalAPI_AFMEvaluationHost.bundle AFMKit_AFMKitMLX.bundle AFMKit_AFMKitDwarfStar.bundle; do
  remove_owned_path "$INSTALL_PREFIX/bin/$bundle"
  remove_owned_path "$INSTALL_PREFIX/libexec/afm/$bundle"
done
remove_owned_path "$INSTALL_PREFIX/share/afm/webui/index.html.gz"

# Remove only empty AFM-owned directories. Unrelated files keep their parent
# directories and are never traversed or deleted.
for directory in \
  "$INSTALL_PREFIX/share/afm/webui" \
  "$INSTALL_PREFIX/share/afm" \
  "$INSTALL_PREFIX/libexec/afm"; do
  if [[ -d "$directory" ]]; then
    run_remove rmdir -- "$directory" 2>/dev/null || true
  fi
done

echo "[uninstall] Removed AFM-owned files from $INSTALL_PREFIX"
