#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHANNEL=""
BASE_VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stable) CHANNEL="stable"; shift ;;
    --nightly) CHANNEL="nightly"; shift ;;
    --version)
      [[ $# -ge 2 ]] || { echo "--version requires a value" >&2; exit 2; }
      BASE_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 (--stable|--nightly) [--version VERSION]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CHANNEL" ]] || { echo "Select --stable or --nightly" >&2; exit 2; }

if [[ -z "$BASE_VERSION" ]]; then
  BASE_VERSION="$(grep 'static let version' Sources/AFMKit/BuildInfo.swift \
    | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')"
fi

if [[ "$CHANNEL" == "nightly" ]]; then
  PACKAGE_ROOT="macafm_next"
  DIST_NAME="macafm_next"
  VERSION="${BASE_VERSION}.dev$(date -u +%Y%m%d)"
else
  PACKAGE_ROOT="macafm"
  DIST_NAME="macafm"
  VERSION="$BASE_VERSION"
fi

BACKUP_DIR="$ROOT_DIR/.build/native-wheel-source-backup/$PACKAGE_ROOT"
rm -rf "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp "$PACKAGE_ROOT/__init__.py" "$BACKUP_DIR/__init__.py"
cp pyproject.toml "$BACKUP_DIR/pyproject.toml"
cp pyproject-next.toml "$BACKUP_DIR/pyproject-next.toml"
if [[ -d "$DIST_NAME.egg-info" ]]; then
  cp -R "$DIST_NAME.egg-info" "$BACKUP_DIR/egg-info"
fi

cleanup() {
  cp "$BACKUP_DIR/__init__.py" "$PACKAGE_ROOT/__init__.py"
  cp "$BACKUP_DIR/pyproject.toml" pyproject.toml
  cp "$BACKUP_DIR/pyproject-next.toml" pyproject-next.toml
  rm -rf "$DIST_NAME.egg-info" "$PACKAGE_ROOT/bin" "$PACKAGE_ROOT/share"
  if [[ -d "$BACKUP_DIR/egg-info" ]]; then
    cp -R "$BACKUP_DIR/egg-info" "$DIST_NAME.egg-info"
  fi
  rm -rf "$ROOT_DIR/build"
  rm -rf "$BACKUP_DIR"
}
trap cleanup EXIT

BIN="${AFM_RELEASE_BIN:-.build/arm64-apple-macosx/release/afm}"
if [[ -z "${AFM_RELEASE_BIN:-}" && ! -x "$BIN" ]]; then
  BIN=".build/release/afm"
fi
[[ -x "$BIN" ]] || {
  echo "[wheel] No release binary found. Run the release build first." >&2
  exit 1
}

METALLIB="$($ROOT_DIR/Scripts/resolve-afmkit-resource.sh --metallib "$(dirname "$BIN")")"
"$ROOT_DIR/Scripts/check-macos26-compatibility.sh" "$BIN" "$METALLIB"

sed -i '' "s/^__version__ = .*/__version__ = \"${VERSION}\"/" "$PACKAGE_ROOT/__init__.py"
if [[ "$CHANNEL" == "nightly" ]]; then
  sed -i '' "s/^version = .*/version = \"${VERSION}\"/" pyproject-next.toml
  cp pyproject-next.toml pyproject.toml
else
  sed -i '' "s/^version = .*/version = \"${VERSION}\"/" pyproject.toml
fi

mkdir -p "$PACKAGE_ROOT/bin" "$PACKAGE_ROOT/share/webui"
cp "$BIN" "$PACKAGE_ROOT/bin/"
for bundle in MacLocalAPI_AFMEvaluationHost.bundle AFMKit_AFMKitMLX.bundle AFMKit_AFMKitDwarfStar.bundle; do
  source_bundle="$(dirname "$BIN")/$bundle"
  [[ -d "$source_bundle" ]] || {
    echo "[wheel] Required runtime bundle missing: $source_bundle" >&2
    exit 1
  }
  cp -R "$source_bundle" "$PACKAGE_ROOT/bin/"
done
cp "$METALLIB" "$PACKAGE_ROOT/bin/default.metallib"
"$ROOT_DIR/Scripts/verify-webui.sh" Resources/webui/index.html.gz
cp Resources/webui/index.html.gz "$PACKAGE_ROOT/share/webui/"

rm -rf "$ROOT_DIR/build"
rm -f "dist/${DIST_NAME}-"*.whl
echo "[wheel] Building $CHANNEL wheel $DIST_NAME $VERSION"
uv build --wheel

WHEEL="$(ls -t "dist/${DIST_NAME}-"*.whl 2>/dev/null | head -1)"
[[ -n "$WHEEL" ]] || { echo "[wheel] No wheel was produced" >&2; exit 1; }
"$ROOT_DIR/Scripts/verify-native-wheel.sh" "$WHEEL" "$PACKAGE_ROOT"

wheel_webui="$ROOT_DIR/.build/${DIST_NAME}-wheel-webui.html.gz"
unzip -p "$WHEEL" "$PACKAGE_ROOT/share/webui/index.html.gz" > "$wheel_webui"
"$ROOT_DIR/Scripts/verify-webui.sh" "$wheel_webui"
rm -f "$wheel_webui"

echo "[wheel] Ready: $WHEEL"
