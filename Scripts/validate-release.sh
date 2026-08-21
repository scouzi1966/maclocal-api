#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${AFM_RELEASE_VALIDATION_ROOT:-$ROOT_DIR/.build/release-validation}"
ARCHIVE="$WORK_ROOT/afm-release-validation.tar.gz"
PREFIX="$WORK_ROOT/install-prefix"

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"
cd "$ROOT_DIR"
export AFM_RELEASE_MODE=1

echo "[release-gate] Validating release tooling and authenticated dependency graph"
Scripts/tests/test-release-tooling.sh
Scripts/resolve-release-dependencies.sh

echo "[release-gate] Running the complete Release test suite"
Scripts/swiftpm-reliable.sh test -c release \
  -Xswiftc -disable-upcoming-feature \
  -Xswiftc MemberImportVisibility

echo "[release-gate] Building source release without local dependency overrides"
if ! Scripts/verify-webui.sh Resources/webui/index.html.gz >/dev/null 2>&1; then
  echo "[release-gate] Building the WebUI from its locked npm dependency graph"
  make webui
fi
./build.sh --stable --yes --skip-submodules --skip-webui
Scripts/create-tarball.sh --output "$ARCHIVE"

echo "[release-gate] Verifying custom-prefix install and uninstall"
INSTALL_PREFIX="$PREFIX" \
  ./build.sh --stable --yes --no-clean --skip-submodules --skip-webui --install
"$PREFIX/bin/afm" --version >/dev/null
Scripts/resolve-afmkit-resource.sh --metallib "$PREFIX/bin" >/dev/null
printf 'unrelated\n' > "$PREFIX/bin/release-gate-unrelated"
INSTALL_PREFIX="$PREFIX" Scripts/uninstall.sh
[[ ! -e "$PREFIX/bin/afm" ]] || {
  echo "[release-gate] custom-prefix uninstall left the binary behind" >&2
  exit 1
}
[[ -f "$PREFIX/bin/release-gate-unrelated" ]] || {
  echo "[release-gate] custom-prefix uninstall removed an unrelated file" >&2
  exit 1
}

echo "[release-gate] Verifying the Xcode 27 nested resource wheel layout"
Scripts/tests/test-xcode27-wheel-layout.sh

echo "[release-gate] Building, installing, and launching the stable wheel"
Scripts/build-stable-wheel.sh

echo "[release-gate] Repeating source packaging from a true clean clone"
Scripts/test-clean-clone-release.sh

echo "[release-gate] Complete Release, source, archive, install, and wheel validation passed"
