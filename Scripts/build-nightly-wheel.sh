#!/usr/bin/env bash
#
# Build a nightly wheel for macafm-next from an existing compiled afm binary.
#
# Usage:
#   ./Scripts/build-nightly-wheel.sh [--version BASE_VERSION]
#       [--build-version CANONICAL_VERSION] [--python-version PEP440_VERSION]
#
# The wheel is written to dist/macafm_next-VERSION-py3-none-macosx_26_0_arm64.whl
# Python metadata uses a PEP 440 dev version while the bundled command reports
# the canonical Homebrew/nightly display version.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Nightly wheel metadata is temporary build input, not a source change. Keep
# exact copies outside the checkout and restore them on every exit path so a
# failed build cannot leave the release branch dirty.
METADATA_BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/afm-nightly-wheel.XXXXXX")"
cp pyproject.toml "$METADATA_BACKUP_DIR/pyproject.toml"
cp pyproject-next.toml "$METADATA_BACKUP_DIR/pyproject-next.toml"
cp macafm_next/__init__.py "$METADATA_BACKUP_DIR/macafm-next-init.py"
mkdir "$METADATA_BACKUP_DIR/egg-info"
cp macafm_next.egg-info/* "$METADATA_BACKUP_DIR/egg-info/"

cleanup() {
    cp "$METADATA_BACKUP_DIR/pyproject.toml" pyproject.toml
    cp "$METADATA_BACKUP_DIR/pyproject-next.toml" pyproject-next.toml
    cp "$METADATA_BACKUP_DIR/macafm-next-init.py" macafm_next/__init__.py
    cp "$METADATA_BACKUP_DIR/egg-info/"* macafm_next.egg-info/
    rm -rf "$REPO_ROOT/macafm_next/bin" "$REPO_ROOT/macafm_next/share"
    rm -rf "$METADATA_BACKUP_DIR"
}
trap cleanup EXIT

# ---------- parse args ----------
BASE_VERSION=""
BUILD_VERSION=""
PYTHON_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) BASE_VERSION="$2"; shift 2 ;;
        --build-version) BUILD_VERSION="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- determine version ----------
if [ -z "$BASE_VERSION" ]; then
    BASE_VERSION=$(grep 'static let version' Sources/AFMKit/BuildInfo.swift \
        | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
fi
if [ -z "$BUILD_VERSION" ]; then
    BUILD_VERSION=$("$REPO_ROOT/Scripts/nightly-version.sh" \
        --base-version "$BASE_VERSION" --field canonical)
fi
if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$("$REPO_ROOT/Scripts/nightly-version.sh" \
        --base-version "$BASE_VERSION" --field python)
fi
echo "[INFO] Nightly display version: ${BUILD_VERSION}"
echo "[INFO] Nightly Python version: ${PYTHON_VERSION}"

mkdir -p "$REPO_ROOT/.build"
METADATA_BACKUP=$(mktemp -d "$REPO_ROOT/.build/afm-next-wheel-metadata.XXXXXX")
cp macafm_next/__init__.py "$METADATA_BACKUP/__init__.py"
cp pyproject-next.toml "$METADATA_BACKUP/pyproject-next.toml"
cp pyproject.toml "$METADATA_BACKUP/pyproject.toml"
if [ -f setup.py ]; then
    cp setup.py "$METADATA_BACKUP/setup.py"
fi
if [ -d macafm_next.egg-info ]; then
    cp -R macafm_next.egg-info "$METADATA_BACKUP/macafm_next.egg-info"
fi

cleanup() {
    cp "$METADATA_BACKUP/__init__.py" macafm_next/__init__.py 2>/dev/null || true
    cp "$METADATA_BACKUP/pyproject-next.toml" pyproject-next.toml 2>/dev/null || true
    cp "$METADATA_BACKUP/pyproject.toml" pyproject.toml 2>/dev/null || true
    if [ -f "$METADATA_BACKUP/setup.py" ]; then
        cp "$METADATA_BACKUP/setup.py" setup.py 2>/dev/null || true
    else
        rm -f setup.py
    fi
    rm -rf macafm_next.egg-info
    if [ -d "$METADATA_BACKUP/macafm_next.egg-info" ]; then
        cp -R "$METADATA_BACKUP/macafm_next.egg-info" macafm_next.egg-info
    fi
    rm -rf macafm_next/bin macafm_next/share "$WHEEL_SMOKE" "$METADATA_BACKUP"
}
WHEEL_SMOKE=""
trap cleanup EXIT

# ---------- locate binary ----------
BIN=".build/arm64-apple-macosx/release/afm"
[ -x "$BIN" ] || BIN=".build/release/afm"
if [ ! -x "$BIN" ]; then
    echo "[ERROR] No compiled binary found. Run ./Scripts/build-from-scratch.sh first."
    exit 1
fi
echo "[INFO] Binary: $(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

METALLIB="$(dirname "$BIN")/MacLocalAPI_AFMKitMLX.bundle/default.metallib"
if [ ! -f "$METALLIB" ]; then
    METALLIB="$(dirname "$BIN")/MacLocalAPI_AFMKitMLX.bundle/Contents/Resources/default.metallib"
fi
"$REPO_ROOT/Scripts/check-macos26-compatibility.sh" "$BIN" "$METALLIB"

# ---------- set version in package files ----------
sed -i '' "s/^__version__ = .*/__version__ = \"${PYTHON_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^__build_version__ = .*/__build_version__ = \"${BUILD_VERSION}\"/" macafm_next/__init__.py
sed -i '' "s/^version = .*/version = \"${PYTHON_VERSION}\"/" pyproject-next.toml

# ---------- stage assets ----------
echo "[INFO] Staging assets into macafm_next/"
mkdir -p macafm_next/bin
cp "$BIN" macafm_next/bin/
for BUNDLE_NAME in MacLocalAPI_AFMKit.bundle MacLocalAPI_AFMKitMLX.bundle MacLocalAPI_AFMKitDwarfStar.bundle; do
    BUNDLE_DIR="$(dirname "$BIN")/$BUNDLE_NAME"
    if [ ! -d "$BUNDLE_DIR" ]; then
        echo "[ERROR] Required runtime bundle missing: $BUNDLE_DIR"
        exit 1
    fi
    cp -R "$BUNDLE_DIR" macafm_next/bin/
    echo "[INFO] Included $BUNDLE_NAME"
done
if [ -f "$METALLIB" ]; then
    cp "$METALLIB" macafm_next/bin/
    echo "[INFO] Included metallib"
fi
"$REPO_ROOT/Scripts/verify-webui.sh" Resources/webui/index.html.gz
mkdir -p macafm_next/share/webui
cp Resources/webui/index.html.gz macafm_next/share/webui/
echo "[INFO] Included webui"

# ---------- build wheel ----------
# Use pyproject-next.toml by temporarily swapping it in. The EXIT trap restores
# the original even when uv or a later verification command fails.
cp pyproject-next.toml pyproject.toml

# The package embeds a macOS arm64 executable and Metal resources. Setuptools
# otherwise classifies the declarative Python wrapper as a pure, cross-platform
# wheel (`py3-none-any`), allowing pip to install an unusable payload elsewhere.
cat > setup.py <<'PY'
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class AFMBinaryWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": AFMBinaryWheel})
PY

echo "[INFO] Building wheel..."
rm -rf dist/macafm_next-*
_PYTHON_HOST_PLATFORM=macosx-26.0-arm64 uv build --wheel 2>&1

# ---------- clean staged assets ----------
rm -rf macafm_next/bin macafm_next/share
echo "[INFO] Cleaned staged assets"

# ---------- verify ----------
WHL=$(ls dist/macafm_next-*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "[ERROR] No wheel found in dist/"
    exit 1
fi
case "$(basename "$WHL")" in
    *-py3-none-macosx_26_0_arm64.whl) ;;
    *)
        echo "[ERROR] Wheel is not restricted to macOS 26 arm64: $WHL"
        exit 1
        ;;
esac
WHL_SIZE=$(du -m "$WHL" | cut -f1)
echo "[INFO] Wheel: $WHL (${WHL_SIZE}MB)"
if [ "$WHL_SIZE" -lt 1 ]; then
    echo "[ERROR] Wheel is too small — assets were not staged correctly"
    exit 1
fi

WHEEL_WEBUI="$REPO_ROOT/.build/afm-next-wheel-webui.html.gz"
WHEEL_WEBUI_ENTRY=$(unzip -Z1 "$WHL" \
    | awk '/(^|\/)macafm_next\/share\/webui\/index.html.gz$/ { print; exit }')
if [ -z "$WHEEL_WEBUI_ENTRY" ]; then
    echo "[ERROR] Wheel does not contain the required WebUI"
    exit 1
fi
unzip -p "$WHL" "$WHEEL_WEBUI_ENTRY" > "$WHEEL_WEBUI"
"$REPO_ROOT/Scripts/verify-webui.sh" "$WHEEL_WEBUI"
rm -f "$WHEEL_WEBUI"

WHEEL_SMOKE=$(mktemp -d "$REPO_ROOT/.build/afm-next-wheel-smoke.XXXXXX")
python3 -m pip install --quiet --no-deps --no-compile \
    --target "$WHEEL_SMOKE/site" "$WHL"
WHEEL_METADATA=$(find "$WHEEL_SMOKE/site" -path '*.dist-info/METADATA' -print -quit)
ACTUAL_PYTHON_VERSION=$(awk '/^Version: / { print $2; exit }' "$WHEEL_METADATA")
if [ "$ACTUAL_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
    echo "[ERROR] Wheel metadata reports '$ACTUAL_PYTHON_VERSION'; expected '$PYTHON_VERSION'"
    exit 1
fi
WHEEL_DESCRIPTOR=$(find "$WHEEL_SMOKE/site" -path '*.dist-info/WHEEL' -print -quit)
if ! grep -Fxq 'Root-Is-Purelib: false' "$WHEEL_DESCRIPTOR"; then
    echo "[ERROR] Wheel metadata still marks the native payload as pure"
    exit 1
fi
ACTUAL_VERSION=$(cd "$WHEEL_SMOKE/site" && \
    AFM_BUILD_VERSION="v-invalid-host-override" PYTHONPATH="$WHEEL_SMOKE/site" \
    python3 -c 'from macafm_next.cli import main; main()' --version)
EXPECTED_VERSION="v${BUILD_VERSION#v}"
if [ "$ACTUAL_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "[ERROR] Wheel reports '$ACTUAL_VERSION'; expected '$EXPECTED_VERSION'"
    exit 1
fi
rm -rf "$WHEEL_SMOKE"
echo "[INFO] Verified wheel metadata version: $ACTUAL_PYTHON_VERSION"
echo "[INFO] Verified wheel runtime version: $ACTUAL_VERSION"
if ! unzip -Z1 "$WHL" | grep -E \
    '^macafm_next/bin/MacLocalAPI_AFMKit\.bundle/(Evals/|Contents/Resources/Evals/)comprehensive\.json$' \
    >/dev/null; then
    echo "[ERROR] Wheel is missing the bundled comprehensive evaluation suite"
    exit 1
fi

echo "[INFO] Done. Wheel ready: $WHL"
