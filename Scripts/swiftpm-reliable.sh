#!/usr/bin/env bash
# Run SwiftPM with a targeted recovery for Xcode 27's explicit-module scanner.
#
# The swiftbuild driver can leave generated C modules unresolved (commonly
# CAsyncHTTPClient, CSystem, CNIO*, and _NumericsShims). Xcode 27 Beta 3 is a
# known-bad toolchain, so select the native driver up front there. Other Xcode
# versions start normally and use the native driver only after the exact scanner
# failure signature. Checkouts and source patches are always preserved.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the standalone AFMKit package on the exact same patched language-model
# checkout as the maclocal-api host during development. Published AFMKit builds
# continue to use its tagged dependency when this override is absent.
export AFMKIT_MLX_SWIFT_LM_PATH="${AFMKIT_MLX_SWIFT_LM_PATH:-$ROOT_DIR/vendor/mlx-swift-lm}"

# All native MLX patch workflows target the AFM fork checkout. The former
# official `mlx-swift` identity can otherwise survive in Package.resolved and
# produce duplicate Cmlx/MLX modules alongside `mlx-swift-afm`.
export MLX_SWIFT_CHECKOUT="${MLX_SWIFT_CHECKOUT:-$ROOT_DIR/.build/checkouts/mlx-swift-afm}"
SUBCOMMAND="${1:-}"
if [[ "$SUBCOMMAND" != "build" && "$SUBCOMMAND" != "test" ]]; then
    echo "Usage: $0 <build|test> [swiftpm options...]" >&2
    exit 2
fi
shift

LOG_DIR="$ROOT_DIR/.build-reliable-logs"
STATE_DIR="$ROOT_DIR/.build-reliable-state"
mkdir -p "$LOG_DIR" "$STATE_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
PRIMARY_LOG="$LOG_DIR/${SUBCOMMAND}-${STAMP}.log"
RETRY_LOG="$LOG_DIR/${SUBCOMMAND}-${STAMP}-native-retry.log"

# XCTest's executable can be hosted outside the package build directory, so
# AFMKitMLX cannot reliably infer the SwiftPM resource-bundle location from
# `_NSGetExecutablePath`. Tests always use the canonical committed metallib;
# an explicit caller override still wins. Export this before SwiftPM launches
# the test process so it also survives clean/retry builds.
if [[ "$SUBCOMMAND" == "test" && -z "${MACAFM_MLX_METALLIB:-}" ]]; then
    CANONICAL_METALLIB="$ROOT_DIR/Sources/AFMKitMLX/Resources/default.metallib"
    if [[ ! -f "$CANONICAL_METALLIB" ]]; then
        echo "[swiftpm-reliable] Missing canonical MLX metallib: $CANONICAL_METALLIB" >&2
        exit 1
    fi
    export MACAFM_MLX_METALLIB="$CANONICAL_METALLIB"
    echo "[swiftpm-reliable] XCTest MLX metallib: $MACAFM_MLX_METALLIB" >&2
fi

test_configuration() {
    local previous=""
    local argument
    for argument in "$@"; do
        if [[ "$previous" == "-c" || "$previous" == "--configuration" ]]; then
            printf '%s\n' "$argument"
            return
        fi
        case "$argument" in
            --configuration=*)
                printf '%s\n' "${argument#*=}"
                return
                ;;
        esac
        previous="$argument"
    done
    printf '%s\n' "debug"
}

test_scratch_path() {
    local previous=""
    local argument
    for argument in "$@"; do
        if [[ "$previous" == "--scratch-path" ]]; then
            printf '%s\n' "$argument"
            return
        fi
        case "$argument" in
            --scratch-path=*)
                printf '%s\n' "${argument#*=}"
                return
                ;;
        esac
        previous="$argument"
    done
    printf '%s\n' "$ROOT_DIR/.build"
}

stage_xctest_metallib() {
    [[ "$SUBCOMMAND" == "test" ]] || return 0

    local source="${MACAFM_MLX_METALLIB:-}"
    [[ -f "$source" ]] || {
        echo "[swiftpm-reliable] XCTest MLX metallib is missing: $source" >&2
        return 1
    }

    local configuration
    configuration="$(test_configuration "$@")"
    local scratch_path
    scratch_path="$(test_scratch_path "$@")"
    if [[ "$scratch_path" != /* ]]; then
        scratch_path="$ROOT_DIR/$scratch_path"
    fi

    # MLX's C++ runtime searches for mlx.metallib beside the loaded test
    # executable. SwiftPM 6.3 does not always emit mlx-swift_Cmlx.bundle for
    # XCTest, so create the colocated resource before linking/running tests.
    # The linker preserves sibling resources in the .xctest bundle.
    local architecture
    architecture="$(uname -m)"
    local predicted_dir="$scratch_path/${architecture}-apple-macosx/$configuration/MacLocalAPIPackageTests.xctest/Contents/MacOS"
    mkdir -p "$predicted_dir"
    cp "$source" "$predicted_dir/mlx.metallib"

    local executable_dir
    while IFS= read -r executable_dir; do
        cp "$source" "$executable_dir/mlx.metallib"
    done < <(find "$scratch_path" -type d -path '*.xctest/Contents/MacOS' -print 2>/dev/null)

    echo "[swiftpm-reliable] Staged MLX metallib for XCTest: $predicted_dir/mlx.metallib" >&2
}

has_scanner_failure() {
    local log_file="$1"
    grep -Eq \
        "clang dependency scanning failure|unable to resolve module dependency:|missing required module '(_NumericsShims|CAsyncHTTPClient|CSystem|CNIO[^']*)'" \
        "$log_file"
}

has_generated_dependency_failure() {
    local log_file="$1"
    grep -Eq \
        "unable to open dependencies file .*\.d\)|SwiftDriver\\\\ Compilation\\\\ Requirements .* failed with a nonzero exit code" \
        "$log_file"
}

has_recoverable_xcode_failure() {
    local log_file="$1"
    has_scanner_failure "$log_file" \
        || has_generated_dependency_failure "$log_file" \
        || grep -q "was not compiled for testing" "$log_file"
}

run_native() {
    local log_file="$1"
    shift
    stage_xctest_metallib "$@" || return $?
    set +e
    swift "$SUBCOMMAND" --build-system native "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

cd "$ROOT_DIR"

# DwarfStar is a vanilla dependency. AFM integration belongs in CDwarfStar and
# AFMKitDwarfStar; fail instead of compiling locally modified engine sources.
if [[ -n "$(git -C "$ROOT_DIR/vendor/ds4" status --porcelain --untracked-files=all)" ]]; then
    echo "[swiftpm-reliable] vendor/ds4 is not a clean upstream checkout." >&2
    echo "[swiftpm-reliable] Keep DwarfStar vanilla and move interface work into AFM-owned adapters." >&2
    exit 1
fi

EXPECTED_DS4_REVISION="$(git -C "$ROOT_DIR" ls-files -s vendor/ds4 | awk '{print $2}')"
ACTUAL_DS4_REVISION="$(git -C "$ROOT_DIR/vendor/ds4" rev-parse HEAD)"
if [[ -z "$EXPECTED_DS4_REVISION" || "$ACTUAL_DS4_REVISION" != "$EXPECTED_DS4_REVISION" ]]; then
    echo "[swiftpm-reliable] vendor/ds4 revision does not match the repository gitlink." >&2
    echo "[swiftpm-reliable] expected=${EXPECTED_DS4_REVISION:-missing} actual=$ACTUAL_DS4_REVISION" >&2
    echo "[swiftpm-reliable] Run: git submodule update --init vendor/ds4" >&2
    exit 1
fi

run_required_patch_step() {
    local label="$1"
    shift
    if ! "$@"; then
        echo "[swiftpm-reliable] Required patch step failed: $label" >&2
        echo "[swiftpm-reliable] Refusing to compile a partially patched dependency." >&2
        exit 1
    fi
}

# The official DeepSeek V4 checkpoint uses F8_E8M0 safetensor metadata for
# byte-packed block scales. The pinned mlx-swift release predates that tag.
# Resolve once on a fresh clone, then apply the idempotent source patch before
# SwiftPM decides whether dependency products are current.
MLX_SAFETENSORS_SOURCE="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx/mlx/io/safetensors.cpp"
if [[ ! -f "$MLX_SAFETENSORS_SOURCE" ]]; then
    echo "[swiftpm-reliable] Resolving mlx-swift before applying official FP8 loader support." >&2
    swift package resolve
fi
run_required_patch_step \
    "DeepSeek V4 kernels" \
    "$ROOT_DIR/Scripts/apply-mlx-deepseek-v4-kernels.sh"
run_required_patch_step \
    "DeepSeek V4 kernel verification" \
    "$ROOT_DIR/Scripts/apply-mlx-deepseek-v4-kernels.sh" --check
run_required_patch_step \
    "official FP8 loader verification" \
    "$ROOT_DIR/Scripts/apply-mlx-official-fp8-loader.sh"
# Keep the persistent vendor checkout synchronized with the authoritative
# Swift sources under Scripts/patches before SwiftPM evaluates dependencies.
# Without this, a Release build can succeed while compiling stale vendored
# model code and silently omit the optimization being benchmarked.
run_required_patch_step \
    "vendored MLX Swift sources" \
    "$ROOT_DIR/Scripts/apply-mlx-patches.sh"
run_required_patch_step \
    "vendored MLX Swift source verification" \
    "$ROOT_DIR/Scripts/apply-mlx-patches.sh" --check

# Xcode 27's native driver can also miss changes in the C sources included by
# CDwarfStar. The dependency is consumed unchanged, so its pinned revision is
# the complete native-source identity.
DS4_SOURCE_STAMP="$STATE_DIR/ds4-source.sha256"
DS4_SOURCE_FINGERPRINT="$({
    printf '%s\n' "$ACTUAL_DS4_REVISION"
} | shasum -a 256 | awk '{print $1}')"
PREVIOUS_DS4_SOURCE_FINGERPRINT="$(cat "$DS4_SOURCE_STAMP" 2>/dev/null || true)"
if [[ "$DS4_SOURCE_FINGERPRINT" != "$PREVIOUS_DS4_SOURCE_FINGERPRINT" ]]; then
    echo "[swiftpm-reliable] DwarfStar source changed; invalidating stale native-driver products." >&2
    rm -rf \
        "$ROOT_DIR/.build/arm64-apple-macosx" \
        "$ROOT_DIR/.build/debug" \
        "$ROOT_DIR/.build/release"
    printf '%s\n' "$DS4_SOURCE_FINGERPRINT" > "$DS4_SOURCE_STAMP"
fi

# Xcode 27 Beta 3's native SwiftPM driver can miss source changes inside the
# local mlx-swift-lm package and report a successful no-op build. Fingerprint
# that package independently of SwiftPM and discard only compiled products when
# it changes. Dependency clones and downloaded artifacts remain intact.
MLX_SOURCE_STAMP="$STATE_DIR/mlx-swift-lm-source.sha256"
MLX_SOURCE_FINGERPRINT="$({
    find "$ROOT_DIR/vendor/mlx-swift-lm/Libraries" -type f -print0
    printf '%s\0' "$ROOT_DIR/vendor/mlx-swift-lm/Package.swift"
    printf '%s\0' "$MLX_SAFETENSORS_SOURCE"
    find "$ROOT_DIR/Scripts/patches" -type f -print0
    printf '%s\0' "$ROOT_DIR/Scripts/apply-mlx-patches.sh"
    find "$ROOT_DIR/Scripts/patches/mlx-swift-deepseek-v4" -type f -print0
} | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | awk '{print $1}')"
PREVIOUS_MLX_SOURCE_FINGERPRINT="$(cat "$MLX_SOURCE_STAMP" 2>/dev/null || true)"
if [[ "$MLX_SOURCE_FINGERPRINT" != "$PREVIOUS_MLX_SOURCE_FINGERPRINT" ]]; then
    echo "[swiftpm-reliable] MLX source changed; invalidating stale native-driver products." >&2
    rm -rf \
        "$ROOT_DIR/.build/arm64-apple-macosx" \
        "$ROOT_DIR/.build/debug" \
        "$ROOT_DIR/.build/release"
    printf '%s\n' "$MLX_SOURCE_FINGERPRINT" > "$MLX_SOURCE_STAMP"
fi

# A normal Release build emits modules without `-enable-testing`. Xcode 27's
# native SwiftPM driver may then incorrectly reuse those modules for a Release
# XCTest invocation, causing `@testable import` to fail. Track build-to-test
# transitions and invalidate only compiled products; dependency checkouts and
# downloaded artifacts remain intact.
CONFIGURATION="$(test_configuration "$@")"
SCRATCH_PATH="$(test_scratch_path "$@")"
if [[ "$SCRATCH_PATH" != /* ]]; then
    SCRATCH_PATH="$ROOT_DIR/$SCRATCH_PATH"
fi
OPERATION_STAMP="$STATE_DIR/last-operation-${CONFIGURATION}"
PREVIOUS_OPERATION="$(cat "$OPERATION_STAMP" 2>/dev/null || true)"
if [[ "$SUBCOMMAND" == "test" && "$PREVIOUS_OPERATION" == "build" ]]; then
    echo "[swiftpm-reliable] Release build preceded tests; invalidating non-testable products." >&2
    rm -rf \
        "$SCRATCH_PATH/$(uname -m)-apple-macosx/$CONFIGURATION" \
        "$SCRATCH_PATH/$CONFIGURATION"
fi
printf '%s\n' "$SUBCOMMAND" > "$OPERATION_STAMP"

DRIVER="${AFM_SWIFTPM_DRIVER:-auto}"
DEVELOPER_DIR="$(xcode-select -p 2>/dev/null || true)"
if [[ "$DRIVER" == "native" ]] ||
   [[ "$DRIVER" == "auto" && "$DEVELOPER_DIR" == *"Xcode-27.0.0-Beta.3.app/Contents/Developer" ]]; then
    # Keep the driver stamp outside .build: `swift package clean` removes that
    # directory as part of scanner recovery, but must not invalidate the driver
    # identity and force another clean on the next invocation.
    DRIVER_STAMP="$STATE_DIR/native-driver-xcode27-beta3"
    DRIVER_ID="$DEVELOPER_DIR|$(xcodebuild -version 2>/dev/null | tr '\n' ' ')"
    CURRENT_ID="$(cat "$DRIVER_STAMP" 2>/dev/null || true)"
    if [[ "$CURRENT_ID" != "$DRIVER_ID" ]]; then
        if [[ -n "$CURRENT_ID" || -d "$ROOT_DIR/.build/out" ]]; then
            echo "[swiftpm-reliable] Isolating native-driver products from swiftbuild products." >&2
            # Preserve dependency clones and downloaded artifacts. Only products,
            # generated modules, and plugin intermediates are driver-specific.
            rm -rf \
                "$ROOT_DIR/.build/out" \
                "$ROOT_DIR/.build/arm64-apple-macosx" \
                "$ROOT_DIR/.build/debug" \
                "$ROOT_DIR/.build/release" \
                "$ROOT_DIR/.build/plugins"
        else
            echo "[swiftpm-reliable] Recording native driver for the existing clean/native build tree." >&2
        fi
        mkdir -p "$ROOT_DIR/.build"
        printf '%s\n' "$DRIVER_ID" > "$DRIVER_STAMP"
    fi
    echo "[swiftpm-reliable] Using native driver for Xcode 27 Beta 3." >&2
    if run_native "$PRIMARY_LOG" "$@"; then
        exit 0
    else
        STATUS=$?
    fi
    if has_recoverable_xcode_failure "$PRIMARY_LOG"; then
        echo "[swiftpm-reliable] Native generated build state is invalid; cleaning products and retrying once." >&2
        swift package clean
        if run_native "$RETRY_LOG" "$@"; then
            exit 0
        else
            STATUS=$?
        fi
        echo "[swiftpm-reliable] Clean native-driver retry failed. Log: $RETRY_LOG" >&2
    fi
    exit "$STATUS"
elif [[ "$DRIVER" != "auto" && "$DRIVER" != "swiftbuild" ]]; then
    echo "AFM_SWIFTPM_DRIVER must be auto, native, or swiftbuild" >&2
    exit 2
fi

if [[ "$DRIVER" == "swiftbuild" ]]; then
    rm -f "$STATE_DIR/native-driver-xcode27-beta3"
fi

set +e
stage_xctest_metallib "$@"
STAGE_STATUS=$?
if [[ $STAGE_STATUS -ne 0 ]]; then
    exit "$STAGE_STATUS"
fi
swift "$SUBCOMMAND" "$@" 2>&1 | tee "$PRIMARY_LOG"
STATUS=${PIPESTATUS[0]}
set -e

if [[ $STATUS -eq 0 ]]; then
    exit 0
fi

if ! has_recoverable_xcode_failure "$PRIMARY_LOG"; then
    exit "$STATUS"
fi

echo "[swiftpm-reliable] Recoverable Xcode generated-build-state failure detected." >&2
echo "[swiftpm-reliable] Cleaning build products and retrying with the native driver." >&2
swift package clean

if run_native "$RETRY_LOG" "$@"; then
    STATUS=0
else
    STATUS=$?
fi

if [[ $STATUS -ne 0 ]]; then
    echo "[swiftpm-reliable] Native-driver retry failed. Log: $RETRY_LOG" >&2
fi
exit "$STATUS"
