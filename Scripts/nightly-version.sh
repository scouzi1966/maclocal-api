#!/usr/bin/env bash
# Resolve all identifiers for one AFM nightly from a single input tuple.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_VERSION=""
BUILD_DATE="$(date -u +%Y%m%d)"
SHORT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
FIELD="canonical"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-version) BASE_VERSION="$2"; shift 2 ;;
        --date) BUILD_DATE="$2"; shift 2 ;;
        --sha) SHORT_SHA="$2"; shift 2 ;;
        --field) FIELD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$BASE_VERSION" ]]; then
    BASE_VERSION=$(grep 'static let version' "$ROOT_DIR/Sources/AFMKit/BuildInfo.swift" \
        | sed 's/.*"\(.*\)".*/\1/' | sed 's/^v//')
fi
BASE_VERSION="${BASE_VERSION#v}"

if [[ ! "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Invalid base version: $BASE_VERSION" >&2
    exit 1
fi
if [[ ! "$BUILD_DATE" =~ ^[0-9]{8}$ ]]; then
    echo "Invalid nightly date: $BUILD_DATE" >&2
    exit 1
fi
if [[ ! "$SHORT_SHA" =~ ^[0-9a-fA-F]+$ ]]; then
    echo "Invalid commit identifier: $SHORT_SHA" >&2
    exit 1
fi
SHORT_SHA=$(printf '%s' "$SHORT_SHA" | tr '[:upper:]' '[:lower:]')

CANONICAL_VERSION="${BASE_VERSION}-next.${BUILD_DATE}.${SHORT_SHA}"
PYTHON_VERSION="${BASE_VERSION}.dev${BUILD_DATE}+${SHORT_SHA}"
RELEASE_TAG="nightly-${BUILD_DATE}-${SHORT_SHA}"

case "$FIELD" in
    base) printf '%s\n' "$BASE_VERSION" ;;
    canonical) printf '%s\n' "$CANONICAL_VERSION" ;;
    python) printf '%s\n' "$PYTHON_VERSION" ;;
    tag) printf '%s\n' "$RELEASE_TAG" ;;
    env)
        printf 'AFM_NIGHTLY_BASE_VERSION=%s\n' "$BASE_VERSION"
        printf 'AFM_NIGHTLY_VERSION=%s\n' "$CANONICAL_VERSION"
        printf 'AFM_NIGHTLY_PYTHON_VERSION=%s\n' "$PYTHON_VERSION"
        printf 'AFM_NIGHTLY_TAG=%s\n' "$RELEASE_TAG"
        ;;
    *) echo "Unknown field: $FIELD" >&2; exit 1 ;;
esac
