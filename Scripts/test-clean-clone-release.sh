#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${AFM_CLEAN_CLONE_ROOT:-$ROOT_DIR/.build/clean-clone-release}"
CLONE_DIR="$WORK_ROOT/repository"
ARCHIVE="$WORK_ROOT/clean-clone-release.tar.gz"
HEAD_REVISION="$(git -C "$ROOT_DIR" rev-parse HEAD)"

rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

echo "[clean-clone] Cloning committed revision $HEAD_REVISION"
git clone --no-local --no-hardlinks "$ROOT_DIR" "$CLONE_DIR"
git -C "$CLONE_DIR" checkout --detach "$HEAD_REVISION"

[[ ! -e "$CLONE_DIR/.build" ]] || {
  echo "[clean-clone] clone unexpectedly contains build state" >&2
  exit 1
}
git -C "$CLONE_DIR" ls-files --error-unmatch Package.resolved >/dev/null

# This intentionally runs before package resolution to guard the original
# clean-clone failure mode.
"$CLONE_DIR/Scripts/check-afmkit-consumer-boundary.sh"
LOCK_HASH_BEFORE="$(shasum -a 256 "$CLONE_DIR/Package.resolved" | awk '{print $1}')"

(
  cd "$CLONE_DIR"
  export AFM_RELEASE_MODE=1
  ./build.sh --stable --yes
  Scripts/create-tarball.sh --output "$ARCHIVE"
)

LOCK_HASH_AFTER="$(shasum -a 256 "$CLONE_DIR/Package.resolved" | awk '{print $1}')"
[[ "$LOCK_HASH_BEFORE" == "$LOCK_HASH_AFTER" ]] || {
  echo "[clean-clone] source build changed Package.resolved" >&2
  exit 1
}

"$CLONE_DIR/Scripts/verify-release-archive.sh" "$ARCHIVE"
echo "[clean-clone] source build and packaged launch passed from an isolated clone"
