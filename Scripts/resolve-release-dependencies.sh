#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=false
REFRESH_LOCK=false

usage() {
  cat <<'USAGE'
Usage: Scripts/resolve-release-dependencies.sh [--check-access] [--refresh-lock]

Resolve exactly the revisions in the tracked Package.resolved. AFMKit is the
single provider package; the caller must have read access while it is private.

  --refresh-lock  Intentionally resolve the exact manifest requirements and
                  rewrite Package.resolved after a reviewed dependency change.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-access) CHECK_ONLY=true ;;
    --refresh-lock) REFRESH_LOCK=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT_DIR"
if ! $REFRESH_LOCK; then
  "$ROOT_DIR/Scripts/check-afmkit-consumer-boundary.sh"
fi

if [[ -n "${MACLOCAL_AFMKIT_PATH:-}" ]]; then
  if [[ "${AFM_RELEASE_MODE:-0}" == "1" ]]; then
    echo "[afmkit-auth] Release validation forbids MACLOCAL_AFMKIT_PATH." >&2
    echo "[afmkit-auth] Resolve the authenticated immutable AFMKit revision instead." >&2
    exit 1
  fi
  [[ -f "$MACLOCAL_AFMKIT_PATH/Package.swift" ]] || {
    echo "[afmkit-auth] Invalid MACLOCAL_AFMKIT_PATH: $MACLOCAL_AFMKIT_PATH" >&2
    exit 1
  }
  echo "[afmkit-auth] Local AFMKit development uses an isolated generated package workspace."
  echo "[afmkit-auth] Invoke Scripts/swiftpm-reliable.sh with MACLOCAL_AFMKIT_PATH set; the tracked lock is not resolved or changed here."
  exit 0
fi

if [[ -n "${MACLOCAL_AFMKIT_WORKSPACE_PATH:-}" ]]; then
  echo "[afmkit-auth] MACLOCAL_AFMKIT_WORKSPACE_PATH is an internal wrapper variable and is not a release input." >&2
  exit 2
fi

AFMKIT_RELEASE_SOURCES=()
while IFS= read -r source; do
  AFMKIT_RELEASE_SOURCES+=("$source")
done < <(python3 - <<'PY'
import json
lock = json.load(open("Package.resolved"))
pins = {pin["identity"]: pin for pin in lock["pins"]}
for identity in ("afmkit",):
    pin = pins.get(identity)
    if pin is None:
        raise SystemExit(f"missing private release lock: {identity}")
    state = pin["state"]
    version = state.get("version", "")
    print("\t".join((identity, pin["location"], state["revision"], version)))
PY
)

AUTH_ROOT="$ROOT_DIR/.build/private-afmkit-auth"
mkdir -p "$AUTH_ROOT"
ASKPASS="$AUTH_ROOT/askpass.sh"
cat > "$ASKPASS" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *) printf '%s\n' "${AFMKIT_READ_TOKEN:?AFMKIT_READ_TOKEN is not set}" ;;
esac
ASKPASS
chmod 700 "$ASKPASS"
GIT_COMMAND="${AFMKIT_GIT_COMMAND:-git}"

run_authenticated() {
  if [[ -n "${AFMKIT_READ_TOKEN:-}" ]]; then
    env \
      GIT_ASKPASS="$ASKPASS" \
      GIT_TERMINAL_PROMPT=0 \
      GIT_CONFIG_COUNT=1 \
      GIT_CONFIG_KEY_0=credential.helper \
      GIT_CONFIG_VALUE_0= \
      "$@"
  else
    env GIT_TERMINAL_PROMPT=0 "$@"
  fi
}

for source in "${AFMKIT_RELEASE_SOURCES[@]}"; do
  IFS=$'\t' read -r AFMKIT_IDENTITY AFMKIT_URL AFMKIT_REVISION AFMKIT_VERSION <<< "$source"
  AFMKIT_REMOTE_REFS="$(
    run_authenticated "$GIT_COMMAND" ls-remote --tags "$AFMKIT_URL" \
      "refs/tags/$AFMKIT_VERSION" \
      "refs/tags/$AFMKIT_VERSION^{}" \
      "refs/tags/v$AFMKIT_VERSION" \
      "refs/tags/v$AFMKIT_VERSION^{}" \
      2>/dev/null || true
  )"
  if grep -Fq "$AFMKIT_REVISION" <<<"$AFMKIT_REMOTE_REFS"; then
    echo "[afmkit-auth] Access verified for $AFMKIT_IDENTITY ${AFMKIT_REVISION:0:12}."
    continue
  fi
  cat >&2 <<EOF
[afmkit-auth] Cannot read the exact provider dependency at $AFMKIT_URL.
[afmkit-auth] Confirm network access and that tag $AFMKIT_VERSION resolves to
[afmkit-auth] ${AFMKIT_REVISION:0:12}. For an explicitly private development source,
[afmkit-auth] authenticate with 'gh auth login' and 'gh auth setup-git', or provide a
[afmkit-auth] masked AFMKIT_READ_TOKEN with repository read access. Production release
[afmkit-auth] eligibility always requires anonymous access and never relies on a token.
EOF
  exit 1
done
$CHECK_ONLY && exit 0

if $REFRESH_LOCK; then
  run_authenticated swift package resolve
else
  run_authenticated swift package --force-resolved-versions resolve
fi
"$ROOT_DIR/Scripts/check-afmkit-consumer-boundary.sh"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
   ! git diff --quiet -- Package.resolved && ! $REFRESH_LOCK; then
  echo "[afmkit-auth] Dependency resolution changed the tracked release lock." >&2
  echo "[afmkit-auth] Review and commit Package.resolved explicitly; release resolution is fail-closed." >&2
  exit 1
fi

if $REFRESH_LOCK; then
  echo "[afmkit-auth] Refreshed Package.resolved; review and commit the lock explicitly."
  exit 0
fi

echo "[afmkit-auth] Resolved the complete tracked release graph without drift."
