#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="$ROOT_DIR/.build/release-tooling-tests"
rm -rf "$WORK_ROOT"
mkdir -p "$WORK_ROOT"

fail() {
  echo "[release-tooling-test] $*" >&2
  exit 1
}

"$ROOT_DIR/Scripts/check-afmkit-consumer-boundary.sh"

mkdir -p "$WORK_ROOT/nested/AFMKit_AFMKitMLX.bundle/Contents/Resources"
printf 'fixture' > "$WORK_ROOT/nested/AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib"
resolved="$($ROOT_DIR/Scripts/resolve-afmkit-resource.sh --metallib "$WORK_ROOT/nested")"
[[ "$resolved" == */Contents/Resources/default.metallib ]] || \
  fail "nested Xcode 27 metallib was not resolved"

fake_git="$WORK_ROOT/git-denied"
cat > "$fake_git" <<'SH'
#!/usr/bin/env bash
exit 128
SH
chmod 700 "$fake_git"
auth_log="$WORK_ROOT/auth-error.log"
if AFMKIT_GIT_COMMAND="$fake_git" \
   AFMKIT_READ_TOKEN="must-not-appear" \
   "$ROOT_DIR/Scripts/resolve-release-dependencies.sh" --check-access \
   >"$auth_log" 2>&1; then
  fail "unauthenticated AFMKit access unexpectedly succeeded"
fi
grep -Fq 'Cannot read the private AFMKit dependency' "$auth_log" || \
  fail "private AFMKit error is not actionable"
grep -Fq 'AFMKIT_READ_TOKEN' "$auth_log" || \
  fail "private AFMKit error does not name the CI secret"
if grep -Fq 'must-not-appear' "$auth_log"; then
  fail "private AFMKit token leaked into diagnostics"
fi

public_gate_log="$WORK_ROOT/public-release-error.log"
if (
  export AFMKIT_READ_TOKEN="public-gate-secret"
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  probe_public_afmkit_source() {
    [[ -z "${AFMKIT_READ_TOKEN:-}" ]] || return 9
    return 1
  }
  check_public_release_eligibility
) >"$public_gate_log" 2>&1; then
  fail "production public-release gate accepted a private dependency"
fi
grep -Fq 'Production publishing is blocked' "$public_gate_log" || \
  fail "public-release failure is not actionable"
grep -Fq 'cannot satisfy the public distribution requirement' "$public_gate_log" || \
  fail "public-release failure does not distinguish development authentication"
if grep -Fq 'public-gate-secret' "$public_gate_log"; then
  fail "public-release gate leaked a development token"
fi

if ! (
  export AFMKIT_READ_TOKEN="public-gate-secret"
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  probe_public_afmkit_source() {
    [[ -z "${AFMKIT_READ_TOKEN:-}" ]]
  }
  check_public_release_eligibility
) >"$WORK_ROOT/public-release-success.log" 2>&1; then
  fail "production public-release gate rejected an anonymous immutable source"
fi

prefix="$WORK_ROOT/custom-prefix"
mkdir -p \
  "$prefix/bin/AFMKit_AFMKitMLX.bundle" \
  "$prefix/bin/AFMKit_AFMKitDwarfStar.bundle" \
  "$prefix/libexec/afm/AFMKit_AFMKitMLX.bundle/Contents/Resources" \
  "$prefix/libexec/afm/AFMKit_AFMKitDwarfStar.bundle/metal" \
  "$prefix/share/afm/webui"
printf 'binary' > "$prefix/bin/afm"
printf 'unrelated' > "$prefix/bin/keep-me"
printf 'unrelated' > "$prefix/libexec/afm/keep-me"
printf 'unrelated' > "$prefix/share/afm/webui/keep-me"
printf 'resource' > "$prefix/share/afm/webui/index.html.gz"
INSTALL_PREFIX="$prefix" "$ROOT_DIR/Scripts/uninstall.sh"
[[ ! -e "$prefix/bin/afm" ]] || fail "custom-prefix binary was not removed"
[[ ! -e "$prefix/bin/AFMKit_AFMKitMLX.bundle" ]] || fail "custom-prefix bundle was not removed"
[[ ! -e "$prefix/libexec/afm/AFMKit_AFMKitDwarfStar.bundle" ]] || fail "libexec bundle was not removed"
[[ ! -e "$prefix/share/afm/webui/index.html.gz" ]] || fail "WebUI was not removed"
[[ -f "$prefix/bin/keep-me" ]] || fail "uninstall deleted an unrelated bin file"
[[ -f "$prefix/libexec/afm/keep-me" ]] || fail "uninstall deleted an unrelated libexec file"
[[ -f "$prefix/share/afm/webui/keep-me" ]] || fail "uninstall deleted an unrelated WebUI file"

for project in "$ROOT_DIR/pyproject.toml" "$ROOT_DIR/pyproject-next.toml"; do
  grep -Fq '"bin/*/*/*/*/*"' "$project" || \
    fail "$(basename "$project") does not include nested Xcode 27 bundle resources"
done

echo "[release-tooling-test] release lock, authentication boundaries, nested resources, and uninstall verified"
