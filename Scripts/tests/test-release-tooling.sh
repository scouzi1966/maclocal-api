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

codeql_workflow="$ROOT_DIR/.github/workflows/codeql-analysis.yml"
if grep -Eq 'AFMKIT_READ_TOKEN|private-dependency-notice|while AFMKit is private' "$codeql_workflow"; then
  fail "CodeQL still treats the public AFMKit release as a private dependency"
fi
grep -Fq 'Scripts/resolve-release-dependencies.sh' "$codeql_workflow" || \
  fail "CodeQL does not resolve the tracked public release graph"

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
grep -Fq 'Cannot read the exact provider dependency' "$auth_log" || \
  fail "AFMKit access error is not actionable"
grep -Fq 'AFMKIT_READ_TOKEN' "$auth_log" || \
  fail "private AFMKit error does not name the CI secret"
if grep -Fq 'must-not-appear' "$auth_log"; then
  fail "private AFMKit token leaked into diagnostics"
fi

expected_afmkit_revision="$(python3 - "$ROOT_DIR/Package.resolved" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1]))
pin = next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")
print(pin["state"]["revision"])
PY
)"
fake_public_git="$WORK_ROOT/git-public"
cat > "$fake_public_git" <<'SH'
#!/usr/bin/env bash
arguments=" $* "
for ref in \
  refs/tags/0.1.16 \
  'refs/tags/0.1.16^{}' \
  refs/tags/v0.1.16 \
  'refs/tags/v0.1.16^{}'; do
  [[ "$arguments" == *" $ref "* ]] || exit 2
done
printf '%s\t%s\n' "$EXPECTED_AFMKIT_REVISION" 'refs/tags/v0.1.16^{}'
SH
chmod 700 "$fake_public_git"
if ! EXPECTED_AFMKIT_REVISION="$expected_afmkit_revision" \
   AFMKIT_GIT_COMMAND="$fake_public_git" \
   "$ROOT_DIR/Scripts/resolve-release-dependencies.sh" --check-access \
   >"$WORK_ROOT/public-access.log" 2>&1; then
  fail "v-prefixed public AFMKit release was not recognized"
fi

if grep -Fq '.github/workflows' "$ROOT_DIR/Scripts/check-afmkit-consumer-boundary.sh" || \
   grep -Fq 'AFMKIT_READ_TOKEN' "$ROOT_DIR/Scripts/check-afmkit-consumer-boundary.sh"; then
  fail "local consumer boundary still depends on hosted workflow configuration"
fi

public_gate_log="$WORK_ROOT/public-release-error.log"
if (
  export AFMKIT_READ_TOKEN="public-gate-secret"
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  read_public_release_sources() {
    echo $'afmkit\thttps://github.com/scouzi1966/AFMKit.git\t1111111111111111111111111111111111111111\t0.1.16'
  }
  probe_public_source() {
    return 1
  }
  check_public_release_eligibility
) >"$public_gate_log" 2>&1; then
  fail "production public-release gate accepted a private dependency"
fi
grep -Fq 'is not anonymously fetchable' "$public_gate_log" || \
  fail "public-release failure is not actionable"
if grep -Fq 'public-gate-secret' "$public_gate_log"; then
  fail "public-release gate leaked a development token"
fi

release_source="$(
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  read_public_release_sources
)"
IFS=$'\t' read -r release_identity release_url release_revision release_version <<<"$release_source"
[[ "$release_identity" == "afmkit" ]] || fail "release source identity is not AFMKit"
[[ "$release_url" == "https://github.com/scouzi1966/AFMKit.git" ]] || \
  fail "release source is not the canonical public HTTPS repository"
[[ "$release_revision" =~ ^[0-9a-f]{40}$ ]] || fail "release lock revision is not immutable"
[[ "$release_version" == "0.1.16" ]] || fail "release manifest is not pinned to exact AFMKit 0.1.16"

private_gate_log="$WORK_ROOT/private-version-error.log"
if (
  export AFMKIT_READ_TOKEN="private-gate-secret"
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  read_public_release_sources() {
    echo $'afmkit\thttps://github.com/scouzi1966/AFMKit.git\t1111111111111111111111111111111111111111\t0.1.16'
  }
  probe_public_source() {
    [[ -z "${AFMKIT_READ_TOKEN:-}" ]]
  }
  check_public_release_eligibility
) >"$private_gate_log" 2>&1; then
  fail "production public-release gate accepted an authenticated-only exact dependency"
fi
grep -Fq 'is not anonymously fetchable' "$private_gate_log" || \
  fail "public-release gate did not reject the authenticated-only exact dependency"
if grep -Fq 'private-gate-secret' "$private_gate_log"; then
  fail "public-release gate leaked a development token"
fi

if ! (
  export AFMKIT_READ_TOKEN="public-gate-secret"
  source "$ROOT_DIR/Scripts/check-public-release-eligibility.sh"
  read_public_release_sources() {
    echo $'afmkit\thttps://github.com/scouzi1966/AFMKit.git\t1111111111111111111111111111111111111111\t0.1.16'
  }
  probe_public_source() {
    return 0
  }
  check_public_release_eligibility
) >"$WORK_ROOT/public-release-success.log" 2>&1; then
  fail "production public-release gate rejected an anonymous exact dependency"
fi

prefix="$WORK_ROOT/custom-prefix"
mkdir -p \
  "$prefix/bin/MacLocalAPI_AFMKit.bundle" \
  "$prefix/bin/MacLocalAPI_AFMEvaluationHost.bundle" \
  "$prefix/bin/AFMKit_AFMKitMLX.bundle" \
  "$prefix/bin/AFMKit_AFMKitDwarfStar.bundle" \
  "$prefix/libexec/afm/MacLocalAPI_AFMKit.bundle/Evals" \
  "$prefix/libexec/afm/MacLocalAPI_AFMEvaluationHost.bundle/Evals" \
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
[[ ! -e "$prefix/bin/MacLocalAPI_AFMKit.bundle" ]] || fail "legacy evaluation bundle was not removed"
[[ ! -e "$prefix/libexec/afm/MacLocalAPI_AFMKit.bundle" ]] || fail "legacy libexec evaluation bundle was not removed"
[[ ! -e "$prefix/bin/MacLocalAPI_AFMEvaluationHost.bundle" ]] || fail "evaluation bundle was not removed"
[[ ! -e "$prefix/libexec/afm/MacLocalAPI_AFMEvaluationHost.bundle" ]] || fail "libexec evaluation bundle was not removed"
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
