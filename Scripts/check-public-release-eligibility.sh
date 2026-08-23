#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

read_public_release_sources() {
  python3 - "$ROOT_DIR" <<'PY'
import json, os, subprocess, sys
root = sys.argv[1]
lock = json.load(open(os.path.join(root, "Package.resolved")))
pins = {pin["identity"]: pin for pin in lock["pins"]}
env = os.environ.copy()
for name in ("MACLOCAL_AFMKIT_PATH", "MACLOCAL_AFMKIT_WORKSPACE_PATH"):
    env.pop(name, None)
package = json.loads(subprocess.check_output(["swift", "package", "dump-package"], cwd=root, env=env))
dependencies = {item["sourceControl"][0]["identity"]: item["sourceControl"][0]
                for item in package["dependencies"] if item.get("sourceControl")}
expected_versions = {
    "afmkit": "0.1.1",
}
for identity, expected_version in expected_versions.items():
    pin, dependency = pins.get(identity), dependencies.get(identity)
    if pin is None or dependency is None:
        raise SystemExit(f"missing release dependency: {identity}")
    requirement = dependency["requirement"]
    if set(requirement) != {"exact"}:
        raise SystemExit(f"{identity} must use an exact semantic version")
    version = requirement["exact"][0]
    if version != expected_version:
        raise SystemExit(f"{identity} must use exact version {expected_version}")
    if pin["state"].get("version") != version:
        raise SystemExit(f"{identity} lock does not match exact version {version}")
    print("\t".join((identity, pin["location"], pin["state"]["revision"], version)))
PY
}

probe_public_source() {
  local url="$1" revision="$2" probe_root="$3"
  local repository="$probe_root/repository.git"
  rm -rf "$probe_root"
  mkdir -p "$probe_root/home"
  git -C "$probe_root" init --bare --quiet --initial-branch=main "$repository"
  env -u AFMKIT_READ_TOKEN -u GH_TOKEN -u GITHUB_TOKEN -u GIT_ASKPASS \
    -u SSH_AUTH_SOCK HOME="$probe_root/home" GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/usr/bin/false \
    git -C "$repository" -c credential.helper= fetch --quiet --depth 1 "$url" "$revision"
  [[ "$(git -C "$repository" rev-parse FETCH_HEAD)" == "$revision" ]]
}

check_public_release_eligibility() {
  local identity url revision version
  while IFS=$'\t' read -r identity url revision version; do
    [[ "$url" =~ ^https://github\.com/ ]] || {
      echo "[release-public] $identity must use a public GitHub HTTPS URL: $url" >&2
      return 1
    }
    if ! probe_public_source "$url" "$revision" "$ROOT_DIR/.build/public-provider-probe/$identity"; then
      echo "[release-public] $identity $version is not anonymously fetchable at ${revision:0:12}." >&2
      return 1
    fi
    echo "[release-public] $identity $version (${revision:0:12}) is exact and anonymously fetchable."
  done < <(read_public_release_sources)
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  check_public_release_eligibility
fi
