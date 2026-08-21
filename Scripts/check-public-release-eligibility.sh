#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

read_afmkit_release_source() {
  python3 - "$ROOT_DIR" <<'PY'
import json
import os
import subprocess
import sys

root = sys.argv[1]
lock = json.load(open(os.path.join(root, "Package.resolved")))
pin = next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")

environment = os.environ.copy()
for name in (
    "MACLOCAL_AFMKIT_PATH",
    "MACLOCAL_AFMKIT_WORKSPACE_PATH",
    "MACLOCAL_MLX_SWIFT_LM_PATH",
    "AFMKIT_MLX_SWIFT_PATH",
    "AFMKIT_MLX_SWIFT_LM_PATH",
):
    environment.pop(name, None)
package = json.loads(
    subprocess.check_output(
        ["swift", "package", "dump-package"], cwd=root, env=environment
    )
)
dependency = next(
    dependency["sourceControl"][0]
    for dependency in package["dependencies"]
    if dependency.get("sourceControl", [{}])[0].get("identity") == "afmkit"
)
requirement = dependency["requirement"]
if len(requirement) != 1:
    raise SystemExit("AFMKit manifest requirement is ambiguous")
requirement_kind, values = next(iter(requirement.items()))
requirement_value = values[0]
print(
    pin["location"],
    pin["state"]["revision"],
    requirement_kind,
    requirement_value,
    pin["state"].get("version", "-"),
)
PY
}

probe_public_afmkit_source() {
  local url="$1"
  local revision="$2"
  local probe_root="$ROOT_DIR/.build/public-afmkit-release-probe"
  local repository="$probe_root/repository.git"

  rm -rf "$probe_root"
  mkdir -p "$probe_root/home"
  git -C "$probe_root" init --bare --quiet --initial-branch=main "$repository"
  env \
    -u AFMKIT_READ_TOKEN \
    -u GH_TOKEN \
    -u GITHUB_TOKEN \
    -u GIT_ASKPASS \
    -u SSH_AUTH_SOCK \
    HOME="$probe_root/home" \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_TERMINAL_PROMPT=0 \
    GIT_ASKPASS=/usr/bin/false \
    git -C "$repository" \
      -c credential.helper= \
      fetch --quiet --depth 1 "$url" "$revision" \
      2>"$probe_root/fetch-error.log"
  [[ "$(git -C "$repository" rev-parse FETCH_HEAD)" == "$revision" ]]
}

probe_without_credentials() (
  unset AFMKIT_READ_TOKEN GH_TOKEN GITHUB_TOKEN GIT_ASKPASS SSH_AUTH_SOCK
  probe_public_afmkit_source "$@"
)

check_public_release_eligibility() {
  local url revision requirement_kind requirement_value resolved_version
  read -r url revision requirement_kind requirement_value resolved_version \
    < <(read_afmkit_release_source)

  if [[ ! "$url" =~ ^https:// ]]; then
    echo "[release-public] AFMKit release source must be a public HTTPS package URL: $url" >&2
    return 1
  fi

  if [[ "$requirement_kind" != "exact" ]] || \
     [[ ! "$requirement_value" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]] || \
     [[ "$resolved_version" != "$requirement_value" ]]; then
    cat >&2 <<EOF
[release-public] Production publishing is blocked: maclocal-api exposes a SwiftPM
[release-public] source-package surface while AFMKit uses the ${requirement_kind}
[release-public] requirement ${requirement_value}. Anonymous access to revision
[release-public] ${revision} is not a versioned package release contract.
[release-public] Replace the dependency with an exact public semantic version and
[release-public] commit a matching Package.resolved version/revision before publishing.
[release-public] Until then, no release workflow may claim SwiftPM publishability;
[release-public] packaged binaries do not remove the repository's source-package surface.
EOF
    return 1
  fi

  if ! probe_without_credentials "$url" "$revision"; then
    cat >&2 <<EOF
[release-public] Production publishing is blocked: AFMKit is not anonymously
[release-public] fetchable at the locked revision ${revision} from ${url}.
[release-public] An AFMKIT_READ_TOKEN may authenticate development builds, but it
[release-public] cannot satisfy the public distribution requirement.
[release-public] Publish AFMKit ${requirement_value} publicly at the locked revision,
[release-public] or exclude maclocal-api's source-package surface from the release policy.
EOF
    return 1
  fi

  echo "[release-public] AFMKit ${requirement_value} (${revision:0:12}) is an exact, anonymously fetchable package."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  check_public_release_eligibility
fi
