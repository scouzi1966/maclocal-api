#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

read_afmkit_release_source() {
  python3 - "$ROOT_DIR/Package.resolved" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1]))
pin = next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")
print(pin["location"], pin["state"]["revision"])
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
  local url revision
  read -r url revision < <(read_afmkit_release_source)

  if [[ ! "$url" =~ ^https:// ]]; then
    echo "[release-public] AFMKit release source must be a public HTTPS package URL: $url" >&2
    return 1
  fi

  if ! probe_without_credentials "$url" "$revision"; then
    cat >&2 <<EOF
[release-public] Production publishing is blocked: AFMKit is not anonymously
[release-public] fetchable at the locked revision ${revision} from ${url}.
[release-public] An AFMKIT_READ_TOKEN may authenticate development builds, but it
[release-public] cannot satisfy the public distribution requirement.
[release-public] Make this exact package revision public, or replace Package.swift
[release-public] and Package.resolved with an approved public immutable package/artifact.
EOF
    return 1
  fi

  echo "[release-public] AFMKit ${revision:0:12} is anonymously fetchable from $url."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  check_public_release_eligibility
fi
