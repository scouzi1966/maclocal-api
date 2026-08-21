#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

fail() {
  echo "[afmkit-boundary] $*" >&2
  exit 1
}

for shadow_target in AFMKitCore AFMKitMLX AFMKitDwarfStar AFMOpenAICompat; do
  [[ ! -d "Sources/$shadow_target" ]] || \
    fail "consumer shadow target still exists: Sources/$shadow_target"
done

for provider_owned_path in \
  Sources/CDwarfStar \
  Sources/CXGrammar \
  vendor/ds4 \
  vendor/xgrammar \
  Tests/AFMKitDwarfStarTests \
  Scripts/check-afmkit-core-api.sh \
  docs/api-baselines; do
  [[ ! -e "$provider_owned_path" ]] || \
    fail "AFMKit-owned provider surface remains in the consumer: $provider_owned_path"
done

facade_files=(Sources/AFMKitFoundationModels/*.swift)
[[ ${#facade_files[@]} -eq 1 && -f "${facade_files[0]}" ]] || \
  fail "AFMKitFoundationModels must contain only its compatibility facade"
grep -Fqx '@_exported import AFMKitApple' "${facade_files[0]}" || \
  fail "AFMKitFoundationModels must re-export the AFMKitApple product"
if grep -Eq '\b(class|struct|enum|protocol|actor|func)[[:space:]]+' "${facade_files[0]}"; then
  fail "AFMKitFoundationModels facade contains a local implementation"
fi

[[ -f Package.resolved ]] || \
  fail "tracked Package.resolved is missing; restore it before resolving dependencies"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files --error-unmatch -- Package.resolved >/dev/null 2>&1 || \
    fail "Package.resolved must be intentionally tracked as the release lock"
fi

python3 - <<'PY'
import json
import os
import re
import subprocess
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[afmkit-boundary] {message}")


lock = json.loads(Path("Package.resolved").read_text())
if not re.fullmatch(r"[0-9a-f]{64}", lock.get("originHash", "")):
    fail("Package.resolved has no valid manifest origin hash")

pins = lock.get("pins", [])
if not pins:
    fail("Package.resolved contains no dependency pins")

pins_by_identity = {}
for pin in pins:
    identity = pin.get("identity")
    if not identity or identity in pins_by_identity:
        fail(f"duplicate or missing package identity: {identity!r}")
    if pin.get("kind") != "remoteSourceControl":
        fail(f"{identity} is not pinned as remote source control")
    state = pin.get("state", {})
    revision = state.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail(f"{identity} has no immutable 40-character revision")
    if state.get("branch"):
        fail(f"{identity} is resolved from mutable branch {state['branch']!r}")
    pins_by_identity[identity] = pin

manifest_environment = os.environ.copy()
for name in (
    "MACLOCAL_AFMKIT_PATH",
    "MACLOCAL_AFMKIT_WORKSPACE_PATH",
    "MACLOCAL_MLX_SWIFT_LM_PATH",
    "AFMKIT_MLX_SWIFT_PATH",
    "AFMKIT_MLX_SWIFT_LM_PATH",
):
    manifest_environment.pop(name, None)
package = json.loads(
    subprocess.check_output(
        ["swift", "package", "dump-package"], env=manifest_environment
    )
)
direct_identities = set()
for dependency in package.get("dependencies", []):
    source = dependency.get("sourceControl")
    if not source:
        fail("release manifest contains a local or non-source-control dependency")
    descriptor = source[0]
    identity = descriptor["identity"]
    direct_identities.add(identity)
    pin = pins_by_identity.get(identity)
    if pin is None:
        fail(f"direct dependency {identity} is absent from Package.resolved")
    requirement = descriptor.get("requirement", {})
    if "revision" in requirement:
        required_revision = requirement["revision"][0]
        if pin["state"]["revision"] != required_revision:
            fail(f"{identity} does not match its manifest revision")
    if "exact" in requirement:
        required_version = requirement["exact"][0]
        if pin["state"].get("version") != required_version:
            fail(f"{identity} does not match exact version {required_version}")

required_direct = {
    "afmkit",
    "mlx-swift-afm",
    "mlx-swift-lm",
}
missing = sorted(required_direct - direct_identities)
if missing:
    fail(f"required AFM release dependencies are missing: {', '.join(missing)}")

afmkit = pins_by_identity["afmkit"]
if afmkit["location"] != "https://github.com/scouzi1966/AFMKit.git":
    fail("AFMKit release lock points at an unexpected source")

checkout = Path(".build/checkouts/AFMKit")
if checkout.is_dir():
    expected_revision = afmkit["state"]["revision"]
    actual_revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != expected_revision:
        fail(
            "resolved AFMKit checkout differs from Package.resolved: "
            f"expected {expected_revision}, got {actual_revision}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        fail("resolved AFMKit checkout contains local modifications")

    repository = subprocess.check_output(
        ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True
    ).strip()
    repository_path = Path(repository)
    actual_location = repository
    if repository_path.is_dir():
        actual_location = subprocess.check_output(
            ["git", "-C", str(repository_path), "remote", "get-url", "origin"],
            text=True,
        ).strip()

    def normalized_location(location: str) -> str:
        location = re.sub(r"^git@github\.com:", "", location)
        location = re.sub(r"^https://github\.com/", "", location)
        return re.sub(r"\.git$", "", location)

    if normalized_location(actual_location) != normalized_location(afmkit["location"]):
        fail(f"resolved AFMKit checkout came from unexpected source {actual_location}")

gitlinks = subprocess.check_output(
    ["git", "ls-files", "-s", "vendor"], text=True
).splitlines()
gitlinks_by_path = {}
for line in gitlinks:
    fields = line.split(None, 3)
    if fields and fields[0] == "160000":
        if not re.fullmatch(r"[0-9a-f]{40}", fields[1]):
            fail(f"invalid submodule revision in release graph: {line}")
        gitlinks_by_path[fields[3]] = fields[1]
for forbidden_gitlink in ("vendor/ds4", "vendor/xgrammar"):
    if forbidden_gitlink in gitlinks_by_path:
        fail(f"provider dependency is still consumer-owned: {forbidden_gitlink}")
declared_paths = subprocess.check_output(
    ["git", "config", "-f", ".gitmodules", "--get-regexp", "path"], text=True
).splitlines()
for line in declared_paths:
    path = line.split(None, 1)[1]
    if path not in gitlinks_by_path:
        fail(f"declared submodule is not pinned by a gitlink: {path}")
PY

if grep -ERn \
  '@testable import (AFMKitCore|AFMKitMLX|AFMKitDwarfStar|AFMOpenAICompat|AFMKitApple)' \
  Tests --include='*.swift' >/dev/null; then
  fail "consumer tests reach into AFMKit provider internals"
fi

if grep -Fq 'environment["MACLOCAL_AFMKIT_PATH"]' Package.swift; then
  fail "MACLOCAL_AFMKIT_PATH must not alter the tracked release manifest"
fi
grep -Fq 'environment["MACLOCAL_AFMKIT_WORKSPACE_PATH"]' Package.swift || \
  fail "generated local AFMKit workspace hook is missing from Package.swift"

if grep -Eq '^(build|debug):.*(PATCH_STAMP|patch)' Makefile; then
  fail "normal Make targets must not depend on the legacy patch stack"
fi
if grep -Fq '$(PATCH_STAMP)' Makefile; then
  fail "Makefile still uses a vendor patch stamp"
fi

if grep -ERn \
  'MacLocalAPI_AFMKit(MLX|DwarfStar)\.bundle' \
  .github/workflows Scripts/build-native-wheel.sh Scripts/build-nightly-wheel.sh \
  Scripts/build-stable-wheel.sh Scripts/create-tarball.sh \
  Scripts/generate-tap-versioned.sh Scripts/publish-next.sh Scripts/publish-stable.sh \
  Scripts/verify-native-wheel.sh Scripts/verify-release-archive.sh >/dev/null; then
  fail "release packaging still references a maclocal-api-owned provider bundle"
fi

for workflow in .github/workflows/nightly.yml .github/workflows/release.yml; do
  grep -Fq 'AFMKit_AFMKitMLX.bundle' "$workflow" || \
    fail "$workflow does not package the AFMKit MLX resource bundle"
  grep -Fq 'AFMKit_AFMKitDwarfStar.bundle' "$workflow" || \
    fail "$workflow does not package the AFMKit DwarfStar resource bundle"
  grep -Fq 'Scripts/validate-release.sh' "$workflow" || \
    fail "$workflow does not run the complete local release gate"
  grep -Fq 'AFMKIT_READ_TOKEN' "$workflow" || \
    fail "$workflow does not declare authenticated private AFMKit access"
  grep -Fq 'Scripts/check-public-release-eligibility.sh' "$workflow" || \
    fail "$workflow can publish without proving anonymous dependency access"
done

for publisher in Scripts/publish-next.sh Scripts/publish-stable.sh; do
  grep -Fq 'check-public-release-eligibility.sh' "$publisher" || \
    fail "$publisher can publish without proving anonymous dependency access"
done

grep -Fq 'Scripts/resolve-release-dependencies.sh' .github/workflows/codeql-analysis.yml || \
  fail "CodeQL does not use the authenticated transition resolver"
grep -Fq 'AFMKIT_READ_TOKEN' .github/workflows/codeql-analysis.yml || \
  fail "CodeQL does not declare private AFMKit authentication"
grep -Fq 'head.repo.fork' .github/workflows/codeql-analysis.yml || \
  fail "CodeQL does not isolate private credentials from fork pull requests"
if grep -Eq '^[[:space:]]*swift package resolve[[:space:]]*$' .github/workflows/codeql-analysis.yml; then
  fail "CodeQL still performs a bare unauthenticated dependency resolve"
fi

example_manifest=Examples/AFMKitCoreOnlyConsumer/Package.swift
grep -Fq '.product(name: "AFMKitCore", package: "AFMKit")' "$example_manifest" || \
  fail "independent core consumer does not use AFMKit's package contract"
if grep -Fq 'package: "MacLocalAPI"' "$example_manifest"; then
  fail "independent core consumer still relies on a removed maclocal compatibility product"
fi
afmkit_revision="$(python3 - <<'PY'
import json
lock = json.load(open("Package.resolved"))
print(next(pin for pin in lock["pins"] if pin["identity"] == "afmkit")["state"]["revision"])
PY
)"
grep -Fq "revision: \"$afmkit_revision\"" "$example_manifest" || \
  fail "independent core consumer revision differs from the tracked AFMKit lock"

for project in pyproject.toml pyproject-next.toml; do
  grep -Fq '"bin/*/*/*/*/*"' "$project" || \
    fail "$project does not package nested Xcode 27 provider resources"
done

grep -Fq '.macOS("26.0")' Package.swift || \
  fail "Package.swift no longer preserves the macOS 26 deployment boundary"

if grep -ERn \
  '\b(MLXModelService|BatchScheduler|RequestSlot|RadixTreeCache|ToolCallStreamingRuntime)\b' \
  Sources/AFMServer \
  | grep -Ev '^[^:]+:[0-9]+:[[:space:]]*//' >/dev/null; then
  fail "AFMServer reaches into AFMKitMLX implementation types"
fi

# Reject provider declarations even if a copied source was renamed or lightly
# reformatted. When AFMKit is already resolved, also compare normalized source
# bodies to catch near copies without relying on filenames.
python3 - <<'PY'
import difflib
import re
from pathlib import Path


def normalized(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return re.sub(r"\s+", "", text)


owned_types = {
    "FoundationModelService",
    "FoundationModelError",
    "JSONSchemaConverter",
    "OpenAIRequest",
    "OpenAIResponse",
    "OpenAIResponseFormatPolicy",
}
declaration = re.compile(
    r"\b(?:class|struct|enum|protocol|actor)\s+(" + "|".join(owned_types) + r")\b"
)
source_suffixes = {".swift", ".c", ".cc", ".cpp", ".h", ".m", ".mm"}
local_files = [
    path for path in Path("Sources").rglob("*")
    if path.is_file() and path.suffix in source_suffixes
]
for path in local_files:
    if declaration.search(path.read_text(errors="ignore")):
        raise SystemExit(
            f"[afmkit-boundary] AFMKit-owned provider declaration copied into {path}"
        )

checkout = Path(".build/checkouts/AFMKit/Sources")
if checkout.is_dir():
    provider_roots = [
        checkout / "AFMKitApple",
        checkout / "AFMOpenAICompat",
        checkout / "AFMKitMLX",
        checkout / "AFMKitDwarfStar",
        checkout / "CDwarfStar",
        checkout / "CXGrammar",
    ]
    provider_sources = []
    for root in provider_roots:
        if root.is_dir():
            provider_sources.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix in source_suffixes
            )
    remote_bodies = [
        (path, normalized(path.read_text(errors="ignore")))
        for path in provider_sources
    ]
    for local in local_files:
        body = normalized(local.read_text(errors="ignore"))
        if len(body) < 500:
            continue
        for remote, remote_body in remote_bodies:
            if abs(len(body) - len(remote_body)) > max(len(body), len(remote_body)) * 0.12:
                continue
            if difflib.SequenceMatcher(None, body, remote_body).ratio() >= 0.92:
                raise SystemExit(
                    f"[afmkit-boundary] near-copied provider source: {local} matches {remote}"
                )
PY

echo "[afmkit-boundary] tracked immutable graph, AFMKit facades, and release ownership verified"
