#!/usr/bin/env python3
"""Package an existing AFM build for local PCC development; never alter keychains."""
import argparse
import datetime
import hashlib
import json
import pathlib
import plistlib
import re
import shutil
import subprocess
import sys
from xml.parsers.expat import ExpatError

ROOT = pathlib.Path(__file__).resolve().parents[1]
PCC = "com.apple.developer.private-cloud-compute"
APP_ID = "com.apple.application-identifier"
TEAM_ID = "com.apple.developer.team-identifier"
APP_ATTEST = "com.apple.developer.devicecheck.app-attest-opt-in"
MACHO_MAGIC = {bytes.fromhex(value) for value in (
    "feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca", "cafebabf", "bfbafeca")}


def run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE)


def validate_profile(profile, bundle_id, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    expires = profile.get("ExpirationDate")
    if not isinstance(expires, datetime.datetime) or expires.replace(tzinfo=datetime.timezone.utc) <= now:
        raise ValueError("Provisioning profile is expired or has no expiration date.")
    if "OSX" not in profile.get("Platform", []):
        raise ValueError("A macOS provisioning profile is required.")
    entitlements = profile.get("Entitlements", {})
    if entitlements.get(PCC) is not True:
        raise ValueError("Provisioning profile does not authorize Private Cloud Compute.")
    team = entitlements.get(TEAM_ID)
    app_id = entitlements.get(APP_ID, "")
    # App ID prefixes need not equal Team IDs; preserve the profile's exact value.
    if not team or app_id.partition(".")[2] != bundle_id or "*" in app_id:
        raise ValueError(f"Profile authorizes {app_id!r}, but this AFM build uses {bundle_id!r}. Generate an AFM-specific PCC profile; do not reuse Vesta's identity.")
    if not profile.get("ProvisionedDevices") or profile.get("ProvisionsAllDevices"):
        raise ValueError("Use a device-bound development profile for local PCC testing.")
    signed = {PCC: True, APP_ID: app_id, TEAM_ID: team}
    if APP_ATTEST in entitlements:
        signed[APP_ATTEST] = entitlements[APP_ATTEST]
    return signed


def select_identity(profile, requested, listing):
    allowed = {hashlib.sha1(cert).hexdigest().upper() for cert in profile.get("DeveloperCertificates", [])}
    identities = re.findall(r'\b([0-9A-Fa-f]{40})\s+"([^"]+)"', listing)
    matches = [(fingerprint.upper(), name) for fingerprint, name in identities
               if (requested.upper() == fingerprint.upper() or requested == name)
               and fingerprint.upper() in allowed
               and name.startswith(("Apple Development:", "Mac Developer:"))]
    if len(matches) != 1:
        raise ValueError("Select a valid Apple Development identity (name or SHA-1) whose certificate is included in the PCC profile. Developer ID signing is not used by this development packager.")
    return matches[0][0]


def validate_device(profile, hardware):
    # Apple Silicon uses the provisioning UDID, which differs from IOPlatformUUID.
    device_id = hardware.get("provisioning_UDID") or hardware.get("platform_UUID", "")
    if not device_id or device_id.upper() not in {value.upper() for value in profile["ProvisionedDevices"]}:
        raise ValueError("This Mac is not included in the provisioning profile.")


def embedded_info(binary):
    # AFM targets Apple Silicon. otool prints little-endian 32-bit words for
    # the arm64 __info_plist section; compare the actual linked identity too.
    output = run("otool", "-arch", "arm64", "-X", "-s", "__TEXT", "__info_plist", str(binary)).decode()
    data = bytearray()
    for line in output.splitlines():
        words = line.split()
        if words and re.fullmatch(r"[0-9a-fA-F]{16}", words[0]):
            for word in words[1:]:
                if re.fullmatch(r"[0-9a-fA-F]{8}", word):
                    data.extend(bytes.fromhex(word)[::-1])
                elif re.fullmatch(r"[0-9a-fA-F]{2}", word):
                    data.extend(bytes.fromhex(word))
    try:
        return plistlib.loads(bytes(data).rstrip(b"\0"))
    except (plistlib.InvalidFileException, ExpatError) as error:
        raise ValueError("Could not read the arm64 binary's embedded Info.plist; use an AFM build from this source tree.") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / ".build/release/afm")
    parser.add_argument("--profile", type=pathlib.Path, required=True)
    parser.add_argument("--identity", required=True, help="Apple Development identity name or SHA-1")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / ".build/pcc/AFM.app")
    parser.add_argument("--check-only", action="store_true", help="Validate provisioning, identity, device, and build without packaging")
    args = parser.parse_args()

    info = plistlib.loads((ROOT / "Sources/AFMCLI/Info.plist").read_bytes())
    profile = plistlib.loads(run("security", "cms", "-D", "-i", str(args.profile)))
    entitlements = validate_profile(profile, info["CFBundleIdentifier"])
    identities = run("security", "find-identity", "-v", "-p", "codesigning").decode()
    identity = select_identity(profile, args.identity, identities)
    hardware = json.loads(run("system_profiler", "SPHardwareDataType", "-json"))["SPHardwareDataType"][0]
    validate_device(profile, hardware)
    binary = args.binary.resolve(strict=True)
    linked_info = embedded_info(binary)
    if linked_info.get("CFBundleIdentifier") != info["CFBundleIdentifier"]:
        raise ValueError("Binary's embedded bundle identifier does not match this AFM source tree.")
    products = binary.parent
    mlx_bundle = products / "AFMKit_AFMKitMLX.bundle"
    metallibs = list(mlx_bundle.rglob("default.metallib"))
    if not metallibs:
        raise ValueError("Build products are missing AFMKit_AFMKitMLX.bundle/default.metallib. Build with Scripts/swiftpm-reliable.sh first.")
    if args.output.exists():
        raise ValueError("Output already exists; select a new --output path. Existing signed apps are never overwritten.")
    if args.output.suffix != ".app":
        raise ValueError("--output must end in .app")
    print(f"Validated PCC profile for {info['CFBundleIdentifier']} on this Mac.")
    if args.check_only:
        return

    contents = args.output / "Contents"
    executable_dir = contents / "MacOS"
    resources = contents / "Resources"
    executable_dir.mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(binary, executable_dir / "afm")
    shutil.copy2(args.profile, contents / "embedded.provisionprofile")
    for bundle in products.glob("*.bundle"):
        shutil.copytree(bundle, resources / bundle.name, symlinks=True)
        # AFMKit's relocatable resource locators also search next to the CLI.
        (executable_dir / bundle.name).symlink_to(pathlib.Path("../Resources") / bundle.name)
    for dylib in products.glob("*.dylib"):
        shutil.copy2(dylib, executable_dir / dylib.name)
    # MLX's C++ loader also searches beside the executable.
    shutil.copy2(metallibs[0], executable_dir / "default.metallib")
    info.update(CFBundlePackageType="APPL", CFBundleVersion="1", LSMinimumSystemVersion="27.0", LSUIElement=True)
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    # Keep a reviewable copy of the exact claims inside the sealed resources.
    entitlement_path = resources / "AFM.entitlements"
    entitlement_path.write_bytes(plistlib.dumps(entitlements))

    # Sign nested Mach-O code first, without applying main-executable entitlements
    # to libraries. Do not use codesign --deep for signing.
    for path in sorted(contents.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_file() or path.is_symlink() or path == executable_dir / "afm":
            continue
        with path.open("rb") as stream:
            is_macho = stream.read(4) in MACHO_MAGIC
        if is_macho:
            run("codesign", "--force", "--sign", identity, str(path))
    run("codesign", "--force", "--sign", identity, "--entitlements", str(entitlement_path), str(args.output))
    run("codesign", "--verify", "--deep", "--strict", str(args.output))
    actual = plistlib.loads(run("codesign", "--display", "--entitlements", "-", "--xml", str(args.output)))
    if any(actual.get(key) != value for key, value in entitlements.items()):
        raise ValueError("Signed app's entitlements differ from the validated claims.")
    print(f"Signed development app: {args.output}")
    print(f"Run: '{args.output}/Contents/MacOS/afm' pcc status --json")
    print("No PCC request was sent. This is a local development bundle, not a notarized distribution.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, plistlib.InvalidFileException) as error:
        sys.exit(str(error))
    except subprocess.CalledProcessError as error:
        sys.exit(f"{error.cmd[0]} failed: {error.stderr.decode(errors='replace').strip()}")
