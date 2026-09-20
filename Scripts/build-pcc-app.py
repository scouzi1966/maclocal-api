#!/usr/bin/env python3
"""Build and sign AFM for local PCC development without running inference."""
import argparse
import datetime
import getpass
import pathlib
import platform
import subprocess
import sys
import tempfile
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINIMUM_MACOS = (27, 0)


def local_prompt(message):
    # getpass otherwise falls back to echoed input when terminal control fails.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass(message)
        except getpass.GetPassWarning as error:
            raise ValueError("A private terminal prompt is unavailable; use --profile and --identity locally with shell tracing disabled.") from error


def require_platform():
    version = platform.mac_ver()[0]
    components = tuple(int(part) for part in version.split(".")[:2]) if version else ()
    if platform.system() != "Darwin" or platform.machine() != "arm64" or components < MINIMUM_MACOS:
        raise ValueError("PCC development builds require macOS 27 or newer on Apple Silicon, with Xcode 27 / Swift 6.4 selected.")


def built_binary(scratch):
    # Resolve symlinks and refuse ambiguous output instead of signing an older
    # binary from another driver or the checkout's shared .build directory.
    candidates = {
        path.resolve() for path in (
            scratch / "out/Products/Release/afm",
            scratch / "arm64-apple-macosx/release/afm",
            scratch / "release/afm",
        ) if path.is_file()
    }
    if len(candidates) != 1:
        raise ValueError("The isolated build did not produce one unambiguous afm executable.")
    return candidates.pop()


def build_and_sign(args):
    require_platform()
    profile_value = args.profile or local_prompt("Absolute path to local PCC development profile (hidden): ")
    profile = pathlib.Path(profile_value).expanduser().resolve()
    if not profile_value or not profile.is_file():
        raise ValueError("Supply an existing local development provisioning profile.")
    identity = args.identity or local_prompt("Local Apple Development certificate SHA-1 (hidden): ")
    if not identity.strip():
        raise ValueError("An Apple Development signing identity is required.")
    output = args.output.expanduser().absolute()
    if output.suffix != ".app" or output.exists():
        raise ValueError("Choose a new --output path ending in .app; existing apps are never overwritten.")

    if args.binary is not None:
        binary = args.binary.expanduser().resolve(strict=True)
        if not binary.is_file():
            raise ValueError("--binary must name an existing AFM executable.")
    else:
        subprocess.run([str(ROOT / "Scripts/check-afmkit-consumer-boundary.sh")], cwd=ROOT, check=True)
        work = ROOT / ".build-pcc"
        work.mkdir(exist_ok=True)
        # Keep all build products for signing/resources; do not clean or reuse a
        # concurrent developer's build directory.
        scratch = pathlib.Path(tempfile.mkdtemp(prefix="build-", dir=work))
        subprocess.run([
            str(ROOT / "Scripts/swiftpm-reliable.sh"), "build", "-c", "release",
            "--product", "afm", "--scratch-path", str(scratch),
        ], cwd=ROOT, check=True)
        binary = built_binary(scratch)

    package_command = [
        sys.executable, str(ROOT / "Scripts/package-pcc-app.py"),
        "--binary", str(binary), "--profile", str(profile),
        "--identity", identity, "--output", str(output),
    ]
    # The existing packager is the single authority for App ID, PCC entitlement,
    # profile expiry, certificate matching, Mac UDID, and signature validation.
    subprocess.run(package_command + ["--check-only"], cwd=ROOT, check=True)
    subprocess.run(package_command, cwd=ROOT, check=True)
    print("PCC development app built and signed. No status check or model request was run.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Local profile path; omitted prompts without echo")
    parser.add_argument("--identity", help="Local Apple Development name or SHA-1; omitted prompts without echo")
    parser.add_argument("--binary", type=pathlib.Path, help="Sign this existing AFM binary instead of building")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path.home() / "Library/Developer/AFM-PCC" / f"AFM-{timestamp}.app",
                        help="New .app output path; default is a timestamped app under ~/Library/Developer/AFM-PCC")
    args = parser.parse_args(argv)
    try:
        build_and_sign(args)
    except (ValueError, OSError, EOFError) as error:
        parser.exit(1, f"PCC build: {error}\n")
    except subprocess.CalledProcessError as error:
        # CalledProcessError's default string includes the entire command,
        # including the identity/profile. Never print that representation.
        parser.exit(error.returncode if error.returncode > 0 else 1,
                    "PCC build/signing step failed. Review its local diagnostics; do not publish raw credential or profile output.\n")


if __name__ == "__main__":
    main()
