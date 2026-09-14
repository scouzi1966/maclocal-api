#!/usr/bin/env python3
"""Exercise PCC command parsing and graceful failures without cloud generation.

Run against an ordinary (non-PCC-entitled) afm binary on macOS 26 or 27.
"""
import argparse
import json
import platform
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", default=".build/release/afm")
args = parser.parse_args()
checks = 0


def check(arguments, expected, success=False, stdin=None):
    global checks
    result = subprocess.run([args.binary, *arguments], input=stdin, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == success, (arguments, result.returncode, result.stdout, result.stderr)
    assert expected in result.stdout + result.stderr, (arguments, result.stdout, result.stderr)
    checks += 1
    return result


check(["pcc", "--help"], "Private Cloud Compute", success=True)
for command in ["status", "respond", "chat", "serve"]:
    check(["pcc", command, "--help"], "USAGE:", success=True)
check(["pcc", "respond", "hello", "--reasoning", "invalid"], "reasoning")
if int(platform.mac_ver()[0].split(".")[0]) < 27:
    for command in [["status"], ["respond", "hello"], ["chat"], ["serve"]]:
        check(["pcc", *command], "Private Cloud Compute requires macOS 27 or later")
else:
    check(["pcc", "serve", "--port", "65536"], "Port must be")
    status = check(["pcc", "status", "--json"], "missingEntitlement")
    decoded = json.loads(status.stdout)
    assert decoded["available"] is False and decoded["hasEntitlement"] is False
    check(["pcc"], "missing")
    check(["pcc", "respond", "hello"], "PCC unavailable (missingEntitlement)")
    check(["pcc", "serve"], "PCC unavailable (missingEntitlement)")
    check(["pcc", "chat"], "interactive terminal")
    check(["pcc", "respond", "-"], "Prompt must not be empty", stdin=" \n")
print(f"PCC CLI: {checks} checks passed; no inference requests sent.")
