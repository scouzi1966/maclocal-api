#!/usr/bin/env python3
"""Test AFM's external CLI process contract without installing Splash or loading a model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 15
FIXTURE_EXIT = 17
SIGNAL_EXIT = 42
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / ".build/release/afm")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    staging = ROOT / ".build-splash-cli-tests"
    staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as directory:
        root = Path(directory)
        fixture = root / "native splash fixture"
        fixture.write_text(f"""#!{sys.executable}
import json, os, signal, sys
if sys.argv[1:] == ['wait-for-signal']:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit({SIGNAL_EXIT}))
    print('ready', flush=True)
    signal.pause()
else:
    print(json.dumps(dict(args=sys.argv[1:], pid=os.getpid(), cwd=os.getcwd(),
        marker=os.environ.get('AFM_SPLASH_TEST_MARKER'), stdin=sys.stdin.read())))
    print('native-stderr', file=sys.stderr)
    sys.exit({FIXTURE_EXIT})
""")
        fixture.chmod(0o755)
        env = dict(os.environ, AFM_SPLASH_EXECUTABLE=str(fixture), AFM_SPLASH_TEST_MARKER="inherited")
        forwarded = ["serve", "--model", "owner/model with spaces", "--future-flag", "$(touch unwanted)", "--", "", "--help", "--version"]
        process = subprocess.Popen([str(binary), "splash", *forwarded], cwd=root, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate("stdin forwarded\n", timeout=TIMEOUT_SECONDS)
        payload = json.loads(stdout)
        assert process.returncode == FIXTURE_EXIT, (process.returncode, stderr)
        assert payload == dict(args=forwarded, pid=process.pid, cwd=str(root), marker="inherited", stdin="stdin forwarded\n"), payload
        assert stderr == "native-stderr\n", stderr
        assert not (root / "unwanted").exists()

        # Native flags must reach Splash even when they are AFM's own help/version flags.
        for flags in [[], ["--help"], ["--version"]]:
            result = subprocess.run([str(binary), "splash", *flags], cwd=root, env=env, input="", capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            assert result.returncode == FIXTURE_EXIT and json.loads(result.stdout)["args"] == flags, result

        # A signal sent to the original AFM PID reaches the native CLI directly.
        process = subprocess.Popen([str(binary), "splash", "wait-for-signal"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            import select
            readable, _, _ = select.select([process.stdout], [], [], TIMEOUT_SECONDS)
            assert readable and process.stdout.readline().strip() == "ready"
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=TIMEOUT_SECONDS)
            assert process.returncode == SIGNAL_EXIT, process.returncode
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

        for override, expected in [(str(root / "missing"), "not an executable file"), (str(binary), "AFM itself")]:
            result = subprocess.run([str(binary), "splash", "serve"], env=dict(env, AFM_SPLASH_EXECUTABLE=override), capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            assert result.returncode != 0 and expected in result.stderr, result
        env.pop("AFM_SPLASH_EXECUTABLE")
        env["PATH"] = str(root)
        result = subprocess.run([str(binary), "splash", "--help"], env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        assert result.returncode != 0 and "brew install incoai/tap/splash" in result.stderr, result
    print("PASS: argv, stdio, environment, cwd, PID, native help/version, exit status, signals, missing install, and recursion")


if __name__ == "__main__":
    main()
