#!/usr/bin/env python3
"""Black-box regression: invalid profile must not destroy an existing capture.

Run after building the release binary, with no concurrent performance workload:
  python3 Scripts/tests/test_qwen_profile_capture_cli.py --binary /path/to/afm \
      --artifacts /Volumes/edata/afm-release-artifacts/capture-regression
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["AFM_QWEN_MTP_PROFILE"] = "invalid-profile-regression-fixture"
    with tempfile.TemporaryDirectory(prefix="capture-profile-", dir=args.artifacts) as directory:
        capture = Path(directory) / "existing.gputrace"
        original = b"existing user GPU capture\n"
        capture.write_bytes(original)
        result = subprocess.run(
            [str(args.binary.resolve()), "mlx", "-m", "unused/model",
             "--gpu-capture", str(capture), "-s", "unused"],
            env=environment, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, "Invalid profile unexpectedly succeeded"
        assert "Unknown Qwen MTP profile" in result.stdout + result.stderr, result
        assert capture.exists(), "Invalid profile deleted the existing capture"
        assert capture.read_bytes() == original, "Invalid profile changed the existing capture"
    print("PASS: invalid profile preserves the existing GPU capture before model loading")


if __name__ == "__main__":
    main()
