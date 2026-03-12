"""CLI wrapper for the afm binary (nightly)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_binary_path() -> Path:
    """Return the path to the afm binary (bundled, Homebrew, or PATH)."""
    # 1. Bundled in pip package
    bundled = Path(__file__).parent / "bin" / "afm"
    if bundled.exists():
        return bundled

    # 2. Homebrew nightly
    brew_next = Path("/opt/homebrew/bin/afm-next")
    if brew_next.exists():
        return brew_next

    # 3. Homebrew stable or system PATH
    found = shutil.which("afm")
    if found:
        return Path(found)

    raise FileNotFoundError(
        "afm binary not found.\n"
        "Install via Homebrew:\n"
        "  brew tap scouzi1966/afm\n"
        "  brew install scouzi1966/afm/afm-next\n"
        "\n"
        "Or download from: https://github.com/scouzi1966/maclocal-api/releases"
    )


def main():
    """Run the afm binary, forwarding all arguments and signals."""
    try:
        binary = get_binary_path()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.access(binary, os.X_OK):
        os.chmod(binary, 0o755)

    try:
        result = subprocess.run([str(binary)] + sys.argv[1:])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
