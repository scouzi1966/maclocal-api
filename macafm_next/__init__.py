"""
macafm-next - Nightly build of AFM for macOS

This is the nightly (development) version of the `afm` command-line tool.
For stable releases, use `pip install macafm` instead.

Requirements:
- macOS 26 (Tahoe) or later
- Apple Silicon Mac (M1/M2/M3/M4 series)
- Apple Intelligence enabled in System Settings
"""

__version__ = "0.9.7.dev20260316"
__author__ = "Sylvain Cousineau"

from .cli import main, get_binary_path

__all__ = ["main", "get_binary_path", "__version__"]
