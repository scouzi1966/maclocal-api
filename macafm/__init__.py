"""
macafm - Access Apple's on-device Foundation Models via CLI and OpenAI-compatible API

This package provides the `afm` command-line tool for macOS that allows you to:
- Access Apple's 3B parameter Foundation Model locally
- Run an OpenAI-compatible API server
- Use single-prompt mode with Unix pipes
- Load custom LoRA adapters

Requirements:
- macOS 26 (Tahoe) or later
- Apple Silicon Mac (M1/M2/M3/M4 series)
- Apple Intelligence enabled in System Settings
"""

__version__ = "0.9.4"
__author__ = "Sylvain Cousineau"

from .cli import main, get_binary_path

__all__ = ["main", "get_binary_path", "__version__"]
