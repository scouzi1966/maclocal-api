"""Sandbox image selection for Docker runs and validation."""

from __future__ import annotations

import os

from harness.sandbox.docker_executor import DEFAULT_IMAGE
from harness.tasks import Task

DEFAULT_RUST_IMAGE = os.environ.get("VULCANBENCH_SANDBOX_RUST_IMAGE", "vulcanbench/sandbox:rust")


def resolve_sandbox_image(task: Task, cli_image: str | None = None) -> str:
    """Pick the Docker image for a task.

    Order: explicit CLI ``--image`` → ``metadata.image`` → Rust language tier → base.
    """
    if cli_image:
        return cli_image
    meta_image = task.metadata.get("image")
    if isinstance(meta_image, str) and meta_image.strip():
        return meta_image.strip()
    langs = task.metadata.get("languages", [])
    if isinstance(langs, list) and "rust" in langs:
        return DEFAULT_RUST_IMAGE
    return DEFAULT_IMAGE
