#!/usr/bin/env python3
"""Validate a Prometheus text exposition with the official Python parser."""

from pathlib import Path
import sys

from prometheus_client.parser import text_string_to_metric_families


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: prometheus-exposition-validator.py METRICS.prom", file=sys.stderr)
        return 2

    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    families = list(text_string_to_metric_families(text))
    if not families:
        print("Prometheus parser returned no metric families", file=sys.stderr)
        return 1
    if not any(family.name.startswith("vllm:") for family in families):
        print("Prometheus parser found no vllm: metric families", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
