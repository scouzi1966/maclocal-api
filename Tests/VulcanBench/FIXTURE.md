# VulcanBench Fixture

This directory vendors the VulcanBench harness used for AFM MLX parity testing.

- Upstream: https://github.com/morganlinton/VulcanBench
- Source commit: `2ee08f5f0cf98d9b1cb4b103e6a340794e191bee`
- Copied from: `/private/tmp/VulcanBench`

Generated run outputs, virtual environments, caches, CI metadata, deployment files, and dashboard/backend service files are intentionally excluded. Keep benchmark result directories outside this fixture.

Basic smoke check:

```bash
cd Tests/VulcanBench
uv run python -m harness.cli --help
```
