# Security Policy

## Supported Versions

VulcanBench is currently in active development. Security fixes are applied to the `main` branch.

| Version | Supported |
|---------|-----------|
| main    | ✅        |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately via GitHub's built-in mechanism:
[Report a vulnerability](https://github.com/morganlinton/VulcanBench/security/advisories/new)

Include as much detail as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The affected component (harness, backend API, dashboard, sandbox, etc.)
- Any suggested mitigation

You can expect an acknowledgement within **72 hours** and a status update within **7 days**.

## Scope

Areas of particular interest:

- **Docker sandbox escape** — the sandbox runs untrusted agent-generated commands in a non-root, network-off container. Any path that allows host escape is critical.
- **Path traversal** — the backend API resolves run IDs against `RUNS_DIR`; bypasses are high severity.
- **Secret leakage** — run artifacts are redacted before being written to disk; gaps in `harness/redaction.py` are high severity.
- **Dependency vulnerabilities** — please report known CVEs in pinned dependencies.

Out of scope: issues in intentionally-buggy task fixture code under `tasks/v1/*/repo/`.

## Disclosure Policy

Once a fix is merged and released, we will publish a GitHub Security Advisory crediting the reporter (unless they prefer to remain anonymous).
