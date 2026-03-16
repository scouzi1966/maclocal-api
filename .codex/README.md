# Repo-Local Codex

This directory contains shared Codex project assets that are safe to version.

What belongs here:
- repo-specific skills and workflows
- reusable prompt templates
- shared conventions for this codebase

What must never be committed here:
- personal memories or chat history
- machine-specific state, caches, or logs
- tokens, credentials, cookies, or auth files
- absolute paths tied to one workstation
- personal names, emails, usernames, or hostnames unless they are already public project identifiers

Portability rules:
- prefer repo-relative paths
- use placeholders such as `$REPO_ROOT`, `$MODEL_ID`, `$MODEL_CACHE`, and `$AFM_BIN`
- avoid hardcoded local cache locations or user home paths

Ignored local subpaths are defined in the repo `.gitignore`, including:
- `.codex/cache/`
- `.codex/local/`
- `.codex/logs/`
- `.codex/memories/`
- `.codex/state/`
- `.codex/tmp/`

Current shared skills live under `.codex/skills/`.
