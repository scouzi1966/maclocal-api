# Shared Skills

These skills are the repo-local Codex equivalents of the portable workflows that were previously kept under `.claude/skills/`.

Canonical source:
- `.claude/skills/` is the source of truth when an equivalent skill exists.
- `.codex/skills/` should prefer symlinks into `.claude/skills/` rather than duplicate copies.
- If there is no `.claude` equivalent yet, the `.codex` version can remain standalone temporarily.

Only shared engineering workflows belong here. Release-owner procedures, personal publishing instructions, and environment-specific defaults stay out.

Current skills:
- `codex-build-afm` - full AFM build workflow
- `codex-test-afm` - regression and validation workflow
- `codex-add-afm-model` - investigate and onboard a new MLX model
- `codex-promptfoo-agentic-eval` - run and review the Promptfoo-based AFM agentic evaluation suite
