# Large-Task Handbook (OSS-Heavy)

This guide explains how to add **medium** and **large** tasks that stress navigation in
realistic codebases—not single-file toy puzzles.

## Scale tiers (`repo_scale`)

| Tier | Code LOC (approx.) | Typical files | Storage |
|------|-------------------|---------------|---------|
| `micro` | &lt; 500 | 1–3 | `repo/` |
| `small` | 500–1k | 3–10 | `repo/` |
| `medium` | 1k–10k | 10–80 | `repo/` or `repo_snapshot.tar.gz` |
| `large` | 10k–50k | 80+ | `repo_snapshot.tar.gz` (Git LFS) |

Set `repo_scale` in `metadata.json`. The harness applies higher default `max_steps`
and wall-clock budgets for medium/large tasks (overridable via `agent_hints` and
`test_timeout_s`). Also set `task_complexity` so effort-sensitivity reports can
separate localized fixes from multi-file, system, and architecture tasks.

## Selecting upstream issues (OSS-heavy)

Prioritize issues/PRs where:

1. The symptom spans **2+ modules** (imports, config, shared helpers).
2. The fix is **localized** (gold patch usually &lt; 500 lines) but discovery is hard.
3. Upstream already has tests—you run **targeted** `fail_to_pass` / `pass_to_pass`, not full CI.
4. License is permissive (MIT/BSD/Apache); **LICENSE** is preserved in the slice.
5. Verifier finishes in **&lt; 2 minutes** on CI.

Prefer issues **after model training cutoffs** → `decontaminated: true`. Otherwise use
`decontaminated: false` and document upstream URL + commit/issue in `decontamination_notes`
and `upstream`.

## Tooling workflow

```bash
# 1. Slice a repo at a pinned commit (paths only you need)
python scripts/slice_repo.py \
  --url https://github.com/org/repo.git \
  --commit abc123 \
  --paths src/pkg,tests \
  --output /tmp/slice

# 2. Scaffold the VulcanBench task directory
python scripts/import_oss_issues.py \
  --id oss-my-task \
  --repo /tmp/slice \
  --issue issue.md \
  --languages python \
  --base-commit abc123 \
  --upstream-url https://github.com/org/repo \
  --upstream-issue https://github.com/org/repo/issues/42 \
  --repo-scale medium \
  --output snapshot   # when slice is large

# 3. Human steps: trim, hidden tests/, gold_patch.diff, validate
make validate-tasks
vulcanbench run --task oss-my-task --model mock:synthetic
```

### Rust workspace slicing

Cargo workspaces need special handling: the full workspace must resolve even
when only a subset of crates is needed. Use ``--cargo-prune`` to compute the
transitive dependency closure of target crates, drop the rest, and rewrite
``Cargo.toml`` ``[workspace] members`` so the slice compiles:

```bash
# Slice only the crates needed and their workspace dependencies
python scripts/slice_repo.py \
  --url https://github.com/org/rust-workspace.git \
  --commit abc123 \
  --cargo-prune --crates target-crate,helper-crate \
  --output /tmp/rust-slice

# Then scaffold as usual
python scripts/import_oss_issues.py \
  --id oss-rust-my-task \
  --repo /tmp/rust-slice \
  --issue issue.md \
  --languages rust \
  --base-commit abc123 \
  --upstream-url https://github.com/org/rust-workspace \
  --upstream-issue https://github.com/org/rust-workspace/issues/42 \
  --repo-scale medium \
  --output snapshot
```

Rust tasks should also declare a ``setup`` warm-up so cold compiles do not eat
the agent/verifier time budget:

```json
{
  "setup": [{"name": "build-tests", "cmd": "cargo build --tests"}],
  "setup_timeout_s": 600,
  "test_timeout_s": 300
}
```

## Extended metadata (OSS)

```json
{
  "repo_scale": "medium",
  "task_complexity": "system",
  "base_commit": "abc123...",
  "upstream": {
    "url": "https://github.com/org/repo",
    "issue": "https://github.com/org/repo/issues/42",
    "pr": "https://github.com/org/repo/pull/99",
    "fix_commit": "def456..."
  },
  "test_timeout_s": 180,
  "setup": [{"name": "build-tests", "cmd": "cargo build --tests"}],
  "setup_timeout_s": 600,
  "agent_hints": {
    "suggested_max_steps": 100,
    "suggested_timeout_s": 1200,
    "entry_paths": ["src/pkg/core.py", "README.md"]
  }
}
```

## Curation checklist

1. Trim slice to minimal reproducible tree (keep LICENSE, package manifests).
2. `issue.md` = real issue text; navigation hints only in `agent_hints.entry_paths`.
3. Hidden `tests/` with narrow verifier commands.
4. `gold_patch.diff` generated mechanically from the fix commit.
5. `python scripts/validate_tasks.py tasks/v1/<id>` passes (gold solves, fail-to-pass real, 3× deterministic).
6. Pilot run: reject tasks solved in &lt; 5 steps on a strong model unless intentionally easy.

## Suites

- `vulcanbench run --suite v1` — full corpus
- `vulcanbench run --suite v1-micro` — fast micro/small tasks (CI-friendly)
- `vulcanbench run --suite v1-large` — medium/large navigation tasks (nightly)

Manifests: [`tasks/v1/suite.json`](../tasks/v1/suite.json) and named suite aliases in
[`harness/suite.py`](../harness/suite.py).

## Storage & LFS

Use `repo_snapshot.tar.gz` for medium/large slices. Track with Git LFS (see root
`.gitattributes`). Validator fails snapshots &gt; 100MB uncompressed unless
`allow_large_snapshot: true`.

## Calibration

After adding a batch, run:

```bash
vulcanbench run --suite v1-large --model openai:gpt-4o --repeat 5 --sandbox docker
```

Record pass@1 in `calibration.json` (optional). Reject tasks with 0% or 100% pass@1 on
a reference model unless they serve as smoke/regression tasks.
