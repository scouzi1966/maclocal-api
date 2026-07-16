# Metrics (v1)

VulcanBench scores each run on five metrics plus a weighted **total**. Any metric
may be `null` with a `reason` when its analyzer or judge is unavailable — scores
are never fabricated.

## functional

Hidden verifier tests (`fail_to_pass` / `pass_to_pass` in task metadata) run
after the agent finishes. Score is `1.0` when all required tests pass, else
proportional to pass rate.

## quality

Static analysis over changed files:

- **Python**: ruff lint + radon complexity/maintainability
- **Rust**: `cargo fmt` + `cargo clippy`
- **Go / TypeScript / Java**: toolchain-dependent; `null` when tools absent

## security

Static security analysis:

- **Python**: bandit
- **Rust**: `cargo audit` + unsafe-delta penalty
- **Go**: gosec (when installed)
- **JS/TS**: npm audit (when applicable)

## efficiency

Derived from token usage and agent steps (lower is better, normalized to 0–1).

## human_like

3-judge LLM ensemble (on by default, reusing the run model unless
`--judge-model` is set). Use `--no-judges` for functional-only runs.

## total

Weighted combination of the five metrics (see `harness/evaluator/scorer.py`).
Functional failures dominate; ancillary metrics refine ranking among passes.

## cost

Per-run USD estimate from the built-in pricing table (`VULCANBENCH_PRICING` to
override). Unknown models report `cost_usd: null`.
