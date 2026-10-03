# Nightly candidate validation

Build candidates from current main and record the exact commit, AFMKit pin,
nightly identifier from `Scripts/nightly-version.sh`, binary SHA-256, and the
PRs included since the previous release. PCC is deferred and is not part of
main. Candidate preparation does not publish a release or update Homebrew.

Run the complete Release Swift tests through `Scripts/swiftpm-reliable.sh`,
then package the release binary with its runtime bundles and WebUI. Validate
that the relocated command reports the candidate version and exposes MLX and
on-device Foundation Models, with no PCC command.

Model qualification includes the full assertion tier, comprehensive prompts
with the explicitly requested `--smart 1:codex-glm` judge, all Promptfoo profiles,
and batch correctness/cache checks. Run GPU stages sequentially. Keep reports
outside the repository under `/Volumes/edata/afm-release-artifacts/`.

The external suites are first-class additional tests:

```bash
python3 Scripts/test-external-benchmarks.py \
  --binary /path/to/candidate/afm \
  --model /path/to/Qwen3.8-Flash-Next-AFM-MLX-4bit \
  --llmprobe /path/to/llmprobe/bin/dist/llmprobe.mjs \
  --context-harness /path/to/llm_context_benchmarks \
  --context-python /path/to/context-venv/bin/python \
  --output /Volumes/edata/afm-release-artifacts/candidate/external
```

This runs llmprobe full conformance/capability/agentic coverage without its
separate performance mode, then the Context performance sweep at 0.5K, 1K,
2K, 4K, 8K, 16K and 32K, with two trials and 128 output tokens. Context uses
cold-prefill mode, temperature zero, top-p one, and thinking disabled. MTP is
not requested. The runner records tool revisions, binary hash, commands,
server logs, reports, and saved responses. Tool checkouts should be clean;
retain their exact revisions for repeatability. These external suites have
their own upstream provenance and do not replace AFM's internal assertions.

A context process exiting zero is insufficient: all requested sizes and
trials, positive finite metrics, and saved responses must be present. This
checks completion, not a performance regression threshold or response quality.
A nonzero llmprobe result is retained as a failure for review, including any
unsupported surfaces and inconclusive cases reported by that tool.

Use `--phase llmprobe` or `--phase context` for a focused rerun into a new output
directory. The runner refuses an occupied port and never stops an existing
server. Its default port is 9999.

Offline harness regression tests:

```bash
python3 Scripts/tests/test_external_benchmarks.py
```

Report failed, skipped, inconclusive, and incomplete work explicitly. Do not
attribute a failure to the model or baseline without supporting evidence.
