# Qwen Next candidate profile

The opt-in `--qwen-mtp-profile throughput-v2` selects the previously measured
corrected-HC recipe without requiring tuning environment variables. It adds
quantized HC fusion, sparse verification attention and verifier submission
interval 2 to `throughput-v1`. Existing defaults and v1 are unchanged.

Candidate native-community command (not yet release-qualified):

```sh
afm mlx -m mlx-community/Qwen3.8-Flash-Next-4bit-mtp --vlm \
  --mtp --mtp-depth 3 --qwen-mtp-profile throughput-v2 \
  --prefill-step-size 8192 --concurrent 2 -w
```

MTP remains explicit. Depth 3 is the native-community tested recipe, not an
automatic choice for other checkpoints. Legacy individual overrides still take
precedence; unset them to reproduce this profile. Batched reductions can change
greedy wording relative to strict verification. The profile is not a guarantee
of identical answers, universal speedup, or full quality parity.

The external harness enables `--vlm` for indexed checkpoints containing both
vision configuration and present vision weights. llmprobe keeps reasoning
available. Context performance tests disable reasoning separately.

Reports and candidate gate status are preserved outside Git under
`/Volumes/edata/afm-release-artifacts/nightly-qualification-20261004`.
