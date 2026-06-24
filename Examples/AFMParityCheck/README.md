# AFMParityCheck — differential parity harness

Proves that **AFMKit-direct** inference produces the **same output as the `afm` HTTP server**.
This is the executable proof behind PR #136's claim that the library *is* the server: a consumer
embedding AFMKit (e.g. [vesta-mac](../../docs/vesta-integration.md)) gets byte-for-byte the same
results it would get by running `afm` and calling `/v1/chat/completions`.

## What it does

For each case it runs the request twice — once in-process via `AFMEngine`/the public services,
once over HTTP against a freshly-spawned `afm mlx … --port 9998` — at **temperature 0**, and
asserts the two agree:

| # | Case | Asserts |
|---|------|---------|
| 1 | greedy chat text | `direct.content == server message.content` |
| 2 | streaming determinism | `direct.stream == direct.full == server.stream == server.full` |
| 3 | logprobs | same token sequence; values within `1e-3` |
| 4 | structured JSON (`response_format`) | both parse and canonicalize to the same object |
| 5 | tool call | same function name; arguments canonicalize equal |

Exit code `0` = full parity; non-zero = at least one mismatch (details printed per case).

## Running

Build the `afm` binary first (`swift build -c release` in the repo root), then:

```bash
cd Examples/AFMParityCheck
MACAFM_MLX_MODEL_CACHE=/path/to/cache \
AFM_BINARY=../../.build/release/afm \
AFM_PARITY_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit \
swift run AFMParityCheck
```

### Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `AFM_PARITY_MODEL` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | model used by both paths |
| `AFM_BINARY` | repo `.build/release/afm` (then `…/arm64-apple-macosx/release`, then `debug`) | server binary to spawn |
| `AFM_PARITY_PORT` | `9998` | server port |
| `MACAFM_MLX_MODEL_CACHE` | — | weight cache, shared by both paths (set it to avoid re-downloads) |

## Notes

- Uses **only AFMKit + Foundation `URLSession`** — no Vapor in this package's graph, the same
  constraint a sandboxed consumer lives under.
- Cases 3/5 report a failure if *neither* side produces logprobs / a tool call — that means the
  chosen model/build doesn't exercise the feature, so pick a tool-capable model to test case 5.
- Determinism depends on temperature 0 greedy decoding on both paths; a model with
  nondeterministic kernels at a given context length would surface here as a case-2 failure.
