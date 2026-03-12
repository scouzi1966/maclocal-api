# AFM CLI and Modes Reference

## Scope

Use this file when changing command-line flags, defaults, mode selection, or single-prompt behavior.

## Command Entrypoints

- Root command and compatibility flow:
  - `Sources/MacLocalAPI/main.swift` (`RootCommand`, `ServeCommand`)
- MLX subcommand:
  - `Sources/MacLocalAPI/main.swift` (`MlxCommand`)
- Vision subcommand:
  - `Sources/MacLocalAPI/VisionCommand.swift`

## Mode Selection Behavior

- If first arg is `mlx`, parse and run `MlxCommand`.
- If first arg is `vision`, parse and run `VisionCommand` (async bridge).
- Otherwise parse `RootCommand`:
  - `-s/--single-prompt` -> one-shot generation path.
  - piped stdin -> one-shot generation path.
  - else start `ServeCommand`.

This behavior is intentional for backward compatibility and to avoid flag parsing collisions.

## Serve Mode (`afm`) Highlights

Important flags:

- `--port`, `--hostname`
- `--instructions`
- `--temperature` (0.0-1.0)
- `--randomness` parser-backed mode string
- `--stop` (comma-separated)
- `--no-streaming`
- `-w/--webui`
- `-g/--gateway`
- `--prewarm` (`y` by default)

Validation:

- temperature range validation.
- randomness parsed via `RandomnessConfig.parse`.

## MLX Mode (`afm mlx`) Highlights

### Complete `afm mlx` Option Catalog

Source of truth: `Sources/MacLocalAPI/main.swift` (`struct MlxCommand`).

Core:

- `-m`, `--model <string>`
  - model id (`org/model` or short name; defaults org to `mlx-community`).
- `-s`, `--single-prompt <string>`
  - one-shot prompt mode (no server).
- `-i`, `--instructions <string>`
  - default: `"You are a helpful assistant"`.
- `-p`, `--port <int>`
  - server port (default behavior prefers `9999`, then ephemeral fallback if busy).
- `-H`, `--hostname <string>`
  - default: `127.0.0.1`.
- `-v`, `--verbose`
- `-V`, `--very-verbose`
- `--no-streaming`
- `--raw`
  - return raw model output; disables `<think>` extraction to `reasoning_content`.
- `-w`, `--webui`
- `-g`, `--gateway`
  - rejected in mlx mode; command exits with failure.

Sampling and generation:

- `-t`, `--temperature <double>`
  - MLX mode help states 0.0-2.0.
- `--top-p <double>`
- `--top-k <int>`
- `--min-p <double>`
- `--presence-penalty <double>`
- `--repetition-penalty <double>`
- `--max-tokens <int>`
  - default intent: 8192 when not provided.
- `--seed <int>`
- `--max-logprobs <int>`
  - server-side upper bound for request `top_logprobs`.
- `--stop <csv>`
  - comma-separated stop sequences.

KV/prefill and performance:

- `--kv-bits <int>`
  - supported values documented as 4 or 8.
- `--prefill-step-size <int>`
  - default intent: 2048.
- `--enable-prefix-caching` / `--no-enable-prefix-caching`
  - toggles automatic prompt prefix cache reuse.

Tool calling and structured output:

- `--guided-json <schema-or-path>`
  - constrain response to JSON schema.
- `--tool-call-parser <name>`
  - values: `hermes`, `llama3_json`, `gemma`, `mistral`, `qwen3_xml`.
- `--fix-tool-args`
  - remap generated arg keys back to provided schema names.

VLM / media:

- `--vlm`
  - force vision-language mode.
- `--media <path...>`
  - media paths for single-prompt VLM mode; implies VLM.
  - runtime constraint: requires `-s`; command errors if used without single-prompt.

Integration:

- `--openclaw-config`
  - prints provider config JSON and exits.

Compatibility switches (accepted, currently ignored, warning emitted):

- `--max-kv-size <int>`
- `--trust-remote-code`
- `--chat-template <string>`
- `--dtype <string>`

### MLX Runtime Rules

- If `--model` omitted and stdin is TTY:
  - run interactive local model picker.
- If `--model` omitted and stdin is piped:
  - fail with model guidance (or registered model listing).
- Piped stdin takes precedence as single-prompt input.
- If both `stdin` and `-s` are absent:
  - command starts server mode.

## Vision Mode (`afm vision`) Highlights

File path processing:

- Expands `~`
- standardizes relative paths

Outputs:

- plain OCR text
- verbose OCR details (`--verbose`)
- table extraction CSV (`--table`)
- hidden debug path (`-D`)

## Environment Variables

- `MACAFM_MLX_MODEL_CACHE`: MLX cache root override.
- `MACAFM_MLX_METALLIB`: explicit metallib path override.
- `LOG_LEVEL`: logging level for server runtime.

`ensureMLXMetalLibraryAvailable` resolution order:

1. `MACAFM_MLX_METALLIB`
2. `Bundle.module` metallib
3. `MacLocalAPI_MacLocalAPI.bundle/default.metallib` adjacent to executable
