# DFlash 2 Test Matrix

## Compile and Unit Gate

Run before any large model inference:

1. Patch drift/application checks from a clean pinned vendor revision.
2. `swift build` for AFMKitCore, AFMKitMLX, AFMKit, AFMServer, and AFMCLI.
3. Focused DFlash descriptor/config/compatibility/tensor tests.
4. Existing speculative, MTP, EAGLE3, DSpARK, Qwen 3.8, Muse, streaming,
   cancellation, prefix-cache, and batch tests.
5. OpenAI request decode and CLI help-json contract tests.

No long Qwen or Muse generation is authorized by this gate. Report readiness
and coordinate with the AFMKit usability study first.

Current gate result (updated 2026-08-21):

- `Scripts/swiftpm-reliable.sh build --target AFMKitMLX`: pass;
- `Scripts/swiftpm-reliable.sh build --target AFMCLI`: pass;
- `Scripts/swiftpm-reliable.sh test --filter AFMMLXDFlash2ConfigurationTests`:
  23 pass;
- `Scripts/swiftpm-reliable.sh test --filter BatchCompletionsControllerTests`:
  17 pass;
- `Scripts/swiftpm-reliable.sh test --filter TokenizeAndOpenAPITests`:
  11 pass;
- `Scripts/swiftpm-reliable.sh test --filter AFMMLXSpeculativeDecodingTests`:
  22 pass;
- focused Release selection across DFlash2 configuration, batch routing, and
  streaming cancellation: 58 XCTest cases pass with zero failures;
- final focused selection across DFlash2 config, speculative policy/setup,
  model architecture, AFMKit adapter/provider, stream translation/controller,
  OpenAPI/metrics, and batch routing: 143 XCTest plus 78 Swift Testing cases
  pass with zero failures;
- `Scripts/check-dflash2-vendor-patch.sh`: pass;
- full patch application plus `--check` from a clean clone of pinned vendor
  `6bab4f5ac55e81903dd74090244c25feb3233338`: pass with status identical to the
  working materialized submodule;
- official Qwen/Muse config snapshots and prior 81-tensor safetensor headers
  were validated without downloading weight payloads;
- official Qwen and Muse target/drafter pairs: bounded greedy live smoke pass
  for request `off`, required non-streaming, required streaming with usage, and
  nonzero speculative counters, run serially under the shared lock;
- review-remediation Qwen production-path qualification: omitted temperature
  and non-neutral penalties use AR for preferred and reject required requests;
  explicit greedy required generation reports normal token/timing telemetry;
  reasoning remains separated from visible content; partial SSE cancellation
  preserves observed output, emits `cancelled` without stop/usage, stops
  speculative work, and leaves the server healthy;
- Qwen prefix-cache qualification: preferred uses AR, required rejects with
  `prefix_cache`, and speculative cycles remain zero;
- Muse batch qualification: two preferred rows complete through AR, a required
  row emits `batch.error` with reason `batch`, and speculative cycles remain
  zero;
- complete Release attempts run the 437-case Swift Testing portion cleanly but
  the XCTest process terminates with signal 11 in the unrelated macOS 27
  Foundation Models request-adapter test under Xcode 27 beta 3; focused DFlash2
  Release coverage is clean and the crash report is retained;
- no AI judge, full live matrix, or local performance run was made.

## Fixture Matrix

| Area | Cases | Expected result |
| --- | --- | --- |
| Architecture detection | DFlash 2 exact architecture; legacy DFlash; MTP; DSpark; arbitrary `*-DFlash2` directory with normal config | Metadata determines type; name-only case is rejected |
| Required config | Missing selector rank/top-k, conv kernel/group, block size, mask, feature taps | Fails before weight allocation with field-specific error |
| Shape validation | Selector codebooks, projection, target feature fusion, conv tensors, layer count | Exact expected/actual shape diagnostic |
| Target pairing | Qwen/Qwen draft; Muse/Muse draft; Qwen/Muse cross-pair; changed vocab/layers/hidden/rope/token IDs | Valid pairs pass; mismatches fail deterministically |
| Backward compatibility | No drafter, MTP, EAGLE3, DSpARK, legacy DFlash descriptor fixture | Existing selection remains unchanged |
| Startup conflicts | DFlash2+MTP, DFlash2+EAGLE3, DFlash2+DwarfStar/DSpark, invalid block limit, missing draft | Actionable startup error; no implicit mode choice |
| Request controls | omitted/nil temperature, disabled, preferred, required, sampling modifiers, penalties, wrong strategy, invalid draft limit | Normal defaults preserved; stable AR fallback or pre-emission errors |
| Fallback timing | unavailable before output; runtime error before output; runtime error after output | Preferred may pre-output fallback only; no replay after output |
| Telemetry | zero cycles, partial acceptance, full acceptance, cancellation, fallback, base timing/usage | Counts and timing definitions remain consistent; cancellation commits no partial success |

Implemented fixture coverage includes exact released Qwen/Muse metadata,
name-only rejection, full metadata/tensor preflight, target shape mismatch,
provider-neutral startup mapping, independent service/model ownership, OpenAI
request decoding, nested selector weight keys, greedy target equivalence,
complete EOS handling, cancellation propagation, nonstreaming reasoning
boundaries, omitted-temperature and penalty policy, prefix/concurrency/batch
fallback reasons, base/speculative telemetry, OpenAPI schema, and existing
speculative policy regression. Runtime-error-after-emission and broader feature
combinations remain matrix work before expanding supported modes.

## Live Target Matrix

Targets:

- `mlx-community/Qwen3.8-27B-4bit` with
  `incoai/Qwen3.8-27B-DFlash2`
- `mlx-community/Muse-Glimmer-30B-4bit` with
  `incoai/Muse-Glimmer-30B-DFlash2`

For every row below, run DFlash 2 off and on. Use the same target snapshot,
prompt bytes, template kwargs, generation settings, maximum tokens, cache state,
and concurrency. Record revisions and hardware.

| Dimension | Cases |
| --- | --- |
| Transport | non-streaming; streaming with usage; streaming client cancellation |
| Sampling | greedy; seeded model-recommended temperature/top-p/top-k if lossless sampler is implemented |
| Reasoning | disabled/lowest; Qwen `xhigh`; Muse `high`; explicit request override |
| Tools | no tools; one tool; parallel tools; multi-turn tool result; malformed partial call cancellation |
| Prefix cache | disabled; enabled miss; exact repeat; shared-prefix extension |
| Scheduling | serial; two concurrent requests; configured batch limit; mixed DFlash2 off/on requests |
| Drafter state | valid local; valid Hub download with progress; missing; incomplete; Qwen/Muse swapped; corrupt tensor shape |
| Runtime control | startup off; startup on/request omitted; request disabled; preferred; required |
| Completion | EOS; maximum tokens; stop sequence; cancellation |

Current expected routing:

- Greedy, serial, text-only, no tools/grammar/logprobs/string stops: DFlash 2.
- Sampling, tools, grammar, logprobs, or string stops: preferred AR fallback;
  required error before emission.
- Prefix cache or `--concurrent`: preferred mode uses AR and emits a neutral
  fallback reason; required mode rejects before generation.
- Reasoning remains downstream token parsing. Qwen prompt-opened reasoning and
  the bounded Muse reasoning/content comparison pass; broader reasoning levels
  remain part of the exhaustive matrix.

## Output Equivalence

Greedy correctness:

- Compare target token IDs, not only decoded text.
- DFlash 2 and autoregressive output must be identical through EOS or the same
  maximum-token boundary.
- Compare extracted reasoning text, visible text, tool calls/arguments, finish
  reason, and usage semantics after parser translation.

Sampling correctness:

- A single seeded sample is a regression aid, not proof of distributional
  equivalence.
- First require the same seed to be reproducible within each mode.
- Run a fixed prompt/sample suite and compare token-frequency distributions with
  a declared statistical test and tolerance. Preserve raw outputs.
- Do not call sampling lossless until rejection-sampling math and empirical
  distribution checks both pass.

## Performance Methodology

1. Record Mac model, SoC, RAM, macOS, power mode, thermal state, Swift/MLX
   revisions, target/draft revisions, quantization, and command.
2. Use one target checkpoint and one prompt set for AR and DFlash 2.
3. Separate cold download/load, model warmup, prompt processing, draft,
   verification, commit, and end-to-end measurements.
4. Run at least three unmeasured warmups and enough measured repetitions to
   report median, p10/p90, and raw samples.
5. Report output tok/s, TTFT, inter-token latency, acceptance length, draft
   tokens, accepted draft tokens, cycles, phase time, peak memory, and errors.
6. Test concurrency 1 first. Higher concurrency comparisons must use the same
   scheduler behavior; an AR-batched result cannot be labeled DFlash 2.
7. A speedup claim is allowed only from same-machine, same-model evidence in
   this matrix. Upstream H200/oMLX claims remain citations only.
