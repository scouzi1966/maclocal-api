# DFlash 2 Decision Log

## D001: DFlash 2 is a new runtime primitive

Decision: implement a distinct DFlash 2 draft/verify primitive in AFMKitMLX or
the supported MLX vendor patch set.

Reason: this checkout has MTP, EAGLE3, and DSpARK but no DFlash. Their model,
cache, and verification contracts differ. Reusing only orchestration avoids
misrepresenting one algorithm as another.

## D002: Detection is metadata-first

Decision: recognize DFlash 2 from `architectures`, required `dflash_config`
fields, and tensor/config validation. Never activate from repository or
directory names.

Reason: both released drafters declare `model_type=qwen3`, while their targets
declare different top-level/text model types. Names are especially unreliable
for imported or renamed checkpoints.

## D003: Opt-in and fail-closed startup

Decision: DFlash 2 remains disabled by default. Explicit preferred enablement
falls back to AR before emission with a diagnostic; `required` fails startup or
the request when the checkpoint/runtime/request is incompatible.

Reason: correctness and Apple-Silicon performance are not yet established, and
silent startup fallback hides deployment mistakes.

## D004: AFMKitCore stays algorithm-neutral

Decision: only provider-neutral speculative policy, capability, fallback, and
telemetry concepts may enter AFMKitCore. DFlash model/config/kernel details stay
in AFMKitMLX; server orchestration stays outside AFMKitCore.

Reason: AFMKitCore is dependency-free and shared across providers. DFlash 2 is
an MLX implementation detail, not a universal provider contract.

## D005: Keep telemetry structured and neutral before adding an event case

Decision: use a neutral AFMKitMLX telemetry value and neutral StatsAggregator /
Prometheus counters. Propose dedicated AFMKitCore public types for the next API
revision, but defer a new generation-event enum case.

Reason: adding an enum case can break exhaustive downstream switches. Versioned
metadata is already stable and lets the implementation prove the shape first.

## D006: No fallback after output

Decision: preferred mode may fall back only before the first output token/event.
After emission, runtime failure terminates the request.

Reason: replaying AR generation after partial speculative output can duplicate
or diverge content, tool-call state, and reasoning channels.

## D007: Prefix cache and batch are gated

Decision: use AR fallback for preferred requests and deterministic rejection for
required DFlash 2 when prefix restore or concurrent/batch scheduling is active,
until complete speculative state and row-aligned verification are implemented.

Reason: current caches and batch scheduler are AR-oriented. oMLX also documents
its DFlash path as single-stream and keeps a separate cache containing draft and
target state.

## D008: Checkpoint block and draft-token counts remain distinct

Decision: parse checkpoint `block_size` as model metadata and expose a clearly
named runtime maximum draft-token setting. Validation derives allowed values
from the implementation contract rather than equating the two fields.

Reason: official examples use inconsistent engine-facing quantities (Qwen
block 8, seven vLLM draft tokens, SGLang argument 8, oMLX runtime block 5).

## D009: No unmeasured performance claim

Decision: expose neutral counts/timings and retain raw benchmark evidence. Do
not advertise a speedup until the same-target live matrix passes.

Reason: upstream results use H200/SGLang or a different oMLX implementation;
they do not establish maclocal-api performance.

## D010: Audit the signed release artifact when the public pin is unavailable

Decision: hash-verify and inspect the signed oMLX release DMG read-only. Use its
bundled Apache-2.0 Python sources as the reference contract; do not alter or
publish changes to upstream oMLX/MLX repositories.

Reason: both repository pins returned 404, but the official release contains
auditable source and its SHA-256 matches GitHub release metadata.

## D011: Start with correctness-first restore/replay

Decision: snapshot target cache before verification, restore, and replay the
committed verifier token plus accepted prefix. Defer optimized partial commit,
draft KV caching, and prefix snapshots.

Reason: this is simple to audit for greedy token equivalence. It may cost more
than the signed reference and therefore cannot support a performance claim.

## D012: Greedy only until rejection sampling is verified

Decision: DFlash 2 runs only for greedy requests. Preferred sampled requests
fall back before emission; required sampled requests fail.

Reason: the signed reference implements selector sampling plus target rejection
sampling. Porting only selector sampling would not preserve the target
distribution.
