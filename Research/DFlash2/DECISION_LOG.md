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

Decision: DFlash 2 remains disabled by default. Explicit startup enablement
requires a resolved, compatible auxiliary checkpoint and errors otherwise.

Reason: correctness and Apple-Silicon performance are not yet established, and
silent startup fallback hides deployment mistakes.

## D004: AFMKitCore stays algorithm-neutral

Decision: only provider-neutral speculative policy, capability, fallback, and
telemetry concepts may enter AFMKitCore. DFlash model/config/kernel details stay
in AFMKitMLX; server orchestration stays outside AFMKitCore.

Reason: AFMKitCore is dependency-free and shared across providers. DFlash 2 is
an MLX implementation detail, not a universal provider contract.

## D005: Use existing metadata event before adding an enum case

Decision: initially transport versioned speculative status/telemetry through
`AFMGenerationEvent.metadata` and response metadata. Propose dedicated public
types for the next AFMKit API revision, but defer a new event enum case.

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

## D010: Inaccessible reference dependency is not copied or guessed

Decision: use the public article, configs, tensor headers, oMLX integration, and
original paper as contracts. Do not recreate undocumented behavior from the
unavailable `z-lab/dflash-mlx` pin.

Reason: both the final and intermediate DFlash MLX repositories returned 404.
The oMLX release is still useful behavioral evidence, but not auditable source
for a production port.

