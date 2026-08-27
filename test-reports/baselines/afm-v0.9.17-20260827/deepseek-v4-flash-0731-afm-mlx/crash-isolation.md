# DeepSeek V4 Flash 0731 prefix-cache crash isolation

## Scope

- AFM: `v0.9.17` (`/opt/homebrew/bin/afm`)
- Model: `scouzi1966/DeepSeek-V4-Flash-0731-AFM-MLX`
- Thinking control: `--no-think`
- Failing comprehensive record: `agent-cached-turn2`
- Fatal error: `MLX/ErrorHandler.swift:345: Fatal error: [metal::malloc] Resource limit (499000) exceeded`

## Controlled results

The original two-case isolation used the same prompt, system string, temperature,
and `max_tokens: 20000`. Only prefix caching changed:

| Prefix caching | Result |
| --- | --- |
| Disabled | Completed normally in about 3 seconds with 49 output tokens |
| Enabled | Ran for about 9 minutes, then AFM terminated with the MLX Metal resource-limit fatal error |

The comprehensive harness starts a fresh AFM server for each record. Therefore,
the records named `agent-cached-turn1`, `agent-cached-turn2`, and
`agent-cached-turn3` do **not** reuse cache entries from one another. The failing
record begins with only AFM's prewarm cache entry.

Verbose direct testing showed that the failing request restores only a three-token
prefix from the 13-token prewarm entry:

```text
outcome=hit | input_tokens=218 | cached_tokens=3 | suffix_tokens=215
```

The serial prefix-cache classifier in AFM v0.9.17 treats only `ArraysCache` and
`CacheList` as recurrent. It does not include `DeepseekV4Cache`, even though the
batch scheduler explicitly classifies `DeepseekV4Cache` as recurrent state that
cannot be trimmed to an arbitrary token boundary. The three-token partial restore
can therefore reuse/truncate DeepSeek V4 hybrid state as if it were ordinary KV
state, changing deterministic generation.

## Harness interaction

The prompt parser preserves the two characters `\` and `n` in one-line `system:`
fields instead of decoding them to newline characters. This changes the model
input:

- With actual newlines, the cache-enabled request stops normally after 73 tokens.
- With the harness's literal `\\n` sequences, the cache-enabled request reaches a
  256-token test cap with `finish_reason: length` and continues similarly at larger
  caps.
- The same literal-escape request without prefix caching stops normally, so the
  malformed prompt is an exposing condition, not the complete engine cause.

At `max_tokens: 20000`, the divergent cache-enabled decode continues until MLX's
graph/resource ceiling is reached. The resource-limit fatal error is therefore the
terminal symptom; the earlier correctness divergence is the primary engine defect.

## Classification

1. **Engine/runtime likely:** unsafe partial prefix restore for DeepSeek V4 hybrid
   cache state in the serial generation path.
2. **Test harness:** escaped newlines are sent literally, and sequential cache tests
   restart the server so they do not test cross-turn reuse.
3. **Crash amplifier:** `max_tokens: 20000` permits the divergent decode to grow
   until MLX aborts at its 499,000-resource ceiling.

Both existing `v0.9.18` beta runs completed `agent-cached-turn2`, so the observed
fatal behavior is not present in those tested beta binaries. The serial recurrent
cache classification should still be corrected and regression-tested explicitly.

## Evidence

- `isolation-cache-off-on.jsonl`
- `isolation-server-logs/mlx-server-21-53943.log`
- `isolation-server-logs/mlx-server-24-53943.log`
- `comprehensive-results.jsonl`
