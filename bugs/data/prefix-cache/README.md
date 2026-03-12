# KVCache Offset Fix (Issue #46)

## Status: NOT REPRODUCIBLE (2026-03-10)

The theoretical crash (`Cannot reshape array of size N into shape (1,1,32,128)`) from
QuantizedKVCache.state setter not restoring `offset` was identified through static code
analysis but could NOT be reproduced in testing.

## Testing Done

- Qwen3-Coder-Next-4bit + `--kv-bits 4 --enable-prefix-caching` on port 9999
- Sent identical requests to trigger prefix cache restore with QuantizedKVCache
- Cache correctly restored 525+ token prefixes (prompt_n dropped from 548 to 320)
- No crash with OR without the fix
- Tested both fixed and unfixed KVCache.swift — identical behavior

## The Fix (saved but NOT applied)

`kvcache-offset-fix.patch` adds offset restoration to:
- `QuantizedKVCache.state` setter (line 963): `self.offset = keys.0.dim(2)`
- `RotatingKVCache.state` setter (line 649): `self.offset = self.keys!.dim(2)`

The fix is logically correct (offset SHOULD match stored key length) but the crash
it was designed to prevent doesn't occur in practice. Possible reasons:
- The model's `prepare()` chunking resets offset through another path
- The trim logic handles the offset=0 case gracefully
- The crash requires specific conditions we haven't reproduced

## Related Issues

- GitHub issue #46: https://github.com/scouzi1966/maclocal-api/issues/46
- Prefix cache token divergence for Qwen3.5-35B-A3B-4bit (separate bug, unrelated)
