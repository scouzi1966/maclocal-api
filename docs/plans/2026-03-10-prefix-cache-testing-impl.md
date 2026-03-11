# Prefix Cache Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate prefix cache correctness and performance with enhanced diagnostic logging, deterministic A/B test harness, and unit tests for both Qwen3.5-35B-A3B-4bit and Qwen3-Coder-Next-4bit.

**Architecture:** Enhanced `-vv` logging in RadixTreeCache and MLXModelService to self-diagnose misses (divergence point, code flow). Standalone `test-prefix-cache.sh` script runs workloads with cache ON vs OFF, compares `prompt_n`/`prompt_ms` savings. Integrated into `test-assertions.sh` as Section 14. Unit tests cover RadixTreeCache data structure in isolation.

**Tech Stack:** Bash, Python (report generation), Swift (unit tests), curl (HTTP requests)

---

### Task 1: Enhanced Logging in RadixTreeCache.findPrefix

**Files:**
- Modify: `Sources/MacLocalAPI/Models/RadixTreeCache.swift:57-105`

**Step 1: Add divergence-point logging to findPrefix**

Replace the current `findPrefix` method to log WHERE tokens diverge on miss or partial match:

```swift
func findPrefix(_ tokens: [Int]) -> (prefixLen: Int, layerStates: [[MLXArray]]?) {
    var node = root
    var matched = 0
    var lastCachedNode: RadixNode? = nil
    var lastCachedLen = 0

    while matched < tokens.count {
        let nextToken = tokens[matched]
        guard let child = node.children[nextToken] else {
            // No child for this token — log divergence
            if debugLogging && matched > 0 {
                // Find what tokens ARE available as children
                let available = Array(node.children.keys.prefix(3))
                print("[PrefixCache] Radix traversal: matched \(matched) tokens, no child for token \(nextToken) at pos \(matched) (available: \(available))")
            }
            break
        }

        let edge = child.edgeTokens
        var edgePos = 0
        while edgePos < edge.count && matched < tokens.count {
            if tokens[matched] != edge[edgePos] {
                // Edge mismatch — log exact divergence
                if debugLogging {
                    print("[PrefixCache] Radix traversal: matched \(matched) tokens, diverged at pos \(matched): input=\(tokens[matched]) vs cached=\(edge[edgePos])")
                }
                break
            }
            edgePos += 1
            matched += 1
        }

        if child.hasCachedState && matched > lastCachedLen {
            lastCachedNode = child
            lastCachedLen = matched
        }

        if edgePos < edge.count { break }
        node = child
    }

    if let cached = lastCachedNode {
        cached.cacheEntry?.touch()
        if debugLogging {
            let entryLen = cached.cacheEntry?.tokens.count ?? 0
            print("[PrefixCache] Radix hit: \(lastCachedLen)/\(tokens.count) tokens matched (entry has \(entryLen) tokens)")
        }
        return (lastCachedLen, cached.cacheEntry?.layerStates)
    }

    if debugLogging {
        if matched > 0 {
            print("[PrefixCache] Radix miss: traversed \(matched) tokens but no cached node found (\(tokens.count) input tokens)")
        } else {
            print("[PrefixCache] Radix miss for \(tokens.count) tokens (no prefix match)")
        }
    }
    return (0, nil)
}
```

**Step 2: Add logging to insert**

Add layer count and trim details to insert logging. In existing `insert` method, enhance the debug print:

```swift
// At each insert point, add:
if debugLogging {
    print("[PrefixCache] Radix insert: \(tokens.count) tokens, \(layerStates.count) layers (entries: \(entryCount))")
}
```

**Step 3: Verify build compiles**

Run: `swift build -c release 2>&1 | tail -3`
Expected: `Build complete!`

**Step 4: Commit**

```bash
git add Sources/MacLocalAPI/Models/RadixTreeCache.swift
git commit -m "feat: add divergence-point logging to RadixTreeCache.findPrefix"
```

---

### Task 2: Enhanced Logging in MLXModelService Cache Code Paths

**Files:**
- Modify: `Sources/MacLocalAPI/Models/MLXModelService.swift`

**Step 1: Enhance non-streaming cache path (lines ~411-460)**

Add code flow and restore detail logging. After the existing `[PrefixCache] First 20 tokens` log, add:

```swift
// After: let inputTokens = useCache ? self.extractTokenArray(input) : []
if debugLogging {
    let cacheState = self.radixCache != nil ? "active(entries:\(self.radixCache!.count))" : "nil"
    print("[PrefixCache] Path: non-streaming | useCache=\(useCache) | radixCache=\(cacheState)")
}
if useCache && inputTokens.count > 20 {
    print("[PrefixCache] Input: \(inputTokens.count) tokens, first20=\(Array(inputTokens.prefix(20)))")
}
```

After cache restore (inside `if effectivePrefix > 0, let states = layerStates`):
```swift
if debugLogging {
    let offsets = generationCache.prefix(3).map { "\($0.offset)" }.joined(separator: ",")
    print("[PrefixCache] Restore: effectivePrefix=\(effectivePrefix), layers=\(generationCache.count), offsets=[\(offsets),...]")
}
```

After cache insert (lines ~528-535):
```swift
if debugLogging {
    let offsets = generationCache.prefix(3).map { "\($0.offset)" }.joined(separator: ",")
    print("[PrefixCache] Insert: \(inputTokens.count) tokens, \(generationCache.count) layers, offsets=[\(offsets),...]")
}
```

Add skip/invalidation logging at each `radixCache?.invalidateAll()` call:
```swift
if debugLogging {
    print("[PrefixCache] Invalidate: generation error")
}
```

**Step 2: Enhance streaming cache path (lines ~705-875)**

Mirror the same logging as non-streaming. The streaming path has the same structure:
- After `extractTokenArray`: log path/input
- After restore: log effectivePrefix/offsets
- After insert: log token count/layers
- At each invalidateAll: log reason (error/cancelled/catch)

**Step 3: Remove the temporary `[PrefixCache] First 20 tokens` and `Total tokens` prints (lines 415-416, 713-714)**

These are replaced by the new structured logging above (gated by `debugLogging` consistently).

**Step 4: Verify build compiles**

Run: `swift build -c release 2>&1 | tail -3`
Expected: `Build complete!`

**Step 5: Commit**

```bash
git add Sources/MacLocalAPI/Models/MLXModelService.swift
git commit -m "feat: add code-flow and restore/insert detail logging to prefix cache paths"
```

---

### Task 3: Create test-prefix-cache.sh Script

**Files:**
- Create: `Scripts/test-prefix-cache.sh`

**Step 1: Write the test script**

The script must:

1. Accept `--model MODEL` (required), `--port PORT` (default 9877), `--bin BIN` (default `.build/release/afm`), `--workload W1|W2|W3|all` (default all)
2. For each workload:
   a. Start afm with `-vv --enable-prefix-caching --tool-call-parser afm_adaptive_xml` → log file
   b. Wait for server ready (poll `/v1/models`)
   c. Send workload requests, capture response JSON (with `timings`)
   d. Kill server, wait
   e. Start afm with `-vv` (NO prefix caching) → separate log
   f. Wait for server ready
   g. Send SAME workload requests
   h. Kill server, wait
3. Parse logs: extract `Radix hit/miss`, `First 20 tokens`, `diverged at pos` lines
4. Generate JSONL + summary to stdout

**Workload definitions (hardcoded):**

W1 — OpenCode session: Use the real OpenCode system prompt extracted from `/Volumes/edata/dev/logs/opencode/afm-35b-qwen3xml.log` (first `RECV MLX full request` with tools). 5 requests:
- Req 1: system + tools + "List files in the current directory"
- Req 2: same system + tools + user + assistant(tool_call) + tool_result + "Now read the README"
- Req 3: same + assistant(tool_call) + tool_result + "Summarize the project"
- Req 4: same + assistant(text) + "Create a hello world file"
- Req 5: same + assistant(tool_call) + tool_result + "Run the file"

W2 — Identical repeats: Same request sent 10 times (system + tools + "What is 2+2?")

W3 — Growing conversation: System prompt + tools, 10 requests each appending one user+assistant turn.

**Per-request capture:**
```bash
# Send request, capture full JSON
resp=$(curl -sf --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' -d "$BODY")

# Extract timings
prompt_n=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['timings']['prompt_n'])")
prompt_ms=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['timings']['prompt_ms'])")
```

**Log parsing (after server killed):**
```bash
# Extract cache hits from afm log
grep -a 'Radix hit:' "$LOG_FILE" | while read -r line; do
    cached=$(echo "$line" | grep -oP 'hit: \K[0-9]+')
    total=$(echo "$line" | grep -oP '/\K[0-9]+')
    echo "cached=$cached total=$total"
done

# Extract divergence points
grep -a 'diverged at pos' "$LOG_FILE"
```

**JSONL output per request:**
```json
{"model":"...","workload":"W2","req":2,"cache":"on","prompt_n":320,"prompt_ms":265.3,"cached_tokens":525,"diverge_pos":null}
{"model":"...","workload":"W2","req":2,"cache":"off","prompt_n":548,"prompt_ms":362.7,"cached_tokens":0,"diverge_pos":null}
```

**Summary output to stdout:**
```
=== Prefix Cache Test Results ===
Model: mlx-community/Qwen3-Coder-Next-4bit
Workload W2 (Identical Repeats, 10 requests):
  Cache ON:  avg prompt_n=320, avg prompt_ms=265
  Cache OFF: avg prompt_n=548, avg prompt_ms=363
  Savings:   228 tokens/req (41.6%), 98ms/req (27.0%)
  Cache hit rate: 9/10 requests (90%)
  ...
```

**Step 2: Make executable**

```bash
chmod +x Scripts/test-prefix-cache.sh
```

**Step 3: Test with a quick dry run**

Run: `./Scripts/test-prefix-cache.sh --model mlx-community/Qwen3-Coder-Next-4bit --workload W2 --port 9877`
Expected: Server starts, 10 requests sent twice (cache on/off), summary printed, no crash.

**Step 4: Commit**

```bash
git add Scripts/test-prefix-cache.sh
git commit -m "feat: add prefix cache A/B test harness with workload simulation"
```

---

### Task 4: Extract Real OpenCode System Prompt and Tools for W1

**Files:**
- Create: `Scripts/test-data/opencode-system-prompt.json`

**Step 1: Extract from archived log**

Parse the real OpenCode request from the March 9 log to get the system prompt and 11 tools exactly as OpenCode sends them:

```bash
python3 -c "
import json, re
with open('/Volumes/edata/dev/logs/opencode/afm-35b-qwen3xml.log', 'r') as f:
    content = f.read()
# Find second RECV (first is title gen, second is main session)
requests = re.findall(r'RECV MLX full request:\n(\{.*?\})\x1b', content, re.DOTALL)
req = json.loads(re.sub(r'\x1b\[[0-9;]*m', '', requests[1]))
with open('Scripts/test-data/opencode-system-prompt.json', 'w') as f:
    json.dump({
        'system': req['messages'][0]['content'],
        'tools': req['tools']
    }, f, indent=2)
print(f'Extracted: {len(req[\"messages\"][0][\"content\"])} chars system, {len(req[\"tools\"])} tools')
"
```

**Step 2: Verify extraction**

```bash
python3 -c "import json; d=json.load(open('Scripts/test-data/opencode-system-prompt.json')); print(f'System: {len(d[\"system\"])} chars, Tools: {len(d[\"tools\"])}')"
```
Expected: `System: 10092 chars, Tools: 11`

**Step 3: Commit**

```bash
git add Scripts/test-data/opencode-system-prompt.json
git commit -m "feat: extract real OpenCode system prompt and tools for prefix cache testing"
```

---

### Task 5: Add Section 14 to test-assertions.sh

**Files:**
- Modify: `Scripts/test-assertions.sh`

**Step 1: Add `--include-prefix-cache` flag**

In the argument parsing block (around line 30), add:
```bash
--include-prefix-cache) INCLUDE_PREFIX_CACHE=true; shift ;;
```

And default:
```bash
INCLUDE_PREFIX_CACHE=false
```

**Step 2: Add Section 14 at the end (before the report generation)**

Section 14 runs the prefix cache A/B comparison inline. It requires starting/stopping servers, so it's gated by the flag. Key assertions:

```bash
# Section 14: Prefix Cache Performance (requires --include-prefix-cache)
if [ "$INCLUDE_PREFIX_CACHE" = "true" ] && should_run_section 14; then
  CURRENT_TIER="standard"
  echo ""
  echo "📦 Section 14: Prefix Cache Performance (A/B comparison)"

  # --- W2: Identical repeats (5x) with cache ON ---
  # Start server with prefix caching
  # Send 5 identical requests, capture prompt_n from each
  # Kill server

  # --- W2: Same requests with cache OFF ---
  # Start server without prefix caching
  # Send same 5 requests, capture prompt_n from each
  # Kill server

  # Assertions:
  # 1. All requests succeed (no crash)
  # 2. Request 1 cache ON: prompt_n == prompt_n cache OFF (first always full)
  # 3. Requests 2-5 cache ON: prompt_n < prompt_n cache OFF
  # 4. Total tokens saved > 0
  # 5. avg(prompt_ms cache ON for req 2-5) < avg(prompt_ms cache OFF for req 2-5)
fi
```

The inline server management uses the same pattern as the existing test suite (`$BIN mlx -m $MODEL ...`).

**Step 3: Implement the full section**

The section starts/stops servers itself using `$BIN` and `$MODEL`. It sends 5 identical requests (a system prompt + user message + tools), captures `timings.prompt_n` and `timings.prompt_ms` from each response.

Assertions via `run_test`:
- `"Prefix cache: no crash with cache ON"` — all 5 succeed
- `"Prefix cache: no crash with cache OFF"` — all 5 succeed
- `"Prefix cache: tokens saved on req 2"` — `prompt_n_on < prompt_n_off`
- `"Prefix cache: tokens saved on req 3"` — same
- `"Prefix cache: avg prefill faster with cache"` — `avg_ms_on < avg_ms_off` for reqs 2-5
- `"Prefix cache: total savings > 20%"` — `(total_off - total_on) / total_off > 0.20`

**Step 4: Verify build and run smoke**

Run: `./Scripts/test-assertions.sh --tier standard --section 14 --include-prefix-cache --model mlx-community/Qwen3-Coder-Next-4bit`
Expected: Section 14 runs, all assertions pass for Qwen3-Coder-Next.

**Step 5: Commit**

```bash
git add Scripts/test-assertions.sh
git commit -m "feat: add Section 14 prefix cache A/B performance assertions"
```

---

### Task 6: RadixTreeCache Unit Tests

**Files:**
- Create: `Tests/MacLocalAPITests/RadixTreeCacheTests.swift` (or add to existing test file if test target exists)

If no Swift test target exists, add as assertion-based tests in `Scripts/test-assertions.sh` Section U (unit tier). The key is testing the data structure logic with realistic-sized arrays.

**Step 1: Check if Swift test target exists**

```bash
grep -r "testTarget" Package.swift
```

If yes, create Swift XCTest. If no, add to Section U in test-assertions.sh as a Swift script that imports the module and runs assertions.

**Step 2: Write unit tests**

Core test cases (whether Swift XCTest or script-based):

```
1. insert_and_find_exact      — insert 10K tokens, find same → hit 10K
2. prefix_match               — insert 10K, find 12K sharing 10K prefix → hit 10K
3. divergence_at_position     — insert 10K, find array diverging at pos 500 → hit 0
4. edge_split                 — two 10K arrays sharing 8K prefix → both findable
5. growing_conversation       — insert [sys], [sys+turn1], [sys+turn1+turn2] → find longest
6. identical_repeats          — insert same 10K array 5 times → no duplication, find hits
7. lru_eviction               — maxEntries=3, insert 4 → oldest evicted, others findable
8. invalidate_all             — insert 3 entries, invalidateAll → all miss
9. layer_state_integrity      — insert with known MLXArrays, verify retrieved arrays match
10. volume_stress             — 50 sequential inserts with growing arrays (1K to 50K tokens)
```

For realistic token arrays, generate deterministic sequences:
```swift
// Simulates OpenCode: 10K system prefix + variable suffix
let systemPrefix = Array(1...10000)  // fixed system+tools tokens
let turn1 = Array(10001...10500)     // user message
let turn2 = Array(10501...11000)     // assistant response
// etc.
```

**Step 3: Run unit tests**

```bash
swift test --filter RadixTreeCache
# or for script-based:
swift build -c release && swift Scripts/test-radix-unit.swift
```

**Step 4: Commit**

```bash
git add Tests/ # or Scripts/
git commit -m "test: add RadixTreeCache unit tests with realistic-sized token arrays"
```

---

### Task 7: Create Skill File

**Files:**
- Create: `.claude/skills/test-prefix-cache/SKILL.md`

**Step 1: Write the skill**

Document:
- How to run: standalone script and assertion suite integration
- Available workloads (W1, W2, W3)
- How to interpret results (token savings, cache hit rate, divergence logs)
- How to read the enhanced `-vv` logging
- Known issues (Qwen3.5-35B-A3B token divergence, expected low hit rate)
- How to add new workloads
- Port/model/bin configuration

**Step 2: Commit**

```bash
git add .claude/skills/test-prefix-cache/SKILL.md
git commit -m "docs: add prefix cache testing skill"
```

---

### Task 8: End-to-End Validation

**Step 1: Run full prefix cache test for Qwen3-Coder-Next-4bit**

```bash
./Scripts/test-prefix-cache.sh --model mlx-community/Qwen3-Coder-Next-4bit --workload all
```

Expected: Cache ON shows significant token savings (40%+ on W2 identical repeats), no crashes.

**Step 2: Run for Qwen3.5-35B-A3B-4bit**

```bash
./Scripts/test-prefix-cache.sh --model mlx-community/Qwen3.5-35B-A3B-4bit --workload all
```

Expected: Low cache hits (known broken), but enhanced logging shows exactly WHERE tokens diverge. This gives us the diagnostic data to fix the divergence in a follow-up task.

**Step 3: Run assertion suite Section 14**

```bash
./Scripts/test-assertions.sh --tier standard --section 14 --include-prefix-cache \
  --model mlx-community/Qwen3-Coder-Next-4bit
```

Expected: All assertions pass.

**Step 4: Archive results**

```bash
cp test-reports/prefix-cache-* /Volumes/edata/dev/logs/opencode/
```

**Step 5: Final commit**

```bash
git add test-reports/
git commit -m "test: prefix cache validation results for Qwen3-Coder-Next and Qwen3.5-35B-A3B"
```
