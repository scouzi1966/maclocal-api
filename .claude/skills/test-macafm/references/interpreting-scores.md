# Interpreting Smart Analysis Scores

Guide for understanding AI-judge (Claude/Codex) scoring in `mlx-model-test.sh --smart` reports.

## Scoring Dimensions

The AI judges score each test on multiple dimensions (typically 1-10):

| Dimension | What it measures |
|-----------|-----------------|
| **Relevance** | Does the output address the prompt? |
| **Correctness** | Is the output factually/logically correct? |
| **Completeness** | Does it cover all requested aspects? |
| **Format** | Does it follow requested format (JSON, bullets, etc.)? |
| **Quality** | Overall coherence, readability, usefulness |

## Known False Positive Patterns

These are cases where the AI judge scores LOW but the server behavior is CORRECT:

### 1. Stop Sequence Truncation Scored as Low Quality
**Pattern**: Judge gives relevance=2, completeness=1 because output is "incomplete."
**Reality**: Stop sequence correctly truncated the output. Truncation IS the expected behavior.
**How to identify**: Test has `stop:` parameter and output ends abruptly before the stop string.
**Action**: Ignore low relevance/completeness scores. Check only that the stop string is absent.

### 2. Empty Content from Stop on First Token
**Pattern**: Judge says "empty response, server broken."
**Reality**: Stop sequence fired on the very first visible token (e.g., `stop: ["\n"]` and model's first output is a newline).
**How to identify**: Content is empty but finish_reason is "stop" (not "error").
**Action**: This is correct behavior. Verify finish_reason.

### 3. JSON Mode Not Constraining Thinking Models
**Pattern**: Judge says "output is not valid JSON" for a thinking model.
**Reality**: AFM uses prompt injection for json_object mode, not grammar-constrained decoding. Thinking models may emit reasoning before JSON.
**How to identify**: Content starts with thinking/reasoning text before the JSON.
**Action**: Known limitation. The thinking content is in reasoning_content; the actual content field should be JSON. If content field itself isn't JSON, that's a real bug.

### 4. Missing Reasoning for Non-Thinking Models
**Pattern**: Judge says "no reasoning_content, think extraction broken."
**Reality**: Model doesn't support `<think>` tags (e.g., Qwen3.5-35B-A3B, Llama models).
**How to identify**: Check the model's chat template for `<think>` or `enable_thinking`.
**Action**: Expected behavior. Not a bug.

### 5. Cache Timing Not Faster
**Pattern**: Judge says "cache-hit not faster than cache-warmup."
**Reality**: First request may be fast enough that cache speedup is within noise. Or prompt is too short for meaningful cache savings (< 32 tokens cached).
**How to identify**: Both requests complete in < 1s, timing difference < 100ms.
**Action**: Check `cached_tokens` in usage response instead of timing. If cached_tokens > 0, cache is working.

### 6. Streaming Parity Slight Differences
**Pattern**: Judge says "streaming output differs from non-streaming."
**Reality**: Minor whitespace or formatting differences can occur due to think extraction buffer boundaries. Content meaning should be identical.
**How to identify**: Diff shows only whitespace/newline differences.
**Action**: Compare semantic content, not exact bytes. Significant content differences are real bugs.

## Score Thresholds

| Average Score | Interpretation |
|---------------|----------------|
| 8-10 | Excellent — feature working perfectly |
| 6-7 | Good — minor issues, likely cosmetic |
| 4-5 | Investigate — may be a real issue or false positive |
| 1-3 | Likely failure — check against false positive patterns above first |

## Workflow for Low Scores

1. **Check false positive patterns** above — most low scores are expected truncation or model limitations
2. **Read the AI's explanation** — judges usually explain WHY they scored low
3. **Compare with assertion tests** — if `test-assertions.sh` passes but smart analysis scores low, it's almost certainly a false positive
4. **Check the `# AI:` comments** in the test file — they describe expected behavior
5. **Only file a bug** if the behavior contradicts both the AI comments AND the assertion tests

## Model-Specific Scoring Notes

### Qwen3.5-35B-A3B-4bit
- No thinking support — all think tests will show "no reasoning_content" (correct)
- Tool call format is xmlFunction — watch for XML parsing edge cases in scores
- Very fast model — cache timing improvements may be too small to measure

### Qwen3-Coder-Next
- Has tool calling but with XML format quirks (duplicate parameters, JSON-in-string)
- No thinking support despite being a "Coder" model
- Tool call scores may be lower due to argument format issues (known, handled by `--fix-tool-args`)

### Models with `<think>` Support (DeepSeek R1, etc.)
- Think tests should all score high
- Watch for stop+think interaction bugs — stop should NOT truncate reasoning
- Long reasoning chains may hit buffer boundaries — check for truncated reasoning_content
