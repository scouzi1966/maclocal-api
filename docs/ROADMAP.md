# AFM Roadmap

## Streaming Tool Call Arguments (Incremental `delta.tool_calls`)

**Status:** Planned
**Priority:** Medium
**Area:** Tool Calling / OpenAI Compatibility

### Current Behavior

afm buffers the entire tool call body until `</tool_call>` is received, runs post-processing (cross-parameter dedup, type coercion, key remapping), then emits the clean JSON arguments as a single `delta.tool_calls` chunk.

### Desired Behavior

Stream `function.arguments` incrementally token-by-token, matching OpenAI's behavior:

```
data: {"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"lo"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cati"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"on\":"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \"Paris"}}]}}
data: {"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}}
data: {"delta":{},"finish_reason":"tool_calls"}
```

### Why It Matters

- Clients that show progressive tool call construction (arguments "typed out") see them appear all at once instead of incrementally
- Better OpenAI API spec compliance
- Most clients (OpenCode, Vercel AI SDK) concatenate fragments and work fine either way, but strict spec compliance improves ecosystem compatibility

### Implementation Notes

- **Simple case:** Stream each `</parameter>` as a JSON key-value fragment as it closes (no buffering needed)
- **Complex case:** When cross-parameter dedup or key remapping is needed, must fall back to buffered mode
- Hybrid approach: stream params incrementally by default, buffer only when the model emits duplicate/conflicting parameters (detected heuristically)
- The XML→JSON translation is the core challenge — can't stream JSON fragments until the parameter name and value are known
- Grammar-constrained mode (`--enable-grammar-constraints`) could simplify this since the output is already well-formed
