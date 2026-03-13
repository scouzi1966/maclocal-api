# Feature: codex-optimize-tooling

## Enforced JSON Tool Calling

- The default MLX tool-calling path now uses `response_format.type = "json_schema"` when `tools` are present and no explicit legacy `--tool-call-parser` override is set.
- This is not a new response format. `json_schema` already existed for guided JSON responses; this branch reuses it to constrain tool calls.
- The server builds an internal tool-call schema shaped like:
  - `{ "name": "<tool>", "arguments": { ... } }`
  - or an array of those objects for multi-call responses.
- Schema enforcement has two layers:
  - the outer tool-call wrapper schema is assembled by the server
  - the inner `arguments` schema is derived from the client-provided `tools[].function.parameters` schema
- This means required fields, arrays, nested objects, enums, and similar constraints are enforced from the client tool schema, but wrapped in a server-defined `{name, arguments}` contract.

## XGrammar Behavior

- Yes, the enforced JSON path guides logits with xgrammar.
- The path goes through the existing native JSON-schema grammar setup, not the legacy XML EBNF tool grammar.
- In code, the enforced plan sets `responseFormat` to `json_schema`, which triggers `setupGrammarConstraint(...)`.
- Legacy XML grammar remains available only when a parser override is set, especially `--tool-call-parser afm_adaptive_xml --enable-grammar-constraints`.

## Failure Model

- Enforced JSON mode is fail-fast at output validation time.
- Invalid JSON, unknown tool names, non-object `arguments`, and schema-mismatched arguments are rejected instead of heuristically repaired.
- This means the new default path does not depend on coercion/fallback parsing to recover malformed tool bodies.

## Legacy Compatibility

- If `--tool-call-parser` is explicitly set, the service keeps the old model-native parser path.
- That legacy path still includes fallback/coercion behavior such as XML parsing and JSON-in-XML recovery.
- This split is intentional: strict correctness by default, compatibility only when explicitly requested.

## Mixed XML and JSON

- In the default enforced path, mixed XML/JSON output is not repaired.
- When `tools` are present and no `--tool-call-parser` is set, the model is constrained to emit JSON matching the server-built schema.
- If the model emits XML, hybrid XML/JSON, or any other non-JSON payload anyway, the server fails the tool-call response instead of trying to recover it.
- Mixed XML/JSON recovery remains available only in the legacy parser path, especially `--tool-call-parser afm_adaptive_xml`.
- That legacy path still contains fallback handling for:
  - XML function blocks
  - JSON embedded inside `<tool_call>...</tool_call>`
  - bare JSON tool calls
  - some malformed hybrid XML/JSON openers

## Observed Qwen 3.5 Behavior

- In live checks against `mlx-community/Qwen3.5-35B-A3B-4bit`, the default enforced path produced JSON-only tool calls.
- Non-streaming returned `finish_reason: "tool_calls"` with `get_weather` and arguments `{"location":"Paris"}`.
- Streaming returned `finish_reason: "tool_calls"` with `add_tags` and arguments `{"item_id":"TASK-456","tags":["review","pending"]}`.
- In those runs, Qwen 3.5 respected the enforced JSON schema and did not drift into XML or mixed XML/JSON output.
- This does not mean malformed output is impossible; it means the tested cases were clean, and malformed output should now be rejected rather than repaired in the default path.

## Verification Notes

- `swift test` passed after the branch changes.
- Live Qwen 3.5 checks confirmed:
  - enforced JSON tool calls work in non-streaming and streaming modes
  - legacy `afm_adaptive_xml + --enable-grammar-constraints` still works for compatibility testing

## Invocation

- No new flags were added.
- The main change is behavioral:
  - If a request includes `tools` and no `--tool-call-parser` override is set, the MLX path now uses enforced JSON-schema tool calling by default.
  - That path constrains generation with xgrammar `json_schema` and fail-fast validates the result.
  - `--tool-call-parser ...` still forces the old legacy parser/template path.
  - `--enable-grammar-constraints` still only applies with `--tool-call-parser afm_adaptive_xml`.
- The default invocation is now just:

```bash
afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit --port 9999
```

- Then call the API with tools normally:

```bash
curl http://127.0.0.1:9999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

- Use legacy mode only if the old XML/coercion behavior is specifically desired:

```bash
afm mlx -m mlx-community/Qwen3.5-35B-A3B-4bit \
  --tool-call-parser afm_adaptive_xml \
  --enable-grammar-constraints
```

- In short:
  - New default path: no new flag, just omit `--tool-call-parser`
  - Old compatibility path: explicitly set `--tool-call-parser ...` as before
