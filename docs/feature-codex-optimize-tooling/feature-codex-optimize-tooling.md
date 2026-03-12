# Feature: codex-optimize-tooling

## Enforced JSON Tool Calling

- The default MLX tool-calling path now uses `response_format.type = "json_schema"` when `tools` are present and no explicit legacy `--tool-call-parser` override is set.
- This is not a new response format. `json_schema` already existed for guided JSON responses; this branch reuses it to constrain tool calls.
- The server builds an internal tool-call schema shaped like:
  - `{ "name": "<tool>", "arguments": { ... } }`
  - or an array of those objects for multi-call responses.

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

## Verification Notes

- `swift test` passed after the branch changes.
- Live Qwen 3.5 checks confirmed:
  - enforced JSON tool calls work in non-streaming and streaming modes
  - legacy `afm_adaptive_xml + --enable-grammar-constraints` still works for compatibility testing
