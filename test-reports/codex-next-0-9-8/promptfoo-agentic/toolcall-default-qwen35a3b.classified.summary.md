# AI Failure Classification

Source report: `test-reports/promptfoo-agentic/toolcall-default-qwen35a3b.json`

- failures classified: 4
- afm_bug: 0
- model_quality: 0
- harness_bug: 4

## Details

### tool_choice required selects weather tool
- provider: afm-default
- classification: harness_bug
- confidence: 0.95
- rationale: The AFM output is structurally valid and contains the correct tool call with the 'location' argument. The test failure is due to the assertion checking for the literal string '"location"' which does not appear as a standalone token in the normalized output (it is nested inside the JSON string of the arguments). This is a harness/assertion logic error, not a bug in the AFM runtime or model quality.
- evidence:
  - The normalized_output shows a valid tool call: {"function":{"name":"get_weather","arguments":"{\"location\":\"Berlin\"}"}}
  - The assertion failed because the string literal '"location"' was not found in the output, but the key 'location' exists inside the JSON string of the 'arguments' field.
  - The model correctly selected the tool and provided the correct argument structure, indicating the failure is in the assertion's string matching logic, not the model's output.

### typed args bool and int coercion
- provider: afm-default
- classification: harness_bug
- confidence: 0.95
- rationale: The AFM output is structurally valid and the model made the correct decision to call the appropriate tool with correct arguments. The test failure is due to the assertion logic failing to match the string '"query"' within the JSON arguments of the tool call, which is a harness/assertion issue rather than a model or AFM protocol failure.
- evidence:
  - The model output is structurally valid JSON with a correct tool call.
  - The tool call arguments are correctly parsed and contain the expected keys: 'case_sensitive', 'max_results', and 'query'.
  - The assertion failed because the test expected the string '"query"' to appear in the output, but the actual output contains the string '"query"' inside the JSON arguments, which the assertion logic likely failed to match due to formatting or escaping issues in the harness.
  - The model correctly identified the need to call 'search_code' and provided all required arguments.

### array of objects argument coercion
- provider: afm-default
- classification: harness_bug
- confidence: 0.95
- rationale: The model's output is structurally valid and contains the correct tool call with the expected 'questions' array inside the 'arguments' JSON string. The failure is due to the assertion checking for the literal string '"questions"' in the raw output, which fails because the string is nested inside the JSON-encoded 'arguments' value (i.e., the output contains the substring '"arguments"' but the assertion logic likely fails to parse the nested JSON to find the key 'questions'). This is a false negative in the harness assertion logic, not a failure of the AFM protocol or model quality.
- evidence:
  - normalized_output contains valid JSON with 'arguments' key containing a 'questions' array
  - Assertion 'contains' value '"questions"' failed because the string is nested inside the JSON-encoded 'arguments' value, not at the top level of the output string

### float number coercion
- provider: afm-default
- classification: harness_bug
- confidence: 0.95
- rationale: The model's output is structurally valid and correctly calls the 'set_temperature' function with the correct arguments ('celsius': 22.5, 'enabled': true). The assertion failure occurs because the test uses a 'contains' check for the string '"celsius"' (including quotes) on the normalized output, which contains the JSON stringified arguments where the key 'celsius' appears without surrounding quotes in the raw JSON structure (i.e., the string '"celsius"' does not exist as a substring in the normalized output string). This is a false negative caused by the assertion logic, not a failure in the model or AFM protocol.
- evidence:
  - normalized_output contains: {"celsius":22.5,\"enabled\":true}
  - Assertion type 'contains' with value '"celsius"' failed because the substring '"celsius"' (with quotes) is not present in the normalized output string.
  - The function call arguments correctly map the user's intent to the schema.

