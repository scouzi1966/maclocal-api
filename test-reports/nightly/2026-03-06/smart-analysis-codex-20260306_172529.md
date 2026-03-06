# Per-Test AI Analysis

### 0. 
**Score: 4/5** 👍 | Status:  | 0.0 tok/s
> Content is correct, coherent, and concise (names three primary colors and why they matter), but the run includes excessive reasoning_content/token use for a concise prompt, which is a minor quality issue.

### 1. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 4/5** 👍 | Status: OK | 110.0 tok/s
> Content is correct, coherent, and follows the prompt with exactly 3 bullet points explaining Rayleigh scattering, but it also includes extensive reasoning_content/token overhead beyond the expected concise output.

### 2. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 3/5** ⚠️ | Status: OK | 118.4 tok/s
> Final content is correct (Red, Blue, Yellow with relevant design rationale), but it does not cleanly meet the concise expectation because it emits an extremely long reasoning trace (1046 tokens) with minor artifact text, indicating inefficient/oververbose output.

### 3. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 5/5** ✅ | Status: OK | 113.7 tok/s
> Content is correct, coherent, and follows the test requirement exactly with 3 bullet points explaining Rayleigh scattering; expected outcome is fully met.

### 4. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 4/5** 👍 | Status: OK | 119.1 tok/s
> Status is OK and the final content is coherent, concise, and directly answers the prompt with three colors plus why they matter; minor accuracy/context caveat for modern design primaries (RGB/CMY) keeps it from excellent.

### 5. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 4/5** 👍 | Status: OK | 113.9 tok/s
> Output content is correct, coherent, and follows the test spec with exactly 3 bullet points explaining why the sky is blue; minor issue is excessive reasoning_content/token usage beyond what was needed.

### 6. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 3/5** ⚠️ | Status: OK | 118.4 tok/s
> Final content is correct and concise (names three colors and why they matter), but the reasoning trace is extremely long and repetitive, consuming many tokens and indicating inefficient/looping generation despite the concise-answer expectation.

### 7. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 4/5** 👍 | Status: OK | 115.9 tok/s
> Content is correct, coherent, and follows the test prompt with exactly 3 bullet points explaining why the sky is blue; minor issue is excessive reasoning_content/token use unrelated to the expected concise output.

### 8. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 4/5** 👍 | Status: OK | 117.9 tok/s
> Content is correct, coherent, and concise per prompt (names three colors and explains design relevance), but the very long, repetitive reasoning trace (1433 completion tokens) indicates inefficiency and weak adherence to concision overall.

### 9. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 4/5** 👍 | Status: OK | 114.8 tok/s
> Output content is correct, coherent, and follows the test constraint of exactly 3 bullet points explaining Rayleigh scattering; minor issue is excessive reasoning_content/token use for a simple format-constrained prompt.

### 10. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 2/5** ❌ | Status: OK | 118.9 tok/s
> Final content is correct and concise, but the output includes extremely long, repetitive reasoning text (1714 completion tokens) that violates the concise expectation and indicates looping behavior.

### 11. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 4/5** 👍 | Status: OK | 114.2 tok/s
> Content is correct, coherent, and follows the prompt with exactly 3 bullet points, but the run produced excessively long reasoning_content (1691 tokens), indicating inefficient generation beyond the expected concise outcome.

### 12. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 4/5** 👍 | Status: OK | 119.1 tok/s
> Content is correct, coherent, and concise (names three primary colors and explains design relevance), but the very large reasoning_content/completion length is inefficient for a concise prompt, so minor deduction.

### 13. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 4/5** 👍 | Status: OK | 110.1 tok/s
> Content is correct, coherent, and follows the spec with exactly 3 bullet points explaining Rayleigh scattering; minor issue is excessive internal reasoning output/token use for a simple task.

### 14. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 4/5** 👍 | Status: OK | 110.2 tok/s
> Content is correct, coherent, and concise per prompt (names three colors and explains design relevance), but the run used excessive hidden reasoning tokens, indicating inefficiency versus expected concise behavior.

### 15. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 2/5** ❌ | Status: OK | 107.6 tok/s
> Status is OK but expected a limerick in content; instead content is empty while reasoning_content consumed the full 4096-token budget with iterative drafting, so it failed the test outcome and is poor quality.

### 16. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 4/5** 👍 | Status: OK | 115.4 tok/s
> Content is correct, coherent, and concise for the prompt (names three primary colors and explains design relevance), but the run generated excessively long reasoning_content (1135 completion tokens) which is a notable efficiency/conciseness issue.

### 17. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 2/5** ❌ | Status: OK | 115.2 tok/s
> Expected a concise limerick in content, but content is empty and the model exhausted max_tokens in repetitive reasoning/drafting loops without delivering the requested poem.

### 18. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 4/5** 👍 | Status: OK | 117.6 tok/s
> Content is correct, coherent, and concise (names three colors and explains design relevance), but the run is inefficient with extremely long reasoning_content despite a concise prompt.

### 19. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 5/5** ✅ | Status: OK | 116.4 tok/s
> Status is OK and the content delivers a coherent, detailed long essay on bread-making history across civilizations, matching the test’s expected outcome with strong structure and relevance.

### 20. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 4/5** 👍 | Status: OK | 118.6 tok/s
> Content is correct, coherent, and concise (names three primary colors and why they matter), but the very long reasoning trace is unnecessary for a concise prompt and indicates inefficiency.

### 21. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 4/5** 👍 | Status: OK | 91.5 tok/s
> Output is coherent, on-topic, and delivers a long cross-civilizational essay as expected, but it appears truncated at the end (mid-thought), so it misses a fully clean completion.

### 22. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 4/5** 👍 | Status: OK | 94.3 tok/s
> Content is correct, coherent, and non-repetitive (meets the repetition-penalty goal), but the model produced very large reasoning_content despite a concise prompt, so efficiency/conciseness is weaker than ideal.

### 23. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 4/5** 👍 | Status: OK | 90.2 tok/s
> Produces a long, coherent essay that stays on-topic and shows little repetition, which fits the repetition-penalty expectation; minor factual/wording issues keep it from excellent.

### 24. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 2/5** ❌ | Status: OK | 94.2 tok/s
> Status is OK but content is empty while reasoning_content consumed the full 4096-token budget with repetitive looping; it failed to produce the concise pirate final answer expected by the test.

### 25. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 4/5** 👍 | Status: OK | 116.6 tok/s
> Output is coherent, accurate, and consistently in pirate speak as expected, but it also exposes extensive reasoning_content and uses an unusually large token budget, which is a notable quality issue.

### 26. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 2/5** ❌ | Status: OK | 119.3 tok/s
> Final content is accurate, but the run badly fails the test’s concise expectation by emitting massive repetitive reasoning (3464 tokens, near max), indicating looping/token-budget waste despite a solid short answer.

### 27. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 5/5** ✅ | Status: OK | 116.2 tok/s
> Status is OK and the response is coherent, accurate, and highly technical, matching the physics-professor expectation with precise terminology, equations, core concepts, algorithms, hardware, and limitations.

### 28. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 3/5** ⚠️ | Status: OK | 119.1 tok/s
> Content is correct and concise (names three primary colors and why they matter), but the response includes an excessively long reasoning trace (1305 completion tokens) that conflicts with the concise expected outcome.

### 29. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 4/5** 👍 | Status: OK | 114.3 tok/s
> Content is clear, coherent, and matches the ELI5/simple-words expectation for quantum computing; minor issue is excessive internal reasoning output/token use despite a short final answer.

### 30. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 4/5** 👍 | Status: OK | 118.9 tok/s
> Content is correct, coherent, and concise (names three primary colors and why they matter), but the run shows excessive exposed reasoning/token use (810 tokens) for a concise prompt, which is a minor quality issue.

### 31. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 5/5** ✅ | Status: OK | 111.8 tok/s
> Output content is exactly a valid JSON object with the required keys (name, age, city) and no extra text, matching the test expectation.

### 32. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 4/5** 👍 | Status: OK | 117.4 tok/s
> Content is correct, coherent, and concise for the prompt (red/blue/yellow with clear design relevance), but the run shows excessive reasoning output/token use relative to the expected concise behavior.

### 33. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 5/5** ✅ | Status: OK | 112.3 tok/s
> Content exactly matches the spec: 5 animals, one per line, numbered 1-5, with no extra text in the final content.

### 34. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 2/5** ❌ | Status: OK | 119.3 tok/s
> Expected guided-json output matching the provided schema, but the model returned plain prose (not JSON) and leaked extensive reasoning despite a concise prompt; content is coherent but fails the test’s formatting objective.

### 35. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 2/5** ❌ | Status: OK | 111.5 tok/s
> For a guided-json schema requiring exactly {name:string, age:integer}, the output failed expectations: it returned explanatory prose/markdown plus a different object (missing required name/age, many extra fields) and excessive reasoning leakage.

### 36. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 2/5** ❌ | Status: OK | 121.6 tok/s
> For a guided-json-nested test, output should conform to the required JSON schema (city/population/landmarks), but the model returned plain prose plus verbose reasoning, so it fails the core expected outcome despite coherent content.

### 37. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 2/5** ❌ | Status: OK | 113.2 tok/s
> For a guided-JSON test, output should conform to the provided schema (city string, population integer, landmarks string array), but the model returned markdown prose plus exposed reasoning instead of valid JSON, so it significantly fails the expected outcome.

### 38. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 4/5** 👍 | Status: OK | 121.5 tok/s
> Correctly names three primary colors and gives coherent design relevance, but it is not very concise and adds an unnecessary codebase-related note beyond the expected outcome.

### 39. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 2/5** ❌ | Status: OK | 95.4 tok/s
> Output fails expected outcome: it only emits a read_file tool tag and never explains main.swift, so the task is largely unmet despite coherent text.

### 40. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 4/5** 👍 | Status: OK | 88.9 tok/s
> Correctly names three primary colors and gives coherent design relevance, but it is not very concise and adds an unnecessary meta note, so it falls slightly short of the expected concise outcome.

### 41. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 2/5** ❌ | Status: OK | 96.1 tok/s
> Output does not complete the requested CLI timeout change and emits a malformed pseudo tool call; finish_reason is \

### 42. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 4/5** 👍 | Status: OK | 98.2 tok/s
> Correctly names three primary colors (red, blue, yellow) and gives clear design relevance, but it is not very concise and adds an unnecessary meta note outside the requested answer.

### 43. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 2/5** ❌ | Status: OK | 96.1 tok/s
> Did not produce the requested unit test; it only asked for context and emitted malformed pseudo-tool-call text, so it fails the expected outcome despite status=OK.

### 44. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 4/5** 👍 | Status: OK | 105.8 tok/s
> Correctly names three primary colors and gives coherent design relevance; mostly meets expected concise outcome, with minor verbosity and an unnecessary meta note about codebase context.

### 45. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 2/5** ❌ | Status: OK | 95.2 tok/s
> Expected outcome was to explain the file (or, in a tool-call flow, emit a proper tool call with finish_reason=tool_calls), but the model only output a raw read_file tag and stopped without providing the explanation.

### 46. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 4/5** 👍 | Status: OK | 102.5 tok/s
> Correctly names three primary colors and gives coherent design relevance, but it is not very concise and adds an unnecessary meta note beyond the expected brief answer.

### 47. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 2/5** ❌ | Status: OK | 95.5 tok/s
> Expected a proper tool invocation flow for a coding-agent task, but output stops at a malformed pseudo tool call, with finish_reason=\

### 48. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 4/5** 👍 | Status: OK | 112.0 tok/s
> Correctly lists three primary colors and gives coherent design relevance, so it meets the test intent; minor downside is it is not very concise and adds unnecessary meta commentary.

### 49. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 95.5 tok/s
> Coherent but fails the test intent: instead of writing a unit test for the previously described timeout feature, it asks for clarification and provides no actual test implementation.

### 50. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 119.4 tok/s
> Expected a concise user-facing answer naming three primary colors and why they matter, but content is empty and the model used the full token budget in truncated reasoning_content instead of producing the final response.

### 51. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 2/5** ❌ | Status: OK | 55.3 tok/s
> status is OK but expected answer content is empty; all 50 completion tokens were spent in reasoning_content (truncated planning text), so it failed to deliver the requested Rome history output.

### 52. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 4/5** 👍 | Status: OK | 98.3 tok/s
> Content is correct, coherent, and concise (names three colors and explains design relevance), matching the prompt, but the generation includes excessively long reasoning_content, which is a minor quality issue.

### 53. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s
> Content is coherent and detailed for ingredients/steps, but it appears truncated at the 2000-token limit and does not clearly deliver the requested tips and variations in the visible answer; excessive reasoning_content also suggests output control issues.

### 54. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 3/5** ⚠️ | Status: OK | 119.8 tok/s
> Content is correct and concise (names three primary colors and why they matter), but the run shows excessive reasoning output/token use for a concise prompt and `logprobs_count` is 0 despite a logprobs test setup.

### 55. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 4/5** 👍 | Status: OK | 111.9 tok/s
> Status is OK and the final content correctly answers the prompt (\

### 56. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 4/5** 👍 | Status: OK | 112.0 tok/s
> Content is correct, coherent, and concise (names three primary colors and explains design relevance), but the very large reasoning_content is unnecessary for a concise prompt and indicates inefficiency.

### 57. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 5/5** ✅ | Status: OK | 109.0 tok/s
> Status is OK and the content directly answers the prompt with a clear, coherent few-paragraph summary covering core ML ideas (definition, learning types, and challenges/impact), matching the expected outcome.

### 58. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 2/5** ❌ | Status: OK | 119.2 tok/s
> Expected a concise final answer in content, but content is empty and the model exhausted max tokens in repetitive reasoning_content without delivering the required response.

### 59. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 5/5** ✅ | Status: OK | 115.3 tok/s
> Status is OK and the content directly answers the prompt with a clear, coherent 3-paragraph summary covering definition, major learning paradigms, applications, and key challenges as expected.

### 60. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 4/5** 👍 | Status: OK | 119.8 tok/s
> Content is correct, coherent, and concise as requested (names three primary colors and why they matter), but the extremely long reasoning trace and high token usage are a notable quality issue for a concise prompt.

### 61. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 4/5** 👍 | Status: OK | 117.7 tok/s
> Status is OK and the content correctly identifies three relevant bottlenecks (MLX/KV cache, JSON+SSE overhead, and concurrency contention) with coherent, architecture-specific rationale, but it is excessively long and includes unnecessary reasoning trace beyond the expected concise top-3 outcome.

### 62. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 2/5** ❌ | Status: OK | 120.9 tok/s
> Final content is correct and concise, but the reasoning_content is highly repetitive/looping and extremely long (2610 tokens) for a simple concise prompt, which is a significant quality failure per spec.

### 63. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 4/5** 👍 | Status: OK | 117.8 tok/s
> Output is correct and coherent, clearly identifies three high-impact bottlenecks aligned with the architecture (MLX/KV cache, JSON+SSE overhead, concurrency contention), but it is excessively long and over-elaborated for a top-3 request.

### 64. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 2/5** ❌ | Status: OK | 117.9 tok/s
> Final content is correct and concise, but the reasoning_content is massively repetitive/looping and consumes excessive tokens for a simple prompt, which is a significant quality issue per spec.

### 65. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 4/5** 👍 | Status: OK | 116.6 tok/s
> Meets the test goal by clearly identifying three high-impact bottlenecks (MLX/KV cache memory behavior, JSON+SSE overhead, and concurrency/serial-access contention) with coherent architecture-specific rationale; minor issue is excessive length/redundancy rather than concise top-3 output.

### 66. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 4/5** 👍 | Status: OK | 118.4 tok/s
> Content is correct, coherent, and concise (names three primary colors and why they matter), matching the test intent; minor issue is excessive reasoning_content/token use despite a concise-answer prompt.

### 67. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 3/5** ⚠️ | Status: OK | 112.6 tok/s
> Final content is a coherent short moon poem and matches the prompt, but the response includes massive repetitive reasoning/looping (2845 tokens) that undermines expected concise output quality.

### 68. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 2/5** ❌ | Status: OK | 118.7 tok/s
> Correct colors and rationale are present, but it badly fails the test’s concise-answer expectation by emitting a very long internal-thought block before the final response.

### 69. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 4/5** 👍 | Status: OK | 115.9 tok/s
> Output is correct and coherent with a clear step-by-step derivation to 391, matching the test’s intended reasoning behavior, but it is excessively long and includes unnecessary internal-thinking markup/artifacts (e.g., <think>, stray 'cw').

### 70. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 2/5** ❌ | Status: OK | 118.2 tok/s
> Stop-sequence test failed expectation: the configured stop string \

### 71. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 5/5** ✅ | Status: OK | 113.2 tok/s
> Passes the stop-sequence expectation for `stop-single`: visible content stops at `2. Banana`, does not include the stop string `3.`, and finish_reason is `stop`.

### 72. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 4/5** 👍 | Status: OK | 82.2 tok/s
> Content is correct, concise, and coherent for the prompt, and output does not include either stop string (``` or END), so stop-multi behavior mostly meets expectations; minor issue is excessive reasoning trace with odd trailing text.

### 73. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 2/5** ❌ | Status: OK | 113.7 tok/s
> Poor for a stop-sequence test: no usable final content was returned, and reasoning_content includes stop strings (``` and END), indicating the run did not cleanly meet the expected stop behavior/output.

### 74. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 2/5** ❌ | Status: OK | 105.1 tok/s
> Status is OK but the model failed the test intent: `content` is empty, it emitted verbose reasoning instead of a concise answer, and the output includes newline characters despite a stop-on-newline setting.

### 75. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 121.5 tok/s
> Fails expected stop-newline behavior and user-facing output: `content` is empty (no one-sentence answer), while `reasoning_content` contains newline characters including the stop string and leaked chain-of-thought.

### 76. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 101.1 tok/s
> Stop-sequence test underperformed: output exposed only verbose reasoning_content (no user-facing content) and contains the stop string pattern (double newlines), violating expected stop behavior despite otherwise correct color facts.

### 77. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 2/5** ❌ | Status: OK | 120.5 tok/s
> Stop-sequence test failed expectations: output includes the stop string (double newlines) and produced only reasoning_content with empty content, so it did not deliver the requested two paragraphs cleanly.

### 78. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 4/5** 👍 | Status: OK | 122.3 tok/s
> Output is correct and coherent (names three primary colors and explains design relevance concisely), and it does not contain the stop string; minor issue is excessive internal reasoning/token use for a concise prompt.

### 79. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 2/5** ❌ | Status: OK | 112.1 tok/s
> Poor stop-sequence behavior: the output includes the stop string \

### 80. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 4/5** 👍 | Status: OK | 111.2 tok/s
> Stop-sequence behavior appears correct (finish_reason=stop and visible content omits the period), but the final answer is incomplete for the prompt since it names colors without the concise design-importance explanation.

### 81. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 4/5** 👍 | Status: OK | 119.2 tok/s
> For a stop-sequence test, it passes the key requirement: output content does not include the stop string \

### 82. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 2/5** ❌ | Status: OK | 120.4 tok/s
> Stop-sequence test failed expectation: output includes the stop string \

### 83. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 5/5** ✅ | Status: OK | 111.4 tok/s
> Stop-sequence behavior appears correct: generation stopped before emitting the stop string \

### 84. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 4/5** 👍 | Status: OK | 80.5 tok/s
> Output content is correct, coherent, and concise (names three primary colors and why they matter), and it does not include the stop strings (` ``` ` or `DONE`), but the very long reasoning_content and odd trailing artifact ('cw') are minor quality issues for this concise prompt.

### 85. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 2/5** ❌ | Status: OK | 111.0 tok/s
> Expected a usable script response for the stop-sequence test, but user-facing content is empty and only internal reasoning text was produced; this fails the intended outcome despite status=OK.

### 86. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 2/5** ❌ | Status: OK | 109.2 tok/s
> Stop-sequence behavior appears to fail the test expectation: despite coherent final content, the generated output includes the stop string pattern (e.g., \

### 87. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 5/5** ✅ | Status: OK | 113.9 tok/s
> Passes stop-sequence behavior: status is OK, finish_reason is \

### 88. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 2/5** ❌ | Status: OK | 87.1 tok/s
> Stop-sequence expectation failed: the output includes the stop string \

### 89. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 5/5** ✅ | Status: OK | 111.8 tok/s
> Meets stop-sequence expectation: generation stopped early with finish_reason=\

### 90. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 2/5** ❌ | Status: OK | 87.8 tok/s
> Stop-sequence expectation is violated because the output contains the stop string \

### 91. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 110.8 tok/s
> Fails the stop-sequence expectation: output includes the stop string \

### 92. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 117.4 tok/s
> Poor fit for the guided-json test: output is free-form prose instead of the required JSON object with a `cities` array, and it includes excessive reasoning for a concise prompt, so it misses the expected outcome despite status=OK.

### 93. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 113.2 tok/s
> Output fails expected outcome: content is just '[\\\

### 94. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 90.5 tok/s
> Output avoids the stop string ',' (as required for stop tests), but the visible answer is truncated/incomplete ('The three primary colors are **red') and fails the concise prompt intent; most tokens were spent in reasoning instead of useful content.

### 95. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 122.3 tok/s
> Output fails the guided-JSON expectation (should return an object with name/age/city) and instead gives plain text with extensive reasoning leakage; although the stop string comma is not present, it does not meet the test’s intended structured outcome.

### 96. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 113.6 tok/s
> Fails the guided-JSON expectation for this test (returned freeform prose instead of the required {color,hex} object), with excessive internal reasoning despite a concise prompt; stop string handling appears fine.

### 97. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 2/5** ❌ | Status: OK | 110.8 tok/s
> For a guided-JSON brace-stop test, output should match the required JSON schema ({color, hex}); instead it returned free-form prose (and leaked reasoning_content), so it largely fails the expected structured outcome despite being coherent and not including the stop string '}'.

### 98. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 5/5** ✅ | Status: OK | 119.2 tok/s
> Output is coherent, concise, and valid JSON with correct primary colors and design rationale; it appears to satisfy the stop-sequence objective since the stop string does not appear in the final content.

### 99. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 2/5** ❌ | Status: OK | 110.6 tok/s
> Failed expected stop-sequence behavior for stop-json-object-key: output hit max tokens with repetitive looping reasoning, empty content, and includes the stop string ('age'), so it did not cleanly return the JSON object.

### 100. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 4/5** 👍 | Status: OK | 117.3 tok/s
> Output is correct, coherent, and concise in `content` (names three primary colors with relevant design rationale), and it does not include the stop string; minor deduction for excessively long `reasoning_content` despite a concise-answer prompt.

### 101. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 5/5** ✅ | Status: OK | 112.0 tok/s
> Stop-sequence behavior appears correct: generation ended with finish_reason=\

### 102. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 5/5** ✅ | Status: OK | 152.2 tok/s
> Meets stop-multi-word expectation: output is correct/coherent and does not contain the stop string \

### 103. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 2/5** ❌ | Status: OK | 111.7 tok/s
> For a stop-sequence test (stop=\

### 104. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 2/5** ❌ | Status: OK | 94.8 tok/s
> Status is OK and stop string was not emitted, but expected concise answer content is missing; all 256 completion tokens were spent in reasoning_content (near max_tokens), so the test outcome is poor for the spec.

### 105. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 96.0 tok/s
> Status is OK, but expected a normal answer in content for a stop-no-match prompt; instead content is empty while reasoning_content consumed the full 256-token budget (close to max_tokens), so the user-facing response failed despite partial task-relevant reasoning.

### 106. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 116.8 tok/s
> Stop-sequence test expectation is violated: output includes stop strings (e.g., \

### 107. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 116.0 tok/s
> Status is OK but expected outcome is not met: `content` is empty (no actual answer), while 767 completion tokens were spent in `reasoning_content`, indicating token budget was consumed without producing the user-facing response for this stop test.

### 108. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 120.4 tok/s
> Poor for a stop-sequence test: although status is OK, the model consumed many tokens in reasoning, produced an incomplete final answer, and the output payload includes the stop string (`**`) in reasoning_content, which violates expected stop behavior.

### 109. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 2/5** ❌ | Status: OK | 122.4 tok/s
> Stop handling appears to trigger before '**' (content omits the stop string), but the visible answer is truncated to a partial first fact and fails the expected 3-fact bold-formatted output; most tokens went into reasoning_content instead of user-facing content.

### 110. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 5/5** ✅ | Status: OK | 104.5 tok/s
> Status is OK, the response correctly and concisely answers with three primary colors and why they matter, and for this stop-sequence test it does not contain the stop string \

### 111. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 2/5** ❌ | Status: OK | 112.0 tok/s
> For a stop-sequence test, the output must not contain the stop string, but `</li>` appears in `reasoning_content`; although `content` is correctly truncated and `finish_reason` is `stop`, this fails the key stop-html-tag expectation.

### 112. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 5/5** ✅ | Status: OK | 86.0 tok/s
> Status is OK, content is correct and concise, and for this stop-unicode test the output does not contain the configured stop string (•), so it meets the expected outcome.

### 113. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 112.0 tok/s
> For a stop-sequence unicode test, output should avoid emitting the stop string and still provide a usable answer; here content is empty, reasoning_content leaked internal planning (including \

### 114. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 105.4 tok/s
> Status is OK and stop string was not emitted in content, but the actual answer is truncated to just \

### 115. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 120.9 tok/s
> Failed expected stop-sequence behavior and task completion: output hit max_tokens with empty content and long reasoning, and includes stop strings (e.g., \

### 116. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 117.0 tok/s
> Failed expected stop-sequence behavior and usable completion: content is empty, reasoning consumed full 4096 tokens, and includes repetitive looping text containing the stop string \

### 117. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 117.3 tok/s
> Stop-sequence behavior is poor for this test: although status is OK and content is coherent, the output includes the stop string \

### 118. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 101.6 tok/s
> Final content is correct and respects numbered-list format with no stop string in `content`, but the run shows severe repetitive looping in `reasoning_content` (1455 completion tokens for a concise prompt), which is a significant quality failure for this test.

### 119. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 4/5** 👍 | Status: OK | 116.6 tok/s
> Main output is correct, coherent, and matches the numbered-list instruction while respecting the stop-sequence behavior (content stops before \

### 120. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 2/5** ❌ | Status: OK | 116.5 tok/s
> Stop-sequence test failed expectation: the output includes the stop string \

### 121. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 116.1 tok/s
> Stop-sequence behavior is not clean: although visible content stops at item 2, the output includes the stop string \

### 122. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 87.6 tok/s
> Answer content is correct, but this stop-sequence run fails expectation because output includes the stop string \

### 123. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 5/5** ✅ | Status: OK | 114.1 tok/s
> Stop-sequence behavior appears correct for this test: content is coherent and cleanly truncated at 2 items without emitting the stop string \

### 124. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 2/5** ❌ | Status: OK | 86.7 tok/s
> Content answer is correct, but this stop-sequence test fails expectations because the output includes the stop string \

### 125. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 3/5** ⚠️ | Status: OK | 114.4 tok/s

### 126. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 3/5** ⚠️ | Status: OK | 86.9 tok/s

### 127. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 3/5** ⚠️ | Status: OK | 72.8 tok/s

### 128. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 3/5** ⚠️ | Status: OK | 108.2 tok/s

### 129. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 3/5** ⚠️ | Status: OK | 113.9 tok/s

### 130. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 3/5** ⚠️ | Status: OK | 120.2 tok/s

### 131. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 3/5** ⚠️ | Status: OK | 113.6 tok/s

### 132. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 3/5** ⚠️ | Status: OK | 120.9 tok/s

### 133. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s

### 134. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 3/5** ⚠️ | Status: OK | 118.7 tok/s

### 135. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 3/5** ⚠️ | Status: OK | 113.3 tok/s

### 136. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 3/5** ⚠️ | Status: OK | 119.7 tok/s

### 137. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 3/5** ⚠️ | Status: OK | 113.5 tok/s

### 138. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 3/5** ⚠️ | Status: OK | 118.9 tok/s

### 139. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s

### 140. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 118.5 tok/s

### 141. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 114.0 tok/s

### 142. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 118.8 tok/s

### 143. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s

### 144. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 121.4 tok/s

### 145. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 121.2 tok/s

### 146. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 115.0 tok/s

### 147. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 120.3 tok/s

### 148. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 119.6 tok/s

### 149. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 3/5** ⚠️ | Status: OK | 116.5 tok/s

### 150. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 3/5** ⚠️ | Status: OK | 120.3 tok/s

### 151. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 3/5** ⚠️ | Status: OK | 113.4 tok/s

### 152. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 3/5** ⚠️ | Status: OK | 119.8 tok/s

### 153. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s

### 154. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 115.7 tok/s

### 155. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 119.9 tok/s

### 156. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 115.4 tok/s

### 157. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 116.4 tok/s

### 158. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 120.0 tok/s

### 159. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 114.3 tok/s

### 160. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 113.6 tok/s

### 161. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 121.2 tok/s

### 162. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 115.2 tok/s

### 163. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 116.7 tok/s

### 164. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 119.7 tok/s

### 165. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 3/5** ⚠️ | Status: OK | 114.8 tok/s

### 166. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 3/5** ⚠️ | Status: OK | 120.0 tok/s

### 167. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 3/5** ⚠️ | Status: OK | 117.4 tok/s

### 168. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 3/5** ⚠️ | Status: OK | 118.0 tok/s

### 169. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 3/5** ⚠️ | Status: OK | 115.3 tok/s

### 170. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 3/5** ⚠️ | Status: OK | 120.2 tok/s

### 171. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 3/5** ⚠️ | Status: OK | 114.4 tok/s

### 172. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 3/5** ⚠️ | Status: OK | 118.1 tok/s

### 173. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 3/5** ⚠️ | Status: OK | 117.9 tok/s

### 174. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 3/5** ⚠️ | Status: OK | 120.6 tok/s

### 175. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 3/5** ⚠️ | Status: OK | 96.3 tok/s

### 176. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 3/5** ⚠️ | Status: OK | 115.2 tok/s

### 177. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 3/5** ⚠️ | Status: OK | 92.6 tok/s

### 178. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 3/5** ⚠️ | Status: OK | 95.4 tok/s

### 179. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 3/5** ⚠️ | Status: OK | 110.8 tok/s

### 180. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 3/5** ⚠️ | Status: OK | 120.3 tok/s

### 181. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 3/5** ⚠️ | Status: OK | 116.6 tok/s

### 182. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 3/5** ⚠️ | Status: OK | 118.2 tok/s

### 183. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 113.0 tok/s

### 184. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 116.7 tok/s

---
**Summary**: 60/185 passed (score ≥ 4), 59 failed (score ≤ 2)

<!-- AI_SCORES [{"i": 0, "s": 4}, {"i": 1, "s": 4}, {"i": 2, "s": 3}, {"i": 3, "s": 5}, {"i": 4, "s": 4}, {"i": 5, "s": 4}, {"i": 6, "s": 3}, {"i": 7, "s": 4}, {"i": 8, "s": 4}, {"i": 9, "s": 4}, {"i": 10, "s": 2}, {"i": 11, "s": 4}, {"i": 12, "s": 4}, {"i": 13, "s": 4}, {"i": 14, "s": 4}, {"i": 15, "s": 2}, {"i": 16, "s": 4}, {"i": 17, "s": 2}, {"i": 18, "s": 4}, {"i": 19, "s": 5}, {"i": 20, "s": 4}, {"i": 21, "s": 4}, {"i": 22, "s": 4}, {"i": 23, "s": 4}, {"i": 24, "s": 2}, {"i": 25, "s": 4}, {"i": 26, "s": 2}, {"i": 27, "s": 5}, {"i": 28, "s": 3}, {"i": 29, "s": 4}, {"i": 30, "s": 4}, {"i": 31, "s": 5}, {"i": 32, "s": 4}, {"i": 33, "s": 5}, {"i": 34, "s": 2}, {"i": 35, "s": 2}, {"i": 36, "s": 2}, {"i": 37, "s": 2}, {"i": 38, "s": 4}, {"i": 39, "s": 2}, {"i": 40, "s": 4}, {"i": 41, "s": 2}, {"i": 42, "s": 4}, {"i": 43, "s": 2}, {"i": 44, "s": 4}, {"i": 45, "s": 2}, {"i": 46, "s": 4}, {"i": 47, "s": 2}, {"i": 48, "s": 4}, {"i": 49, "s": 2}, {"i": 50, "s": 2}, {"i": 51, "s": 2}, {"i": 52, "s": 4}, {"i": 53, "s": 3}, {"i": 54, "s": 3}, {"i": 55, "s": 4}, {"i": 56, "s": 4}, {"i": 57, "s": 5}, {"i": 58, "s": 2}, {"i": 59, "s": 5}, {"i": 60, "s": 4}, {"i": 61, "s": 4}, {"i": 62, "s": 2}, {"i": 63, "s": 4}, {"i": 64, "s": 2}, {"i": 65, "s": 4}, {"i": 66, "s": 4}, {"i": 67, "s": 3}, {"i": 68, "s": 2}, {"i": 69, "s": 4}, {"i": 70, "s": 2}, {"i": 71, "s": 5}, {"i": 72, "s": 4}, {"i": 73, "s": 2}, {"i": 74, "s": 2}, {"i": 75, "s": 2}, {"i": 76, "s": 2}, {"i": 77, "s": 2}, {"i": 78, "s": 4}, {"i": 79, "s": 2}, {"i": 80, "s": 4}, {"i": 81, "s": 4}, {"i": 82, "s": 2}, {"i": 83, "s": 5}, {"i": 84, "s": 4}, {"i": 85, "s": 2}, {"i": 86, "s": 2}, {"i": 87, "s": 5}, {"i": 88, "s": 2}, {"i": 89, "s": 5}, {"i": 90, "s": 2}, {"i": 91, "s": 2}, {"i": 92, "s": 2}, {"i": 93, "s": 2}, {"i": 94, "s": 2}, {"i": 95, "s": 2}, {"i": 96, "s": 2}, {"i": 97, "s": 2}, {"i": 98, "s": 5}, {"i": 99, "s": 2}, {"i": 100, "s": 4}, {"i": 101, "s": 5}, {"i": 102, "s": 5}, {"i": 103, "s": 2}, {"i": 104, "s": 2}, {"i": 105, "s": 2}, {"i": 106, "s": 2}, {"i": 107, "s": 2}, {"i": 108, "s": 2}, {"i": 109, "s": 2}, {"i": 110, "s": 5}, {"i": 111, "s": 2}, {"i": 112, "s": 5}, {"i": 113, "s": 2}, {"i": 114, "s": 2}, {"i": 115, "s": 2}, {"i": 116, "s": 2}, {"i": 117, "s": 2}, {"i": 118, "s": 2}, {"i": 119, "s": 4}, {"i": 120, "s": 2}, {"i": 121, "s": 2}, {"i": 122, "s": 2}, {"i": 123, "s": 5}, {"i": 124, "s": 2}, {"i": 125, "s": 3}, {"i": 126, "s": 3}, {"i": 127, "s": 3}, {"i": 128, "s": 3}, {"i": 129, "s": 3}, {"i": 130, "s": 3}, {"i": 131, "s": 3}, {"i": 132, "s": 3}, {"i": 133, "s": 3}, {"i": 134, "s": 3}, {"i": 135, "s": 3}, {"i": 136, "s": 3}, {"i": 137, "s": 3}, {"i": 138, "s": 3}, {"i": 139, "s": 3}, {"i": 140, "s": 3}, {"i": 141, "s": 3}, {"i": 142, "s": 3}, {"i": 143, "s": 3}, {"i": 144, "s": 3}, {"i": 145, "s": 3}, {"i": 146, "s": 3}, {"i": 147, "s": 3}, {"i": 148, "s": 3}, {"i": 149, "s": 3}, {"i": 150, "s": 3}, {"i": 151, "s": 3}, {"i": 152, "s": 3}, {"i": 153, "s": 3}, {"i": 154, "s": 3}, {"i": 155, "s": 3}, {"i": 156, "s": 3}, {"i": 157, "s": 3}, {"i": 158, "s": 3}, {"i": 159, "s": 3}, {"i": 160, "s": 3}, {"i": 161, "s": 3}, {"i": 162, "s": 3}, {"i": 163, "s": 3}, {"i": 164, "s": 3}, {"i": 165, "s": 3}, {"i": 166, "s": 3}, {"i": 167, "s": 3}, {"i": 168, "s": 3}, {"i": 169, "s": 3}, {"i": 170, "s": 3}, {"i": 171, "s": 3}, {"i": 172, "s": 3}, {"i": 173, "s": 3}, {"i": 174, "s": 3}, {"i": 175, "s": 3}, {"i": 176, "s": 3}, {"i": 177, "s": 3}, {"i": 178, "s": 3}, {"i": 179, "s": 3}, {"i": 180, "s": 3}, {"i": 181, "s": 3}, {"i": 182, "s": 3}, {"i": 183, "s": 3}, {"i": 184, "s": 3}] -->
