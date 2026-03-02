# Per-Test AI Analysis

### 0. 
**Score: 2/5** ❌ | Status:  | 0.0 tok/s
> Result payload only contains _meta/test-run info and lacks status/content fields expected by the spec, so the model output cannot be validated against the test outcome.

### 1. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 2/5** ❌ | Status: OK | 113.3 tok/s
> Output is coherent for the given prompt, but it fails the test spec expectation (should be a precise 3-bullet Rayleigh scattering response), and it includes excessively long reasoning instead of the expected structured target.

### 2. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 5/5** ✅ | Status: OK | 121.3 tok/s
> Status OK and content is factually correct, coherent, and exactly 3 structured bullet points on Rayleigh scattering, matching the greedy-decoding expectation for precise output.

### 3. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 4/5** 👍 | Status: OK | 112.6 tok/s
> Content is correct, coherent, and concise on the expected topic (primary colors and design relevance), matching the default-sampling goal, but the run is inefficient with very large reasoning_content/token usage for a simple concise prompt.

### 4. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 5/5** ✅ | Status: OK | 118.4 tok/s
> Output is coherent, accurate, and follows the prompt exactly with 3 bullet points on the correct topic, matching the expected default-sampling behavior.

### 5. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 2/5** ❌ | Status: OK | 119.9 tok/s
> Response stays on-topic and correct, but it badly misses the test’s concise-output expectation by generating extremely long, repetitive reasoning/looping text; high temperature should add variety, not verbose self-repetition.

### 6. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 5/5** ✅ | Status: OK | 119.1 tok/s
> Output is coherent, accurate, and exactly 3 bullet points on-topic; despite high temperature, it remains clear and not garbled, meeting the expected outcome.

### 7. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 3/5** ⚠️ | Status: OK | 110.5 tok/s
> Content is correct and coherent, but it does not meet the expected 3-bullet format and includes excessive leaked reasoning despite a concise prompt, so it only partially matches the top-p test expectation.

### 8. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 4/5** 👍 | Status: OK | 117.3 tok/s
> Meets the expected outcome with a coherent, accurate response in exactly 3 bullet points, but the extremely long reasoning_content (912 completion tokens) suggests inefficient generation for this simple top-p test.

### 9. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 4/5** 👍 | Status: OK | 104.9 tok/s
> Output is coherent and correctly answers with three primary colors plus concise design relevance, matching the top-k test expectation; minor quality issues include excessively long reasoning_content and a small trailing artifact (\

### 10. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 5/5** ✅ | Status: OK | 117.9 tok/s
> Meets expected top-k outcome: content is coherent, accurate, and follows the prompt exactly with 3 clear bullet points explaining Rayleigh scattering and perception.

### 11. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 4/5** 👍 | Status: OK | 116.5 tok/s
> Output content is correct, coherent, and concise as expected for the min-p test, but the very long reasoning_content (1249 completion tokens) suggests unnecessary verbosity/inefficiency.

### 12. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 4/5** 👍 | Status: OK | 121.3 tok/s
> Meets the min-p test expectation with a coherent, correct answer in exactly 3 bullet points; minor issue is excessive reasoning_content/token use despite solid final content.

### 13. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 4/5** 👍 | Status: OK | 110.1 tok/s
> Sampler chain appears stable (no crash/conflict) and the final content is correct and coherent for the prompt, but the very long reasoning_content and token overuse reduce quality for a concise-response test.

### 14. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 5/5** ✅ | Status: OK | 115.3 tok/s
> Sampler combination behaved correctly: generation completed without conflict/crash and produced a coherent answer in exactly 3 bullet points matching the test expectation.

### 15. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 2/5** ❌ | Status: OK | 118.8 tok/s
> Status is OK and output is coherent, but it fails the test spec’s expected outcome (a creative limerick for seed determinism testing); this is a direct expectation mismatch, and run-to-run identity cannot be verified from run 1 alone.

### 16. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 2/5** ❌ | Status: OK | 122.5 tok/s
> Expected a finished creative limerick for seed-determinism testing, but output exhausted max tokens in reasoning_content and produced empty content with no final poem.

### 17. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 4/5** 👍 | Status: OK | 118.8 tok/s
> Status is OK and the final content is correct and concise per prompt; however, this single result does not provide seed-42-run1 for required identity check, so determinism expectation cannot be fully confirmed here.

### 18. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 2/5** ❌ | Status: OK | 123.6 tok/s
> Output is poor for the spec: it produced no user-facing content (empty `content`) and exhausted the 4096-token budget in repetitive `reasoning_content`, so the limerick task was not completed; determinism vs run1 cannot be confirmed from this single result.

### 19. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 3/5** ⚠️ | Status: OK | 120.0 tok/s
> Status is OK and final content is correct/coherent, but it does not match the test spec’s expected long control-style output and includes excessively long, repetitive reasoning text, so it only partially meets the expected outcome.

### 20. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 5/5** ✅ | Status: OK | 123.4 tok/s
> Status is OK and output is a coherent, detailed long essay directly on bread-making history across civilizations, matching the no-penalty control expectation (natural repetition only, no major quality issues).

### 21. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 4/5** 👍 | Status: OK | 95.2 tok/s
> Output content is correct, coherent, and non-repetitive with varied wording, matching the presence-penalty goal; however, the very long reasoning trace (1582 tokens) is excessive for a concise prompt and is a minor quality issue.

### 22. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 5/5** ✅ | Status: OK | 97.9 tok/s
> Output is long, coherent, and uses varied vocabulary with minimal repetition, matching the strong presence-penalty expectation without becoming too short or incoherent.

### 23. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 3/5** ⚠️ | Status: OK | 95.9 tok/s
> Final content is coherent and correct, but the generated reasoning is highly repetitive/looping and extremely long for a concise prompt, which weakens the expected reduced-repetition behavior for this repetition-penalty test.

### 24. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 5/5** ✅ | Status: OK | 98.0 tok/s
> Output is long, coherent, and on-topic with no obvious phrase-level looping/repetition, which matches the repetition-penalty expectation of reduced repetition while maintaining quality.

### 25. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 2/5** ❌ | Status: OK | 121.0 tok/s
> Although the final content is coherent pirate-speak, it fails the test spec expectation to describe quantum computing concepts, and the output includes excessively long repetitive reasoning instead of a concise answer.

### 26. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 5/5** ✅ | Status: OK | 121.0 tok/s
> Meets the test expectation well: response is consistently pirate-themed (e.g., 'Arrr', 'matey', 'ye') while accurately covering core quantum concepts (qubits, superposition, entanglement, and hardware fragility).

### 27. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 4/5** 👍 | Status: OK | 119.3 tok/s
> Meets the scientist persona well: formal, technical, and precise (e.g., trichromatic cones, superposition, colorimetry), but the run is inefficient with extremely long reasoning despite a concise final answer.

### 28. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 5/5** ✅ | Status: OK | 123.4 tok/s
> Matches the scientist persona very well: formal academic tone, precise structure, and strong technical terminology including qubits, superposition, entanglement, Hilbert space, unitary gates, and decoherence; coherent and fully aligned with expected outcome.

### 29. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 2/5** ❌ | Status: OK | 122.3 tok/s
> status is OK but content is empty while reasoning consumed the full 4096-token budget, and the reasoning is highly repetitive/looping instead of giving the expected concise ELI5 answer.

### 30. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 4/5** 👍 | Status: OK | 122.8 tok/s
> Content matches the ELI5 persona well with simple analogies and no jargon in the user-facing answer, but the emitted reasoning_content includes technical/jargon terms (e.g., \

### 31. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 2/5** ❌ | Status: OK | 117.3 tok/s
> Failed core instruction-following expectation: output is prose (and long reasoning), not valid JSON with exactly keys \

### 32. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 5/5** ✅ | Status: OK | 122.6 tok/s
> Meets expected instruction-following: content is valid JSON with exactly keys \

### 33. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 2/5** ❌ | Status: OK | 116.7 tok/s
> Output is coherent but fails the test’s expected format: not exactly 5 numbered lines of 'N. animal_name' and includes unrelated explanatory prose instead.

### 34. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 5/5** ✅ | Status: OK | 120.4 tok/s
> Content exactly matches the expected strict format: 5 lines, numbered 1-5, each with an animal name, and no extra text in the output content.

### 35. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 2/5** ❌ | Status: OK | 115.4 tok/s
> Expected guided JSON object matching {\

### 36. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 2/5** ❌ | Status: OK | 124.1 tok/s
> Output fails the guided-json expectation: content is not a raw valid JSON object for schema {\

### 37. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 2/5** ❌ | Status: OK | 117.7 tok/s
> Output is coherent but fails the guided-json-nested expectation: content is plain text (not valid JSON) and does not include required schema fields city, population (integer), and landmarks array (3+ strings).

### 38. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 2/5** ❌ | Status: OK | 123.3 tok/s
> Expected guided JSON schema output, but response is Markdown text (not valid JSON) and does not provide required fields as typed keys (`city`, integer `population`, `landmarks` array of strings), so it fails core schema-conformance expectations.

### 39. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 2/5** ❌ | Status: OK | 98.1 tok/s
> Response is coherent, but it does not meet the test spec expectation of a Swift-code-related answer for baseline turn 1; it answers a general color-design prompt instead, so expected outcome is not satisfied.

### 40. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 2/5** ❌ | Status: OK | 89.9 tok/s
> Status is OK but it did not provide the expected coherent explanation of main.swift; it only emitted a read_file-style tag and stopped, so it largely failed the test intent.

### 41. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 4/5** 👍 | Status: OK | 98.0 tok/s
> Response is coherent and correct for the prompt (lists three primary colors and relevant design importance), matching the expected quality baseline; minor issues are lack of concision and an unnecessary meta note about codebase context.

### 42. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 3/5** ⚠️ | Status: OK | 97.8 tok/s
> Output is coherent and on-topic, but it only states an intent to explore the codebase and does not provide the expected concrete code suggestion for adding the --timeout CLI flag.

### 43. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 2/5** ❌ | Status: OK | 97.8 tok/s
> Output is coherent, but it fails the test-specific expectation: spec says turn 3 should produce unit test code (baseline no-cache), while the model returned a general primary-colors design explanation.

### 44. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 2/5** ❌ | Status: OK | 107.7 tok/s
> Expected a unit test code response for the timeout feature, but the model only asked for missing context and did not provide any test implementation, so it fails the test’s desired outcome despite status=OK.

### 45. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 4/5** 👍 | Status: OK | 97.7 tok/s
> Content is correct, coherent, and fully answers the prompt (three primary colors + why they matter), so cache quality appears solid for turn 1; minor issues are lack of concision and an unnecessary codebase-related note.

### 46. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 2/5** ❌ | Status: OK | 101.5 tok/s
> Output did not fulfill the prompt (it only emitted a read_file tag and no explanation), so quality is poor and does not meet the expected turn-1 response quality even with prefix caching enabled.

### 47. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 4/5** 👍 | Status: OK | 95.7 tok/s
> Status is OK and the answer is correct/coherent for the prompt (three primary colors with clear design relevance), so quality is solid for cached turn 2; minor misses are lack of concision and an unnecessary codebase-related note. Cache-hit speed expectation cannot be fully verified from this single result alone.

### 48. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 2/5** ❌ | Status: OK | 115.3 tok/s
> Response quality is poor for turn 2: it only states intent to explore and stops without adding the --timeout flag or making progress, so it does not meet the expected no-cache-equivalent completion quality (cache speed benefits are not enough).

### 49. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 4/5** 👍 | Status: OK | 98.2 tok/s
> Status is OK and the response is correct/coherent for the prompt (lists red, blue, yellow with relevant design rationale), so quality is solid for cached turn 3; minor deduction because it is not very concise and adds an unnecessary codebase-related note.

### 50. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 122.1 tok/s
> Coherent but fails the test intent: instead of writing the requested timeout unit test for cached turn 3, it asks for missing context, so it does not meet the expected same-quality turn output (cache-hit behavior can’t be validated from content).

### 51. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 2/5** ❌ | Status: OK | 56.3 tok/s
> Did not meet the max_tokens truncation expectation: completion hit 50 with content empty and only partial reasoning text, but finish_reason is \

### 52. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 2/5** ❌ | Status: OK | 100.0 tok/s
> Hit 50 completion tokens with truncated reasoning-only output, but it failed the test expectation because finish_reason is \

### 53. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 2/5** ❌ | Status: OK | 114.2 tok/s
> Output is coherent and ends naturally with finish_reason=\

### 54. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 2/5** ❌ | Status: OK | 117.0 tok/s
> finish_reason is \

### 55. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 2/5** ❌ | Status: OK | 111.8 tok/s
> Status is OK, but this fails the test spec: expected output answer \

### 56. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 2/5** ❌ | Status: OK | 108.2 tok/s
> Content correctly answers \

### 57. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 4/5** 👍 | Status: OK | 112.7 tok/s
> Status is OK and the final content is coherent, correct, and not truncated/crashed under --max-kv-size 2048 as expected; minor issue is excessive reasoning output/token use despite a concise prompt.

### 58. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 5/5** ✅ | Status: OK | 122.7 tok/s
> Status is OK and the content delivers a clear, accurate multi-paragraph ML summary that matches the expected outcome; no truncation, crash, looping, or garbling despite the small KV setting.

### 59. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 4/5** 👍 | Status: OK | 111.2 tok/s
> Meets expected quality for kv-quantized output: final content is correct, coherent, and concise (names three colors and why they matter), with no obvious degradation; minor issue is very large reasoning_content/token usage despite a concise prompt.

### 60. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 5/5** ✅ | Status: OK | 123.1 tok/s
> Output is coherent, on-topic, and well-structured in a few paragraphs with accurate ML fundamentals; despite kv-cache quantization, quality appears strong and consistent with expected small-kv-level behavior.

### 61. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 2/5** ❌ | Status: OK | 115.4 tok/s
> Output content is correct but the test expected coherent architecture analysis; instead it answers an unrelated color prompt, and reasoning_content is extremely repetitive/looping (2610 tokens) for a concise task.

### 62. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 5/5** ✅ | Status: OK | 119.9 tok/s
> Status is OK and the content delivers a coherent, relevant top-3 architecture bottleneck analysis (GPU/KV cache, serialization/SSE, concurrency contention) that matches the test spec’s expected performance-focused outcome.

### 63. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 2/5** ❌ | Status: OK | 118.3 tok/s
> Final answer content is correct, but the run shows severe repetitive/looping reasoning spill (2610 completion tokens for a concise prompt), which indicates poor output quality versus expected normal prefill behavior.

### 64. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 4/5** 👍 | Status: OK | 121.3 tok/s
> Output is correct, coherent, and directly answers the top-3 bottlenecks with actionable investigation steps, matching expected quality; minor issue is extreme verbosity/token use (very long completion with repetitive reasoning), which is less ideal.

### 65. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 2/5** ❌ | Status: OK | 120.3 tok/s
> Final content is correct, but the run shows severe repetitive/looping reasoning (2610 completion tokens for a concise prompt), which is a significant quality issue despite status=OK; this does not meet the expected 'same output quality' behavior.

### 66. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 4/5** 👍 | Status: OK | 122.1 tok/s
> Status is OK and the main content is coherent, on-topic, and correctly identifies 3 high-impact bottlenecks as expected; quality is solid, but the response is excessively long with verbose/repetitive reasoning content, so not excellent.

### 67. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 2/5** ❌ | Status: OK | 119.1 tok/s
> Status is OK and content is coherent, but it fails the test spec’s expected outcome (short poem about the moon) and instead answers a different prompt; it also includes excessive reasoning_content/token use, indicating poor result quality for this case.

### 68. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 2/5** ❌ | Status: OK | 121.6 tok/s
> Expected a short coherent moon poem in the final output, but `content` is empty and `reasoning_content` is repetitive/looping internal drafting that exhausts max tokens instead of delivering the required response.

### 69. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 4/5** 👍 | Status: OK | 112.8 tok/s
> Raw-mode behavior is mostly correct: `<think>` text was preserved in `content` and not extracted (`reasoning_content` is empty), matching the passthrough expectation; minor issue is the response is extremely verbose/non-concise for the prompt.

### 70. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 2/5** ❌ | Status: OK | 121.2 tok/s
> Raw mode behavior is partly correct (<think> appears in content and reasoning_content is empty), but the output is excessively repetitive/looping, hits max_tokens (4096), and never cleanly delivers the expected concise step-by-step result to 391.

### 71. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 2/5** ❌ | Status: OK | 114.1 tok/s
> Stop-sequence behavior failed expectation: although finish_reason is \

### 72. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 4/5** 👍 | Status: OK | 83.6 tok/s
> finish_reason is \

### 73. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 4/5** 👍 | Status: OK | 115.3 tok/s
> Status is OK, finish_reason is \

### 74. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 2/5** ❌ | Status: OK | 108.3 tok/s
> Stop test partially failed: although finish_reason is \

### 75. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 124.8 tok/s
> Stop-newline behavior failed the test intent: `content` is empty (no one-line answer), while verbose `reasoning_content` contains many newline characters despite stop=[\

### 76. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 103.5 tok/s
> Expected a single-line answer in content with newline stop handling; instead content is empty and only multiline reasoning_content was produced, so the test objective was not met despite finish_reason=\

### 77. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 2/5** ❌ | Status: OK | 123.3 tok/s
> Stop-sequence behavior failed expectations: assistant content is empty (only verbose reasoning_content), and the output includes the stop string pattern (double newlines) instead of a single concise final paragraph.

### 78. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 2/5** ❌ | Status: OK | 128.0 tok/s
> Stop behavior failed the test intent: output contains double-newline-separated reasoning including a mountains paragraph, and no final ocean-only content paragraph was returned despite finish_reason=\

### 79. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 3/5** ⚠️ | Status: OK | 111.2 tok/s
> finish_reason is \

### 80. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 2/5** ❌ | Status: OK | 113.0 tok/s
> finish_reason is \

### 81. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 4/5** 👍 | Status: OK | 124.3 tok/s
> Meets the stop-period expectation in visible output: content is cut before the first period and finish_reason is \

### 82. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 5/5** ✅ | Status: OK | 124.7 tok/s
> Meets stop-period expectation: content is cut before the first period (only beginning of first sentence) and finish_reason is \

### 83. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 2/5** ❌ | Status: OK | 113.4 tok/s
> CLI stop appears to trigger (finish_reason=\

### 84. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 2/5** ❌ | Status: OK | 81.6 tok/s
> Stop-sequence test failed strict expectation: although content stops at items 1-2, the output includes the stop string \

### 85. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 4/5** 👍 | Status: OK | 114.8 tok/s
> Meets stop-sequence expectation: output does not contain either \

### 86. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 2/5** ❌ | Status: OK | 111.6 tok/s
> Stop handling likely triggered on \

### 87. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 2/5** ❌ | Status: OK | 112.8 tok/s
> Stop-merge expectation was not met cleanly: output includes the stop string \

### 88. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 2/5** ❌ | Status: OK | 86.7 tok/s
> Stop-merge behavior is only partially correct: visible content stops before \

### 89. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 2/5** ❌ | Status: OK | 115.0 tok/s
> Stop-dedup behavior is not clean: although status is OK and main content is coherent, the output includes the stop string \

### 90. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 5/5** ✅ | Status: OK | 91.1 tok/s
> Meets expected stop-dedup behavior: generation stopped cleanly before \

### 91. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 110.5 tok/s
> Stop-sequence behavior did not match the spec: expected numbered output truncated before \

### 92. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 119.8 tok/s
> Expected non-streaming stop truncation before \

### 93. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 112.8 tok/s
> Expected guided-JSON city output truncated at stop string \

### 94. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 5/5** ✅ | Status: OK | 93.9 tok/s
> Matches stop+guided-json expectation: status is OK, finish_reason is \

### 95. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 121.3 tok/s
> Fails the test’s expected outcome: with guided JSON + stop on comma, output should be truncated JSON (e.g., {\

### 96. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 112.5 tok/s
> Failed the stop+guided-json expectation: instead of truncated JSON before the first comma (e.g., {\

### 97. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 2/5** ❌ | Status: OK | 112.9 tok/s
> Expected guided JSON output truncated at the first \

### 98. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 2/5** ❌ | Status: OK | 119.8 tok/s
> Failed the stop+guided-json expectation: output is free-form prose (and reasoning) instead of guided JSON truncated at the first '}', so schema/stop behavior did not match the test spec.

### 99. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 2/5** ❌ | Status: OK | 115.5 tok/s
> Output is coherent JSON, but it does not show the expected stop behavior for json_object+stop (no truncation when the stop string \

### 100. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 2/5** ❌ | Status: OK | 120.9 tok/s
> Stop-sequence behavior failed expectation: output contains the stop string \

### 101. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 2/5** ❌ | Status: OK | 110.2 tok/s
> Result does not meet the test spec: expected a renewable-energy essay that stops before the phrase \

### 102. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 2/5** ❌ | Status: OK | 145.3 tok/s
> Stop-sequence expectation was violated because the output includes the stop string \

### 103. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 3/5** ⚠️ | Status: OK | 113.3 tok/s
> Finish reason is \

### 104. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 5/5** ✅ | Status: OK | 94.2 tok/s
> Meets expected stop behavior: visible content includes only Step 1-2 and halts before \

### 105. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 100.0 tok/s
> Stop no-match behavior looks correct (no \

### 106. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 116.9 tok/s
> Stop string was not triggered (good), but expected user-facing TCP vs UDP answer is missing: content is empty and the model used its full token budget in reasoning_content, ending without a proper final response.

### 107. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 123.0 tok/s
> Immediate-stop expectation was not met: despite finish_reason=\

### 108. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 125.6 tok/s
> Expected immediate stop with empty/near-empty completion_tokens, but model generated a long reasoning trace (767 tokens) before stopping, so it failed the stop-immediate expectation despite status OK and finish_reason=stop.

### 109. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 5/5** ✅ | Status: OK | 114.8 tok/s
> Meets stop-sequence expectation: visible content stops at \

### 110. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 5/5** ✅ | Status: OK | 105.3 tok/s
> Meets stop-sequence expectation: visible content cleanly stops before the first \

### 111. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 2/5** ❌ | Status: OK | 111.8 tok/s
> Did not meet the test’s expected stop-tag behavior: output lacks the required <ul>/<li> structure and does not demonstrate stopping at the first </li>, so the specific HTML stop-sequence feature was not validated.

### 112. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 5/5** ✅ | Status: OK | 80.9 tok/s
> Matches expected stop behavior: content includes `<ul>` and first `<li>` item text (`Apple`) and stops before emitting `</li>`, with `finish_reason`=`stop`.

### 113. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 4/5** 👍 | Status: OK | 113.6 tok/s
> Status is OK and output does not contain the Unicode stop character (•), so it aligns with the stop-sequence expectation; however, the very long reasoning output makes the result less clean for a concise prompt.

### 114. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 107.5 tok/s
> Stop-unicode behavior failed expectation: output includes the stop character \

### 115. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 123.0 tok/s
> Stop-sequence test failed expectations: generated output includes stop strings (e.g., \

### 116. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 123.2 tok/s
> Failed expected stop-behavior outcome: no user-facing content was produced, the model exhausted max tokens in reasoning, and the output includes stop strings (e.g., \

### 117. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 120.2 tok/s
> Poor stop-sequence behavior: the output contains the stop string \

### 118. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 103.7 tok/s
> Pirate-style content is on-topic, but this stop-sequence test is not cleanly met: the output payload includes the stop string \

### 119. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 2/5** ❌ | Status: OK | 117.4 tok/s
> Content is correctly formatted as a numbered 1-3 list, but the emitted reasoning is highly repetitive/looping and includes the stop string pattern (e.g., \

### 120. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 2/5** ❌ | Status: OK | 116.5 tok/s
> Although visible content matches the expected numbered list and stops before item 4, this stop-sequence test fails because the output payload includes the stop string \

### 121. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 119.6 tok/s
> Stop-sequence behavior does not meet the test intent: although finish_reason is \

### 122. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 5/5** ✅ | Status: OK | 82.3 tok/s
> Meets expected stop behavior at temperature=1.0: visible content stops before the stop string \

### 123. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 2/5** ❌ | Status: OK | 117.5 tok/s
> Fails the stop-sequence expectation from the test spec: output is not the expected numbered flower list, and the stop string \

### 124. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 5/5** ✅ | Status: OK | 89.3 tok/s
> Meets the stop+seed expectation for run 1: visible content is correctly truncated to items 1-2 and does not include the stop string \

### 125. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 2/5** ❌ | Status: OK | 116.9 tok/s
> Stop-sequence test failed expectation: the output includes the stop string \

### 126. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 4/5** 👍 | Status: OK | 89.7 tok/s
> Status is OK and content correctly stops before the stop string ('3.') with coherent output, which matches the stop-sequence expectation; however this single result cannot by itself confirm required run1/run2 identity, and the excessive reasoning trace is a minor quality issue.

### 127. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 2/5** ❌ | Status: OK | 69.4 tok/s
> Expected stop handling is not met: the output includes the stop string \

### 128. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 2/5** ❌ | Status: OK | 108.8 tok/s
> Stop-sequence test failed expectation: output includes the stop string \

### 129. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 2/5** ❌ | Status: OK | 116.0 tok/s
> Output is coherent but fails the test’s expected outcome: it did not return a valid JSON object with keys \

### 130. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 5/5** ✅ | Status: OK | 123.3 tok/s
> Output content is valid JSON and exactly matches expected keys/values (\

### 131. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 3/5** ⚠️ | Status: OK | 115.2 tok/s

### 132. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 3/5** ⚠️ | Status: OK | 122.5 tok/s

### 133. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 3/5** ⚠️ | Status: OK | 115.3 tok/s

### 134. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 3/5** ⚠️ | Status: OK | 119.5 tok/s

### 135. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 3/5** ⚠️ | Status: OK | 113.0 tok/s

### 136. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 3/5** ⚠️ | Status: OK | 121.0 tok/s

### 137. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 3/5** ⚠️ | Status: OK | 115.5 tok/s

### 138. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 3/5** ⚠️ | Status: OK | 122.5 tok/s

### 139. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 118.5 tok/s

### 140. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 121.2 tok/s

### 141. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 118.0 tok/s

### 142. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 122.4 tok/s

### 143. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 114.1 tok/s

### 144. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 121.7 tok/s

### 145. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 121.9 tok/s

### 146. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 115.1 tok/s

### 147. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 123.1 tok/s

### 148. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 117.6 tok/s

### 149. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 3/5** ⚠️ | Status: OK | 114.8 tok/s

### 150. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 3/5** ⚠️ | Status: OK | 118.4 tok/s

### 151. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 3/5** ⚠️ | Status: OK | 118.4 tok/s

### 152. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 3/5** ⚠️ | Status: OK | 120.5 tok/s

### 153. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 114.5 tok/s

### 154. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 114.8 tok/s

### 155. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 118.5 tok/s

### 156. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 112.4 tok/s

### 157. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 114.5 tok/s

### 158. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 121.1 tok/s

### 159. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 111.6 tok/s

### 160. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 112.4 tok/s

### 161. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 116.9 tok/s

### 162. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 114.3 tok/s

### 163. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 116.7 tok/s

### 164. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 119.4 tok/s

### 165. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 3/5** ⚠️ | Status: OK | 114.5 tok/s

### 166. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 3/5** ⚠️ | Status: OK | 120.2 tok/s

### 167. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 3/5** ⚠️ | Status: OK | 111.4 tok/s

### 168. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 3/5** ⚠️ | Status: OK | 118.2 tok/s

### 169. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 3/5** ⚠️ | Status: OK | 111.2 tok/s

### 170. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 3/5** ⚠️ | Status: OK | 80.2 tok/s

### 171. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 3/5** ⚠️ | Status: OK | 105.1 tok/s

### 172. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 3/5** ⚠️ | Status: OK | 113.1 tok/s

### 173. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 3/5** ⚠️ | Status: OK | 102.6 tok/s

### 174. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 3/5** ⚠️ | Status: OK | 97.4 tok/s

### 175. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 3/5** ⚠️ | Status: OK | 99.5 tok/s

### 176. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 3/5** ⚠️ | Status: OK | 110.3 tok/s

### 177. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 3/5** ⚠️ | Status: OK | 114.6 tok/s

### 178. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 3/5** ⚠️ | Status: OK | 116.8 tok/s

### 179. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 3/5** ⚠️ | Status: OK | 110.9 tok/s

### 180. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 3/5** ⚠️ | Status: OK | 121.0 tok/s

### 181. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 3/5** ⚠️ | Status: OK | 116.7 tok/s

### 182. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 3/5** ⚠️ | Status: OK | 113.8 tok/s

### 183. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 111.6 tok/s

### 184. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 117.0 tok/s

---
**Summary**: 50/185 passed (score ≥ 4), 75 failed (score ≤ 2)

<!-- AI_SCORES [{"i": 0, "s": 2}, {"i": 1, "s": 2}, {"i": 2, "s": 5}, {"i": 3, "s": 4}, {"i": 4, "s": 5}, {"i": 5, "s": 2}, {"i": 6, "s": 5}, {"i": 7, "s": 3}, {"i": 8, "s": 4}, {"i": 9, "s": 4}, {"i": 10, "s": 5}, {"i": 11, "s": 4}, {"i": 12, "s": 4}, {"i": 13, "s": 4}, {"i": 14, "s": 5}, {"i": 15, "s": 2}, {"i": 16, "s": 2}, {"i": 17, "s": 4}, {"i": 18, "s": 2}, {"i": 19, "s": 3}, {"i": 20, "s": 5}, {"i": 21, "s": 4}, {"i": 22, "s": 5}, {"i": 23, "s": 3}, {"i": 24, "s": 5}, {"i": 25, "s": 2}, {"i": 26, "s": 5}, {"i": 27, "s": 4}, {"i": 28, "s": 5}, {"i": 29, "s": 2}, {"i": 30, "s": 4}, {"i": 31, "s": 2}, {"i": 32, "s": 5}, {"i": 33, "s": 2}, {"i": 34, "s": 5}, {"i": 35, "s": 2}, {"i": 36, "s": 2}, {"i": 37, "s": 2}, {"i": 38, "s": 2}, {"i": 39, "s": 2}, {"i": 40, "s": 2}, {"i": 41, "s": 4}, {"i": 42, "s": 3}, {"i": 43, "s": 2}, {"i": 44, "s": 2}, {"i": 45, "s": 4}, {"i": 46, "s": 2}, {"i": 47, "s": 4}, {"i": 48, "s": 2}, {"i": 49, "s": 4}, {"i": 50, "s": 2}, {"i": 51, "s": 2}, {"i": 52, "s": 2}, {"i": 53, "s": 2}, {"i": 54, "s": 2}, {"i": 55, "s": 2}, {"i": 56, "s": 2}, {"i": 57, "s": 4}, {"i": 58, "s": 5}, {"i": 59, "s": 4}, {"i": 60, "s": 5}, {"i": 61, "s": 2}, {"i": 62, "s": 5}, {"i": 63, "s": 2}, {"i": 64, "s": 4}, {"i": 65, "s": 2}, {"i": 66, "s": 4}, {"i": 67, "s": 2}, {"i": 68, "s": 2}, {"i": 69, "s": 4}, {"i": 70, "s": 2}, {"i": 71, "s": 2}, {"i": 72, "s": 4}, {"i": 73, "s": 4}, {"i": 74, "s": 2}, {"i": 75, "s": 2}, {"i": 76, "s": 2}, {"i": 77, "s": 2}, {"i": 78, "s": 2}, {"i": 79, "s": 3}, {"i": 80, "s": 2}, {"i": 81, "s": 4}, {"i": 82, "s": 5}, {"i": 83, "s": 2}, {"i": 84, "s": 2}, {"i": 85, "s": 4}, {"i": 86, "s": 2}, {"i": 87, "s": 2}, {"i": 88, "s": 2}, {"i": 89, "s": 2}, {"i": 90, "s": 5}, {"i": 91, "s": 2}, {"i": 92, "s": 2}, {"i": 93, "s": 2}, {"i": 94, "s": 5}, {"i": 95, "s": 2}, {"i": 96, "s": 2}, {"i": 97, "s": 2}, {"i": 98, "s": 2}, {"i": 99, "s": 2}, {"i": 100, "s": 2}, {"i": 101, "s": 2}, {"i": 102, "s": 2}, {"i": 103, "s": 3}, {"i": 104, "s": 5}, {"i": 105, "s": 2}, {"i": 106, "s": 2}, {"i": 107, "s": 2}, {"i": 108, "s": 2}, {"i": 109, "s": 5}, {"i": 110, "s": 5}, {"i": 111, "s": 2}, {"i": 112, "s": 5}, {"i": 113, "s": 4}, {"i": 114, "s": 2}, {"i": 115, "s": 2}, {"i": 116, "s": 2}, {"i": 117, "s": 2}, {"i": 118, "s": 2}, {"i": 119, "s": 2}, {"i": 120, "s": 2}, {"i": 121, "s": 2}, {"i": 122, "s": 5}, {"i": 123, "s": 2}, {"i": 124, "s": 5}, {"i": 125, "s": 2}, {"i": 126, "s": 4}, {"i": 127, "s": 2}, {"i": 128, "s": 2}, {"i": 129, "s": 2}, {"i": 130, "s": 5}, {"i": 131, "s": 3}, {"i": 132, "s": 3}, {"i": 133, "s": 3}, {"i": 134, "s": 3}, {"i": 135, "s": 3}, {"i": 136, "s": 3}, {"i": 137, "s": 3}, {"i": 138, "s": 3}, {"i": 139, "s": 3}, {"i": 140, "s": 3}, {"i": 141, "s": 3}, {"i": 142, "s": 3}, {"i": 143, "s": 3}, {"i": 144, "s": 3}, {"i": 145, "s": 3}, {"i": 146, "s": 3}, {"i": 147, "s": 3}, {"i": 148, "s": 3}, {"i": 149, "s": 3}, {"i": 150, "s": 3}, {"i": 151, "s": 3}, {"i": 152, "s": 3}, {"i": 153, "s": 3}, {"i": 154, "s": 3}, {"i": 155, "s": 3}, {"i": 156, "s": 3}, {"i": 157, "s": 3}, {"i": 158, "s": 3}, {"i": 159, "s": 3}, {"i": 160, "s": 3}, {"i": 161, "s": 3}, {"i": 162, "s": 3}, {"i": 163, "s": 3}, {"i": 164, "s": 3}, {"i": 165, "s": 3}, {"i": 166, "s": 3}, {"i": 167, "s": 3}, {"i": 168, "s": 3}, {"i": 169, "s": 3}, {"i": 170, "s": 3}, {"i": 171, "s": 3}, {"i": 172, "s": 3}, {"i": 173, "s": 3}, {"i": 174, "s": 3}, {"i": 175, "s": 3}, {"i": 176, "s": 3}, {"i": 177, "s": 3}, {"i": 178, "s": 3}, {"i": 179, "s": 3}, {"i": 180, "s": 3}, {"i": 181, "s": 3}, {"i": 182, "s": 3}, {"i": 183, "s": 3}, {"i": 184, "s": 3}] -->
