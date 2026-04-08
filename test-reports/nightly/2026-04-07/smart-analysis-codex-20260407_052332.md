# Per-Test AI Analysis

### 0. 
**Score: 4/5** 👍 | Status:  | 0.0 tok/s
> Correct and coherent baseline answer that names three primary colors and explains their design relevance concisely; minor downgrade because the model generated an unnecessarily long reasoning trace for such a simple prompt.

### 1. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 4/5** 👍 | Status: OK | 107.4 tok/s
> Content meets the expected outcome with exactly 3 clear, accurate bullet points explaining Rayleigh scattering, but the result is weakened by the large amount of separate reasoning_content for a simple prompt.

### 2. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 4/5** 👍 | Status: OK | 119.8 tok/s
> Correct, coherent, and concise visible answer that names red, blue, and yellow and explains their design relevance, but the run emitted excessive reasoning_content/token use for a simple concise prompt.

### 3. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 5/5** ✅ | Status: OK | 104.9 tok/s
> Accurate, coherent, and exactly 3 bullet points, which matches the default-sampling test expectation for a structured explanation of Rayleigh scattering.

### 4. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 5/5** ✅ | Status: OK | 115.1 tok/s
> Concise, coherent answer that names three primary colors and explains why they matter in design; it cleanly meets the expected outcome.

### 5. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 5/5** ✅ | Status: OK | 106.8 tok/s
> Scientifically accurate, coherent, and exactly 3 bullet points; for the high-temp test it stays on-topic and meets the expected outcome.

### 6. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 4/5** 👍 | Status: OK | 116.3 tok/s
> Correct, coherent, and concise in content: it names three primary colors and explains their design relevance. The main issue is the excessive reasoning_content/token usage for a prompt that explicitly asked for brevity.

### 7. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 5/5** ✅ | Status: OK | 112.5 tok/s
> Meets the expected outcome: the content is accurate, coherent, and formatted as exactly 3 bullet points explaining why the sky is blue.

### 8. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 3/5** ⚠️ | Status: OK | 118.7 tok/s
> Status OK and the final content correctly names three primary colors with a concise design-focused explanation, but the run produced an extremely long reasoning trace (3080 completion tokens) for a prompt that expected a concise answer, so it only partially meets the expected outcome.

### 9. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 5/5** ✅ | Status: OK | 116.6 tok/s
> Final content is correct, coherent, and matches the expected outcome exactly: 3 bullet points explaining Rayleigh scattering and why the sky appears blue.

### 10. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 5/5** ✅ | Status: OK | 119.3 tok/s
> Correct, concise response naming three primary colors and explaining their design relevance; coherent and fully meets the expected baseline outcome.

### 11. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 5/5** ✅ | Status: OK | 109.5 tok/s
> Content is accurate, coherent, and follows the expected outcome exactly with 3 clear bullet points explaining why the sky is blue.

### 12. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 4/5** 👍 | Status: OK | 118.4 tok/s
> Content is correct, coherent, and concise for the baseline prompt, but the run produced excessive reasoning_content despite the concise expected outcome.

### 13. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 5/5** ✅ | Status: OK | 110.5 tok/s
> status=OK and the content delivers an accurate explanation in exactly 3 bullet points, matching the test expectation cleanly.

### 14. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 5/5** ✅ | Status: OK | 108.3 tok/s
> Correct and concise answer that matches the prompt and expected baseline outcome; content is coherent and on-topic, with only extra reasoning_content present outside the final answer.

### 15. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 2/5** ❌ | Status: OK | 111.0 tok/s
> Expected a limerick about a cat, but content is empty and the model spent its full 4096-token budget in repetitive reasoning without producing the answer.

### 16. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 5/5** ✅ | Status: OK | 118.4 tok/s
> Correct, coherent, and concise answer to the baseline prompt; the final content clearly names three primary colors and explains their design relevance, so it meets the expected outcome for this seeded run.

### 17. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 2/5** ❌ | Status: OK | 112.7 tok/s
> Expected a creative limerick for the seed test, but the model produced no visible answer; it spent the full 4096-token budget in repetitive reasoning and ended with finish_reason=length, so it fails the expected outcome.

### 18. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 4/5** 👍 | Status: OK | 118.3 tok/s
> Correct, coherent, and concise final answer naming three primary colors and why they matter in design, which mostly meets the expected baseline outcome; reduced from excellent because it also generated unusually long reasoning_content and 1230 completion tokens for a simple concise prompt.

### 19. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 5/5** ✅ | Status: OK | 113.0 tok/s
> Long, coherent essay spanning major eras and civilizations with clear structure and no obvious repetition or degeneration, meeting the expected outcome for this no-penalty long-form prompt.

### 20. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 5/5** ✅ | Status: OK | 114.7 tok/s
> Correct, concise baseline answer that names three colors and explains their design importance clearly; it meets the expected outcome.

### 21. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 4/5** 👍 | Status: OK | 85.4 tok/s
> Meets the expected outcome with a long, coherent, well-structured essay spanning multiple civilizations and showing little repetition despite the presence penalty, but some historical claims are shaky and the large reasoning spill/token usage keeps it short of excellent.

### 22. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 5/5** ✅ | Status: OK | 85.7 tok/s
> Correct, concise baseline answer naming three primary colors and why they matter; coherent and non-repetitive, so it meets the expected outcome.

### 23. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 4/5** 👍 | Status: OK | 87.3 tok/s
> Long, coherent essay with reduced phrase-level repetition, so it mostly meets the repetition-penalty expectation, but it hits max_tokens with finish_reason=length, leaks extensive reasoning_content, and includes some shaky historical claims.

### 24. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 2/5** ❌ | Status: OK | 86.9 tok/s
> Expected a concise pirate answer naming three primary colors, but content was empty and the model exhausted max_tokens in repetitive reasoning without producing the final response.

### 25. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 4/5** 👍 | Status: OK | 115.9 tok/s
> Clear, on-topic pirate-speak explanation of qubits, superposition, entanglement, and decoherence; it meets the expected persona and content, with only a minor garbled artifact.

### 26. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 4/5** 👍 | Status: OK | 119.1 tok/s
> Final content is correct, concise, and technically precise for the baseline prompt, but the run also produced excessively long reasoning_content for a simple concise expected outcome.

### 27. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 5/5** ✅ | Status: OK | 114.1 tok/s
> Precise, coherent, technically detailed explanation that matches the physics-professor prompt and expected outcome with no notable issues.

### 28. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 2/5** ❌ | Status: OK | 115.6 tok/s
> Content is correct and concise, but the run is dominated by repetitive looping reasoning_content and excessive token use (3658/4096), which conflicts with the expected concise outcome.

### 29. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 5/5** ✅ | Status: OK | 111.5 tok/s
> Status OK and the content clearly explains quantum computing with simple, child-friendly analogies that match the expected ELI5 outcome; it is coherent, on-topic, and easy to follow.

### 30. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 5/5** ✅ | Status: OK | 116.6 tok/s
> Baseline under json-output may be unconstrained; content is correct, concise, and coherent, naming red/blue/yellow and briefly explaining their design importance.

### 31. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 5/5** ✅ | Status: OK | 110.1 tok/s
> Meets the json-output expectation: content is valid JSON with the required name, age, and city keys, so the test outcome is correct despite extra separate reasoning content.

### 32. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 5/5** ✅ | Status: OK | 119.2 tok/s
> Concise, coherent, and correct baseline answer: it names three primary colors and briefly explains their design relevance; baseline unconstrained text is acceptable here despite the numbered-list label.

### 33. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 5/5** ✅ | Status: OK | 109.3 tok/s
> Visible content exactly matches the expected 5 numbered animal lines with no extra output text; the strict numbered-list requirement is satisfied.

### 34. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 4/5** 👍 | Status: OK | 116.6 tok/s
> Because this is the [all] baseline under guided-json, plain text is acceptable here, and the content correctly names three primary colors and explains their design value concisely; however, the very long reasoning_content makes the overall result less clean than an excellent score.

### 35. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 2/5** ❌ | Status: OK | 109.6 tok/s
> guided-json-simple expected raw JSON matching name:string and age:integer, but the response is explanatory plain text with fenced JSON and wrong fields.

### 36. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 4/5** 👍 | Status: OK | 120.8 tok/s
> Baseline guided-json run can be plain text; the content correctly and concisely answers the primary-colors prompt, with a minor deduction for the very large reasoning_content/token use on a simple baseline case.

### 37. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 2/5** ❌ | Status: OK | 111.5 tok/s
> Guided-json test expected valid JSON matching the schema, but the model returned markdown/plain text instead, so it fails the required output format despite coherent Tokyo details.

### 38. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 4/5** 👍 | Status: OK | 120.0 tok/s
> Correctly names three primary colors and explains their design relevance, but it is less concise than requested and adds an unnecessary note unrelated to the prompt.

### 39. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 2/5** ❌ | Status: OK | 88.1 tok/s
> It recognized that it needed to read `Sources/MacLocalAPI/main.swift`, but this agent/tool-use turn should produce a proper tool call with `finish_reason=\

### 40. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 4/5** 👍 | Status: OK | 86.7 tok/s
> Response is correct and coherent for the prompt, naming three primary colors and explaining their design importance, but it is not concise and adds an unnecessary meta note about the codebase context.

### 41. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 2/5** ❌ | Status: OK | 87.7 tok/s
> Agent/tool-call test failed the expected outcome: it returned malformed tool-call text and finish_reason was \

### 42. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 4/5** 👍 | Status: OK | 88.5 tok/s
> Correct and coherent answer that names three primary colors and explains their design relevance, matching the prompt, but it is less concise than requested and ends with an irrelevant code-implementation note.

### 43. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 2/5** ❌ | Status: OK | 84.6 tok/s
> Did not write the requested unit test; instead it reported missing context and emitted malformed tool-call markup with finish_reason=stop, so it fails the expected outcome despite status=OK.

### 44. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 4/5** 👍 | Status: OK | 101.2 tok/s
> Correctly names red, blue, and yellow and explains why they matter in design, but it is less concise than requested and adds an irrelevant coding-context note.

### 45. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 3/5** ⚠️ | Status: OK | 84.9 tok/s
> Status OK and it takes the correct first agent step by issuing a valid read_file call for Sources/MacLocalAPI/main.swift, but the test spec expected the same coherent Swift-file explanation quality as no-cache turn 1; this only partially meets that outcome because no explanation was produced.

### 46. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 4/5** 👍 | Status: OK | 95.0 tok/s
> Correctly names red, blue, and yellow and gives coherent design rationale, so it mostly meets the expected outcome; loses a point for being less concise than requested and for adding unnecessary meta commentary about the codebase.

### 47. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 2/5** ❌ | Status: OK | 84.9 tok/s
> Expected a valid tool call for this agentic coding prompt, but the reply stops with malformed pseudo-tool markup and finish_reason=\

### 48. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 4/5** 👍 | Status: OK | 110.0 tok/s
> Correctly names red, blue, and yellow and explains their design relevance coherently, but it is less concise than requested and adds an unnecessary codebase-related note.

### 49. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 88.0 tok/s
> Expected unit test code for the timeout feature on cached turn 3, but the response only asks for missing context and provides no test implementation, so it misses the test’s desired outcome despite status=OK.

### 50. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 117.9 tok/s
> Expected a concise answer naming three primary colors and why they matter, but the result produced no user-facing content, only truncated reasoning_content, and hit max_tokens with finish_reason=length.

### 51. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 2/5** ❌ | Status: OK | 43.7 tok/s
> Expected a concise user-facing history of Rome, but the run produced no content at all and spent its 50-token budget in reasoning before hitting length; per the scoring rules, empty content with non-empty reasoning_content near max_tokens is poor.

### 52. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 2/5** ❌ | Status: OK | 94.9 tok/s
> Expected a concise answer naming three colors, but content is empty; the model used its full 2000-token budget on reasoning and stopped at length without delivering the final response.

### 53. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 3/5** ⚠️ | Status: OK | 115.4 tok/s
> Detailed and coherent recipe content with ingredients and clear steps, but it is truncated mid-sentence near max_tokens and does not fully provide the requested complete steps, tips, and variations.

### 54. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 3/5** ⚠️ | Status: OK | 118.9 tok/s
> Content is correct, coherent, and concise for the baseline prompt, but this @ logprobs run returned logprobs_count=0 despite --max-logprobs 5, so the feature expectation is only partially met.

### 55. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 2/5** ❌ | Status: OK | 110.5 tok/s
> Answer content is correct (1+1=2), but this logprobs test expected logprobs_count > 0 with top-5 token probabilities and the result returned logprobs_count=0, so it fails the feature under test.

### 56. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 5/5** ✅ | Status: OK | 112.2 tok/s
> Correct, concise smoke-test answer naming red/blue/yellow and explaining their design relevance; it fully meets the expected outcome, though the hidden reasoning is unnecessarily long.

### 57. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 5/5** ✅ | Status: OK | 114.3 tok/s
> Clear, accurate, and well-structured summary in a few paragraphs; it directly matches the prompt and meets the expected baseline outcome without repetition or formatting issues.

### 58. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 4/5** 👍 | Status: OK | 119.2 tok/s
> Correct, coherent, and concise baseline answer naming three primary colors and why they matter in design; slight downgrade because it generated excessive reasoning_content for a simple prompt.

### 59. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 5/5** ✅ | Status: OK | 107.7 tok/s
> Clear, coherent three-paragraph summary that directly matches the prompt and covers the expected core ideas: definition, supervised/unsupervised/reinforcement learning, workflow, and key challenges.

### 60. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 5/5** ✅ | Status: OK | 115.6 tok/s
> Correct and concise baseline answer naming three primary colors and why they matter; it stays on-topic and handles the long unrelated prefill context well.

### 61. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 5/5** ✅ | Status: OK | 115.5 tok/s
> Correct, coherent, and architecture-specific: it identifies MLX/KV cache memory pressure, JSON/SSE overhead, and serial-access concurrency contention as the top bottlenecks, with actionable investigation steps that match the expected outcome.

### 62. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 5/5** ✅ | Status: OK | 118.1 tok/s
> Status OK and the final content is correct, coherent, and concise, meeting the expected baseline outcome even under the large-prefill context.

### 63. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 5/5** ✅ | Status: OK | 115.3 tok/s
> Correct, coherent, and architecture-specific top-3 bottlenecks with actionable investigation steps; fully meets the expected outcome for this prompt.

### 64. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 4/5** 👍 | Status: OK | 117.3 tok/s
> Correct, concise content that preserves expected output quality under prefill-small-256 despite the long system prompt, but the reasoning is unnecessarily verbose and repetitive, so it falls short of excellent.

### 65. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 5/5** ✅ | Status: OK | 111.5 tok/s
> Coherent and relevant top-3 architecture bottleneck analysis with concrete investigation steps; meets the prefill-small-256 expectation of the same output quality as the baseline.

### 66. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 4/5** 👍 | Status: OK | 117.9 tok/s
> Correct, coherent, and concise answer that mostly meets the expected outcome, but the very large reasoning_content/token use for a simple prompt keeps it from being excellent.

### 67. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 3/5** ⚠️ | Status: OK | 114.9 tok/s
> Final content is a coherent short poem about the moon and meets the expected no-streaming outcome, but the reasoning_content is massively repetitive and far too long for a simple prompt, which noticeably hurts overall quality.

### 68. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 2/5** ❌ | Status: OK | 118.2 tok/s
> Final answer is correct, but it badly misses the expected concise outcome by emitting a long, repetitive <think> trace before the answer.

### 69. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 4/5** 👍 | Status: OK | 114.6 tok/s
> Correctly answers 17 * 23 = 391 with a coherent step-by-step raw-mode response, but it is excessively verbose and includes a small arithmetic typo (\

### 70. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 5/5** ✅ | Status: OK | 114.3 tok/s
> Correct, concise answer that meets the prompt, and for this stop-sequence test the visible output does not include the stop string \

### 71. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 5/5** ✅ | Status: OK | 109.2 tok/s
> stop-single behaved as expected: the visible content is coherent, ends before the stop string \

### 72. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 5/5** ✅ | Status: OK | 82.6 tok/s
> Correct, concise answer that matches the prompt, and the stop-multi expectation is met because the output does not include either stop string (` ``` ` or `END`).

### 73. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 3/5** ⚠️ | Status: OK | 110.1 tok/s
> finish_reason is \

### 74. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 2/5** ❌ | Status: OK | 106.1 tok/s
> finish_reason is \

### 75. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 119.1 tok/s
> stop-newline expected a one-sentence answer without emitting the stop string, but content is empty and the model produced reasoning text with newlines instead of a usable final response.

### 76. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 98.9 tok/s
> Stop-sequence test did not produce the expected concise final answer: `content` is empty and only reasoning was emitted, so the run fails the intended visible-output outcome.

### 77. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 2/5** ❌ | Status: OK | 118.5 tok/s
> Stop-sequence test failed: the output contains the stop string \

### 78. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 4/5** 👍 | Status: OK | 125.1 tok/s
> Status OK; the answer is correct, coherent, and concise, and the stop-word test appears to pass because the generated output does not contain the stop string. Excessive reasoning/token use for a simple prompt keeps it from a 5.

### 79. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 3/5** ⚠️ | Status: OK | 109.9 tok/s
> finish_reason is \

### 80. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 2/5** ❌ | Status: OK | 113.0 tok/s
> Stop-sequence test failed expectations: the output includes the forbidden \

### 81. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 2/5** ❌ | Status: OK | 120.1 tok/s
> stop-period expected output to cut off before the first \

### 82. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 2/5** ❌ | Status: OK | 125.1 tok/s
> Content answer is correct, but this is a stop-sequence test and the output includes the stop string \

### 83. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 2/5** ❌ | Status: OK | 110.1 tok/s
> Expected a concise pirate-style answer naming three primary colors, but the model exhausted its 4096-token budget in reasoning, produced no content, and devolved into repetitive looping.

### 84. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 4/5** 👍 | Status: OK | 78.0 tok/s
> Correct, concise visible answer and the multi-stop test appears to pass because the output does not contain either stop string, but it spent an excessive number of tokens in reasoning for a simple prompt.

### 85. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 2/5** ❌ | Status: OK | 106.1 tok/s
> Stop-sequence test did not meet the expected outcome: visible content is empty, and the model only produced reasoning that references the stop strings instead of a bash script response.

### 86. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 2/5** ❌ | Status: OK | 109.1 tok/s
> Visible content is correct and concise, but this stop-sequence merge case expects the stop string not to appear in output; reasoning_content contains '3.' repeatedly, so it fails the stop expectation despite finish_reason='stop'.

### 87. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 5/5** ✅ | Status: OK | 107.5 tok/s
> Expected stop-merge behavior: API stop \

### 88. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 4/5** 👍 | Status: OK | 86.1 tok/s
> Correct, coherent answer, and the visible content does not contain the stop string \

### 89. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 5/5** ✅ | Status: OK | 110.2 tok/s
> CLI and API both provided stop \

### 90. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 2/5** ❌ | Status: OK | 88.9 tok/s
> Visible content is correct, but this stop-sequence run does not cleanly meet expected stop compliance: reasoning_content includes the stop string \

### 91. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 111.1 tok/s
> Content is empty while reasoning consumed the full 4096-token budget and became repetitive/looping, so the stop/list-format test did not produce the expected numbered output.

### 92. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 5/5** ✅ | Status: OK | 118.7 tok/s
> Baseline guided-json run can be plain text; this response is correct, concise, coherent, and does not include the stop string.

### 93. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 110.0 tok/s
> Expected stop-guided-json truncation before \

### 94. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 92.6 tok/s
> Baseline under guided-json need not be JSON, but the visible answer is truncated after 'red' and does not deliver the expected concise explanation; the stop sequence was respected, yet the usable output is poor.

### 95. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 115.5 tok/s
> Guided-json test expected a valid JSON object with required name/age/city fields, but the model returned plain text instead of schema-matching JSON; the stop string was avoided, but the core expected outcome failed.

### 96. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 118.1 tok/s
> It eventually gives the correct colors, but badly fails the expected concise outcome by producing a long repetitive self-editing monologue; this reads as looping/overgeneration rather than a solid answer.

### 97. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 2/5** ❌ | Status: OK | 105.0 tok/s
> Expected guided-json output matching the {color, hex} schema, but the model returned plain-text prose instead of valid JSON; content is coherent, yet it fails the core format requirement for this test.

### 98. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 5/5** ✅ | Status: OK | 115.9 tok/s
> Baseline prompt is answered correctly and coherently: it names red, blue, and yellow, explains their design relevance, produces valid JSON, and the stop string \

### 99. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 2/5** ❌ | Status: OK | 109.6 tok/s
> Expected a valid JSON object without the stop string, but it exhausted max tokens in repetitive reasoning, produced no content, returned invalid JSON, and included \

### 100. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 5/5** ✅ | Status: OK | 118.8 tok/s
> Correct, concise answer that meets the stop-long-phrase expectation: the generated text stays coherent and does not include the stop string.

### 101. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 5/5** ✅ | Status: OK | 110.3 tok/s
> Expected stop-long-phrase behavior: two coherent renewable-energy paragraphs only, finish_reason=stop, and output stops before \

### 102. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 4/5** 👍 | Status: OK | 155.3 tok/s
> Content is correct and concise enough, finish_reason is \

### 103. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 5/5** ✅ | Status: OK | 110.1 tok/s
> Meets the stop-sequence expectation: visible output is coherent, shows only Steps 1-2, omits the stop string \

### 104. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 2/5** ❌ | Status: OK | 96.6 tok/s
> Expected a concise normal answer for stop-no-match, but the model produced no user-facing content, spent the full 256-token budget in reasoning_content, and stopped by length; the stop string was not emitted, but the test outcome is still poor.

### 105. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 91.9 tok/s
> The stop-no-match test did not emit the stop string, but the model spent its full token budget in reasoning_content and returned no actual TCP-vs-UDP answer, so it misses the expected outcome.

### 106. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 116.4 tok/s
> Expected immediate stop with empty or near-empty output and completion_tokens near 0, but the model used 817 completion tokens in reasoning_content and produced no visible content; finish_reason=stop is correct, but the test intent was not met cleanly.

### 107. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 118.8 tok/s
> stop-immediate expected empty or near-empty output with finish_reason=\

### 108. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 123.9 tok/s
> Visible content is truncated to \

### 109. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 2/5** ❌ | Status: OK | 115.8 tok/s
> Stop-special-chars result is poor: surfaced content truncates after part of the first fact, so it does not deliver 3 moon facts, and the stop string \

### 110. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 5/5** ✅ | Status: OK | 101.4 tok/s
> Correct concise answer, finish_reason is 'stop', and the stop-sequence expectation is met because '</li>' does not appear in content or reasoning_content.

### 111. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 2/5** ❌ | Status: OK | 110.2 tok/s
> stop-html-tag expected output without the stop string \

### 112. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 5/5** ✅ | Status: OK | 85.8 tok/s
> Status is OK, content is correct and concise, and for this stop-unicode test the output does not contain the configured stop string (•), so it meets the expected outcome.

### 113. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 110.7 tok/s
> Expected a 5-item bullet list without emitting the stop string, but content is empty and reasoning leaks the stop character \

### 114. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 106.1 tok/s
> Stop handling appears to cut off before the stop string, but the visible content is only \

### 115. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 122.1 tok/s
> stop-four-max failed expectations: it should stop cleanly before the first stop string, but it consumed the full max_tokens in reasoning, included stop strings like 3./three, and produced no final content.

### 116. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 2/5** ❌ | Status: OK | 118.6 tok/s
> Status is OK, but the model exhausted the full 4096-token budget in repetitive reasoning, produced no final content, and included the stop string \

### 117. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 3/5** ⚠️ | Status: OK | 116.4 tok/s
> Stop test passed because visible content does not contain the stop string \

### 118. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 97.5 tok/s
> Content is correct and concise, but this is a stop-sequence test and the output still contains the stop string \

### 119. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 5/5** ✅ | Status: OK | 107.8 tok/s
> Meets the expected stop-plus-system behavior: visible content is a correct numbered list of exercise benefits, stops cleanly after item 3 before \

### 120. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 2/5** ❌ | Status: OK | 112.4 tok/s
> stop-high-temp expects output to stop before \

### 121. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 119.3 tok/s
> Expected a concise pirate-style answer naming three primary colors, but content is empty and the model exhausted the 4096-token budget in repetitive reasoning loops, so it failed the intended outcome despite status=OK.

### 122. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 84.5 tok/s
> Answer content is correct, but this stop-sequence test should not return the stop string; the reasoning_content includes \

### 123. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 5/5** ✅ | Status: OK | 112.2 tok/s
> Correctly returns a numbered flower list truncated before \

### 124. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 2/5** ❌ | Status: OK | 87.2 tok/s
> Coherent visible answer, but for stop-seed-run2 it leaks a very long reasoning trace, includes the stop string 3. in the output, and burns 1135 tokens on a concise prompt, so it misses the expected clean stop behavior.

### 125. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 2/5** ❌ | Status: OK | 112.5 tok/s
> Stop-sequence test should not emit the stop string, but reasoning_content includes '3.' and the visible answer is only a truncated two-item flower list.

### 126. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 2/5** ❌ | Status: OK | 87.5 tok/s
> Stop-vs-max_tokens expectation failed: reasoning_content includes the stop string \

### 127. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 2/5** ❌ | Status: OK | 70.8 tok/s
> Used the token budget on reasoning instead of producing the 10-line mountain list, and the output contains the stop string \

### 128. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 2/5** ❌ | Status: OK | 106.4 tok/s
> Expected a concise pirate-style answer, but user-facing content is empty and the model spent the full 4096-token budget in repetitive looping reasoning, so it only partially meets the test.

### 129. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 5/5** ✅ | Status: OK | 110.0 tok/s
> Valid JSON object matching the response_format requirement, with the requested keys and correct values for Python; this meets the test's expected outcome.

### 130. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 4/5** 👍 | Status: OK | 120.5 tok/s
> Baseline text under response_format is acceptable here, and the content correctly names three primary colors and explains their design relevance concisely; minor issue is the unusually large reasoning_content for a simple prompt.

### 131. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 2/5** ❌ | Status: OK | 108.0 tok/s
> response-format schema expected valid JSON matching the schema, but the model returned a plain-text character profile instead of JSON.

### 132. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 4/5** 👍 | Status: OK | 115.5 tok/s
> Status OK and the final content correctly names three primary colors and explains their design relevance concisely, which matches the expected baseline text outcome, but the run also produced excessive reasoning content for such a simple prompt.

### 133. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 2/5** ❌ | Status: OK | 109.2 tok/s
> Status OK but content is empty; the model used the full 4096-token budget on repetitive reasoning loops and never delivered the expected concise pirate answer, so this is a poor result.

### 134. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 5/5** ✅ | Status: OK | 116.9 tok/s
> Correct, concise response naming red, blue, and yellow and explaining their design relevance; it cleanly matches the prompt's expected smoke-test outcome.

### 135. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 5/5** ✅ | Status: OK | 110.5 tok/s
> Correct, coherent step-by-step solution ending at 150 miles; it fully matches the expected outcome for the prompt.

### 136. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 4/5** 👍 | Status: OK | 119.5 tok/s
> Meets the baseline expected outcome with a correct, coherent answer naming red, blue, and yellow and explaining their design relevance, but the exposed raw thinking and minor typo make it much less concise than the prompt requested.

### 137. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 5/5** ✅ | Status: OK | 110.8 tok/s
> Correct, coherent step-by-step solution reaching 150 miles; for a think-raw run the visible reasoning and final answer match the expected outcome, with only a trivial stray artifact.

### 138. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 4/5** 👍 | Status: OK | 120.4 tok/s
> Correct and coherent concise answer naming three primary colors with relevant design importance, so it mostly meets the expected outcome; score reduced because it generated an excessively long reasoning trace for a concise prompt.

### 139. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 4/5** 👍 | Status: OK | 113.7 tok/s
> Content meets the expected outcome: a coherent 4-line poem about the ocean. Reduced from 5 because the model generated an excessively long reasoning trace and very high completion token count for a simple prompt.

### 140. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 5/5** ✅ | Status: OK | 119.8 tok/s
> Status OK and the baseline content is concise, coherent, and on-spec: it names three primary colors and briefly explains their design relevance.

### 141. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 4/5** 👍 | Status: OK | 114.4 tok/s
> Correct, coherent 4-line ocean poem that meets the prompt, but it used an excessive amount of reasoning_content for such a simple request.

### 142. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 5/5** ✅ | Status: OK | 114.9 tok/s
> Correct, concise baseline answer that names three primary colors and clearly explains why they matter in design; status OK and finish_reason=stop.

### 143. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 2/5** ❌ | Status: OK | 105.6 tok/s
> Status OK, but this max_completion_tokens alias test failed: it answered the parameter line instead of the French Revolution prompt, used 1174 completion tokens, and ended with finish_reason=\

### 144. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 2/5** ❌ | Status: OK | 112.8 tok/s
> Content is coherent, but this max_completion_tokens alias test failed its expected outcome: the response ran to 2300 completion tokens instead of truncating around 100, and finish_reason was \

### 145. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 5/5** ✅ | Status: OK | 115.6 tok/s
> Correct, concise baseline answer naming red, blue, and yellow and explaining their design importance; it fully meets the expected outcome.

### 146. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 2/5** ❌ | Status: OK | 108.8 tok/s
> Expected a concise code-only reply, but the model produced no user-visible code and instead exhausted its token budget on repetitive internal reasoning, ending with finish_reason=length.

### 147. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 5/5** ✅ | Status: OK | 117.4 tok/s
> Correct Python string-reversal function using slicing, with coherent examples and explanation; fully meets the expected outcome for this prompt.

### 148. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 118.6 tok/s
> Correct concise answer, but this --verbose run should look like a normal non-verbose response and instead produces an excessively long, repetitive reasoning trace, so it only partially meets the expected outcome.

### 149. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 5/5** ✅ | Status: OK | 112.1 tok/s
> Friendly, coherent greeting that directly answers the casual prompt and offers help; content matches the expected conversational outcome, and the extra reasoning is consistent with the verbose variant.

### 150. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 5/5** ✅ | Status: OK | 114.1 tok/s
> Concise, coherent, and directly meets the prompt: it names three primary colors and explains their design relevance clearly; no major quality or format issues in the visible content.

### 151. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 5/5** ✅ | Status: OK | 111.7 tok/s
> Friendly, coherent, on-topic reply that matches the expected casual greeting outcome; content is correct and shows no repetition, formatting, or tool-call issues.

### 152. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 4/5** 👍 | Status: OK | 116.4 tok/s
> Baseline prompt is answered correctly and concisely with three primary colors and clear design relevance; no tool call is needed here, but the run shows unnecessary reasoning_content/token use (810 completion tokens) for a short expected answer.

### 153. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 2/5** ❌ | Status: OK | 108.5 tok/s
> Tool-call test missed the expected outcome: finish_reason was \

### 154. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 2/5** ❌ | Status: OK | 114.7 tok/s
> tool-call-auto expected finish_reason=\

### 155. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 4/5** 👍 | Status: OK | 117.7 tok/s
> Content is correct, coherent, and concise for the prompt, but this simple baseline answer is inefficient because it includes large reasoning_content and high token usage.

### 156. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 2/5** ❌ | Status: OK | 109.8 tok/s
> Poor: this tool-call-xml test is expected to return finish_reason=\

### 157. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 2/5** ❌ | Status: OK | 114.7 tok/s
> Expected an XML tool call with finish_reason=\

### 158. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 5/5** ✅ | Status: OK | 117.6 tok/s
> Baseline [all] prompt under tool-call-multi produced a coherent, concise answer naming red, blue, and yellow and explaining their design relevance, which matches the expected smoke-test outcome.

### 159. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 109.2 tok/s
> Expected two tool calls with finish_reason=\

### 160. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 2/5** ❌ | Status: OK | 113.0 tok/s
> Expected a multi-tool call for weather and time with finish_reason=\

### 161. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 5/5** ✅ | Status: OK | 118.2 tok/s
> Correct baseline answer naming red, blue, and yellow and explaining their design relevance concisely; because is_baseline=true, a normal text response is the expected outcome here.

### 162. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 2/5** ❌ | Status: OK | 105.0 tok/s
> Tool-call test failed the expected outcome: it returned explanatory text with finish_reason=\

### 163. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 2/5** ❌ | Status: OK | 110.3 tok/s
> Plain-text shell advice instead of the expected tool call; finish_reason is \

### 164. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 5/5** ✅ | Status: OK | 116.0 tok/s
> This baseline prompt should be judged on text quality here, and the answer is concise, coherent, and correct: it names three primary colors and clearly explains their design relevance without unnecessary tool use.

### 165. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-array-param @ tool-call-array-param
**Score: 2/5** ❌ | Status: OK | 108.2 tok/s
> Tool-call test failed its expected outcome: the model returned plain text with finish_reason=\

### 166. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-array-param @ tool-call-array-param
**Score: 2/5** ❌ | Status: OK | 112.7 tok/s
> Expected a tool call with finish_reason=tool_calls and valid array arguments, but the model returned plain text with finish_reason=stop; the checklist is coherent, but it fails the tool-call test objective.

### 167. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-array-param @ tool-call-array-param
**Score: 4/5** 👍 | Status: OK | 115.9 tok/s
> Baseline prompt was answered correctly and concisely, which matches the expected normal-text outcome for this tool-call variant; minor deduction for the very large reasoning_content on such a simple request.

### 168. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-nullable-param @ tool-call-nullable-param
**Score: 2/5** ❌ | Status: OK | 111.6 tok/s
> Tool-call test failed the expected outcome: it returned plain text with finish_reason=\

### 169. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-nullable-param @ tool-call-nullable-param
**Score: 2/5** ❌ | Status: OK | 109.5 tok/s
> Tool-call test failed its expected outcome: it returned plain text with finish_reason=\

### 170. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-nullable-param @ tool-call-nullable-param
**Score: 5/5** ✅ | Status: OK | 118.8 tok/s
> Correct, coherent, and concise plain-text answer with no tool call; it matches the tool-call-none expectation and fully answers the prompt.

### 171. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 5/5** ✅ | Status: OK | 109.6 tok/s
> Matches the expected no-tool outcome: it did not attempt a tool call, stopped normally, and gave a coherent, accurate limitation response for a real-time weather question.

### 172. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 4/5** 👍 | Status: OK | 118.4 tok/s
> Correct, coherent, and concise final answer names three primary colors and explains their design relevance as expected; minor downgrade because the run used excessive reasoning_content/tokens for a minimal concise prompt.

### 173. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 5/5** ✅ | Status: OK | 104.3 tok/s
> Simple greeting prompt produced a correct, coherent greeting and meets the expected minimal-prompt outcome; final content is clean and appropriate.

### 174. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 4/5** 👍 | Status: OK | 116.8 tok/s
> Correct, coherent, and concise response naming the primary colors and their design relevance, matching the smoke-test expectation; downgraded from 5 because it produced an unusually large reasoning trace for a simple prompt.

### 175. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 5/5** ✅ | Status: OK | 109.0 tok/s
> Content correctly repeats the sentence 8 times and ends with DONE, matching the long-prompt expectation without truncation or garbling.

### 176. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 4/5** 👍 | Status: OK | 117.5 tok/s
> Status OK and the visible content correctly names three primary colors with a concise, coherent design rationale, which meets the expected baseline outcome; minor deduction because it also generated excessive reasoning_content for a concise prompt.

### 177. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 2/5** ❌ | Status: OK | 109.5 tok/s
> Expected an exact repeat of the special-character string, but the model spent nearly its full token budget in reasoning_content and emitted no user-facing content.

### 178. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 5/5** ✅ | Status: OK | 113.1 tok/s
> Status OK; the content is correct, coherent, and concise, naming three primary colors and briefly explaining their importance in design through color mixing, impact, and visual hierarchy.

### 179. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 5/5** ✅ | Status: OK | 110.5 tok/s
> Status OK; content correctly translates the sentence into French, Spanish, Japanese, and Arabic and formats it as a numbered list, matching the expected outcome.

### 180. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 2/5** ❌ | Status: OK | 118.3 tok/s
> Exhausted the full token budget in repetitive reasoning, produced no final content, and failed to deliver the concise answer expected.

### 181. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 5/5** ✅ | Status: OK | 95.1 tok/s
> Returned a correct, clean Sieve of Eratosthenes function in Python; the visible content is code-only and matches the expected expert-code outcome.

### 182. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 2/5** ❌ | Status: OK | 111.5 tok/s
> Token budget was exhausted in repetitive reasoning; content is empty and it never delivered the expected concise answer about the three primary colors and their design relevance.

### 183. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 2/5** ❌ | Status: OK | 90.2 tok/s
> Expected a clean Swift code snippet, but content is empty and the model exhausted the full token budget in reasoning_content with repetitive looping text, so it only partially meets the test intent.

### 184. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 4/5** 👍 | Status: OK | 94.2 tok/s
> Correctly names three primary colors and explains their design relevance, meeting the prompt, but the overall result is not truly concise because it includes excessive reasoning_content for a simple answer.

### 185. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 5/5** ✅ | Status: OK | 109.1 tok/s
> Correct math, clear step-by-step work, and the final total of $21.27 matches the expected outcome.

### 186. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 5/5** ✅ | Status: OK | 119.6 tok/s
> Content is correct, concise, and directly explains why the primary colors matter in design, matching the expected outcome; the verbose reasoning is separate and the final answer itself is solid.

### 187. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 4/5** 👍 | Status: OK | 98.7 tok/s
> Detailed, coherent technical blog post that explains MoE mechanics, efficiency, and dense-model comparisons with concrete examples, but it has a minor inconsistency in the hypothetical active-parameter counts.

### 188. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 5/5** ✅ | Status: OK | 118.4 tok/s
> Correct, concise answer that names three primary colors and explains their design relevance; as a baseline under strict-format, plain text is expected and the content meets the prompt well.

### 189. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 5/5** ✅ | Status: OK | 88.9 tok/s
> Content exactly matches the strict-format expectation: three lines with apple, banana, cherry and no extra visible text, numbering, or punctuation.

### 190. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 109.7 tok/s

---
**Summary**: 106/191 passed (score ≥ 4), 74 failed (score ≤ 2)

<!-- AI_SCORES [{"i": 0, "s": 4}, {"i": 1, "s": 4}, {"i": 2, "s": 4}, {"i": 3, "s": 5}, {"i": 4, "s": 5}, {"i": 5, "s": 5}, {"i": 6, "s": 4}, {"i": 7, "s": 5}, {"i": 8, "s": 3}, {"i": 9, "s": 5}, {"i": 10, "s": 5}, {"i": 11, "s": 5}, {"i": 12, "s": 4}, {"i": 13, "s": 5}, {"i": 14, "s": 5}, {"i": 15, "s": 2}, {"i": 16, "s": 5}, {"i": 17, "s": 2}, {"i": 18, "s": 4}, {"i": 19, "s": 5}, {"i": 20, "s": 5}, {"i": 21, "s": 4}, {"i": 22, "s": 5}, {"i": 23, "s": 4}, {"i": 24, "s": 2}, {"i": 25, "s": 4}, {"i": 26, "s": 4}, {"i": 27, "s": 5}, {"i": 28, "s": 2}, {"i": 29, "s": 5}, {"i": 30, "s": 5}, {"i": 31, "s": 5}, {"i": 32, "s": 5}, {"i": 33, "s": 5}, {"i": 34, "s": 4}, {"i": 35, "s": 2}, {"i": 36, "s": 4}, {"i": 37, "s": 2}, {"i": 38, "s": 4}, {"i": 39, "s": 2}, {"i": 40, "s": 4}, {"i": 41, "s": 2}, {"i": 42, "s": 4}, {"i": 43, "s": 2}, {"i": 44, "s": 4}, {"i": 45, "s": 3}, {"i": 46, "s": 4}, {"i": 47, "s": 2}, {"i": 48, "s": 4}, {"i": 49, "s": 2}, {"i": 50, "s": 2}, {"i": 51, "s": 2}, {"i": 52, "s": 2}, {"i": 53, "s": 3}, {"i": 54, "s": 3}, {"i": 55, "s": 2}, {"i": 56, "s": 5}, {"i": 57, "s": 5}, {"i": 58, "s": 4}, {"i": 59, "s": 5}, {"i": 60, "s": 5}, {"i": 61, "s": 5}, {"i": 62, "s": 5}, {"i": 63, "s": 5}, {"i": 64, "s": 4}, {"i": 65, "s": 5}, {"i": 66, "s": 4}, {"i": 67, "s": 3}, {"i": 68, "s": 2}, {"i": 69, "s": 4}, {"i": 70, "s": 5}, {"i": 71, "s": 5}, {"i": 72, "s": 5}, {"i": 73, "s": 3}, {"i": 74, "s": 2}, {"i": 75, "s": 2}, {"i": 76, "s": 2}, {"i": 77, "s": 2}, {"i": 78, "s": 4}, {"i": 79, "s": 3}, {"i": 80, "s": 2}, {"i": 81, "s": 2}, {"i": 82, "s": 2}, {"i": 83, "s": 2}, {"i": 84, "s": 4}, {"i": 85, "s": 2}, {"i": 86, "s": 2}, {"i": 87, "s": 5}, {"i": 88, "s": 4}, {"i": 89, "s": 5}, {"i": 90, "s": 2}, {"i": 91, "s": 2}, {"i": 92, "s": 5}, {"i": 93, "s": 2}, {"i": 94, "s": 2}, {"i": 95, "s": 2}, {"i": 96, "s": 2}, {"i": 97, "s": 2}, {"i": 98, "s": 5}, {"i": 99, "s": 2}, {"i": 100, "s": 5}, {"i": 101, "s": 5}, {"i": 102, "s": 4}, {"i": 103, "s": 5}, {"i": 104, "s": 2}, {"i": 105, "s": 2}, {"i": 106, "s": 2}, {"i": 107, "s": 2}, {"i": 108, "s": 2}, {"i": 109, "s": 2}, {"i": 110, "s": 5}, {"i": 111, "s": 2}, {"i": 112, "s": 5}, {"i": 113, "s": 2}, {"i": 114, "s": 2}, {"i": 115, "s": 2}, {"i": 116, "s": 2}, {"i": 117, "s": 3}, {"i": 118, "s": 2}, {"i": 119, "s": 5}, {"i": 120, "s": 2}, {"i": 121, "s": 2}, {"i": 122, "s": 2}, {"i": 123, "s": 5}, {"i": 124, "s": 2}, {"i": 125, "s": 2}, {"i": 126, "s": 2}, {"i": 127, "s": 2}, {"i": 128, "s": 2}, {"i": 129, "s": 5}, {"i": 130, "s": 4}, {"i": 131, "s": 2}, {"i": 132, "s": 4}, {"i": 133, "s": 2}, {"i": 134, "s": 5}, {"i": 135, "s": 5}, {"i": 136, "s": 4}, {"i": 137, "s": 5}, {"i": 138, "s": 4}, {"i": 139, "s": 4}, {"i": 140, "s": 5}, {"i": 141, "s": 4}, {"i": 142, "s": 5}, {"i": 143, "s": 2}, {"i": 144, "s": 2}, {"i": 145, "s": 5}, {"i": 146, "s": 2}, {"i": 147, "s": 5}, {"i": 148, "s": 3}, {"i": 149, "s": 5}, {"i": 150, "s": 5}, {"i": 151, "s": 5}, {"i": 152, "s": 4}, {"i": 153, "s": 2}, {"i": 154, "s": 2}, {"i": 155, "s": 4}, {"i": 156, "s": 2}, {"i": 157, "s": 2}, {"i": 158, "s": 5}, {"i": 159, "s": 3}, {"i": 160, "s": 2}, {"i": 161, "s": 5}, {"i": 162, "s": 2}, {"i": 163, "s": 2}, {"i": 164, "s": 5}, {"i": 165, "s": 2}, {"i": 166, "s": 2}, {"i": 167, "s": 4}, {"i": 168, "s": 2}, {"i": 169, "s": 2}, {"i": 170, "s": 5}, {"i": 171, "s": 5}, {"i": 172, "s": 4}, {"i": 173, "s": 5}, {"i": 174, "s": 4}, {"i": 175, "s": 5}, {"i": 176, "s": 4}, {"i": 177, "s": 2}, {"i": 178, "s": 5}, {"i": 179, "s": 5}, {"i": 180, "s": 2}, {"i": 181, "s": 5}, {"i": 182, "s": 2}, {"i": 183, "s": 2}, {"i": 184, "s": 4}, {"i": 185, "s": 5}, {"i": 186, "s": 5}, {"i": 187, "s": 4}, {"i": 188, "s": 5}, {"i": 189, "s": 5}, {"i": 190, "s": 3}] -->
