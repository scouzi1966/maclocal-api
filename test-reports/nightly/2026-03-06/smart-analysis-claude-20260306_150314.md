# Per-Test AI Analysis

### 0. 
**Score: 5/5** ✅ | Status:  | 0.0 tok/s
> Exactly 3 bullet points explaining Rayleigh scattering, accurate and well-structured. Greedy decoding produced coherent, on-topic output.

### 1. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 5/5** ✅ | Status: OK | 79.0 tok/s
> Exactly 3 bullet points, scientifically accurate (Rayleigh scattering), coherent and concise. Meets all expectations.

### 2. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 5/5** ✅ | Status: OK | 103.7 tok/s
> Exactly 3 bullet points, scientifically accurate explanation of Rayleigh scattering, concise and well-structured. High temperature didn't degrade quality.

### 3. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 5/5** ✅ | Status: OK | 114.0 tok/s
> Exactly 3 bullet points explaining Rayleigh scattering clearly and correctly. Top-p sampling produced coherent, well-structured output meeting all expectations.

### 4. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 5/5** ✅ | Status: OK | 102.1 tok/s
> Exactly 3 bullet points, scientifically accurate (Rayleigh scattering), coherent and well-structured. top-k sampling produced focused output with no repetition or issues.

### 5. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 5/5** ✅ | Status: OK | 111.6 tok/s
> Correct, coherent 3 bullet points explaining Rayleigh scattering. min-p sampling produced focused, accurate output.

### 6. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 5/5** ✅ | Status: OK | 106.0 tok/s
> Exactly 3 bullet points, scientifically accurate explanation of Rayleigh scattering, coherent and concise. Combined samplers (top-k 50, min-p 0.03, top-p 0.95) worked correctly.

### 7. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 2/5** ❌ | Status: OK | 107.1 tok/s
> Content is empty; model spent entire 4096 token budget on reasoning_content without producing a final limerick. The reasoning shows endless failed drafting attempts without converging on an answer.

### 8. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 2/5** ❌ | Status: OK | 116.9 tok/s
> Content is empty; model spent entire 4096 token budget on reasoning_content without producing a final limerick. The reasoning shows endless failed drafting attempts in a loop without converging on an answer.

### 9. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 5/5** ✅ | Status: OK | 117.7 tok/s
> Excellent long-form essay covering breadmaking history across civilizations (Egypt, Rome, China, Americas, Medieval Europe, Industrial Revolution, modern artisan revival). Well-structured with clear sections, rich detail, and coherent narrative. Hit max_tokens (4096) with substantive content throughout. No repetition, garbled text, or off-topic drift.

### 10. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 5/5** ✅ | Status: OK | 117.1 tok/s
> Excellent long essay on bread history across civilizations with presence_penalty=1.5. Content is coherent, well-structured, covers diverse civilizations (Egypt, Mesopotamia, Rome, medieval Europe, India, China, Americas), no repetition despite high presence penalty, and fills the 4096 token budget meaningfully.

### 11. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 5/5** ✅ | Status: OK | 93.0 tok/s
> Excellent long essay on bread history across civilizations with repetition_penalty=1.2. Content is coherent, well-structured, globally diverse (Egypt, Rome, India, Ethiopia, Mesoamerica, China), and shows no repetitive loops — the penalty is working as intended. 3954 tokens generated near the 4096 max with natural conclusion.

### 12. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 5/5** ✅ | Status: OK | 93.3 tok/s
> Excellent pirate-speak response explaining quantum computing with accurate concepts (qubits, superposition, entanglement). Fully in character, coherent, and meets the system prompt expectation.

### 13. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 5/5** ✅ | Status: OK | 113.4 tok/s
> Excellent response: precise technical terminology throughout (Hilbert space, unitary evolution, decoherence, Born rule, BQP), well-structured with clear sections, matches physics professor persona perfectly, coherent and comprehensive coverage of quantum computing fundamentals, algorithms, hardware, and challenges.

### 14. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 5/5** ✅ | Status: OK | 116.5 tok/s
> Perfect ELI5 explanation: simple words, short sentences, great analogies (spinning coin, light switch), age-appropriate tone, and coherent structure matching the system prompt's requirement to explain as if user is 5 years old.

### 15. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 5/5** ✅ | Status: OK | 114.6 tok/s
> Valid JSON with correct keys (name, age, city), proper types, no extraneous text in content. Meets expected outcome perfectly.

### 16. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 5/5** ✅ | Status: OK | 105.1 tok/s
> Exactly 5 animals, numbered 1-5, one per line, no extra text. Perfect adherence to prompt.

### 17. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 2/5** ❌ | Status: OK | 109.4 tok/s
> guided-json expected output constrained to schema {name, age} but model produced free-form JSON with unrelated fields (id, first_name, last_name, email, etc.) wrapped in markdown. The prompt-injection approach failed to constrain output to the required schema.

### 18. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 3/5** ⚠️ | Status: OK | 115.9 tok/s
> guided-json test expects valid JSON output matching the schema (object with city, population, landmarks array), but model returned Markdown prose instead. Content is high-quality and meets the semantic request (3+ landmarks), but fails the structured JSON constraint.

### 19. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 5/5** ✅ | Status: OK | 113.4 tok/s
> Agent correctly uses read_file tool call in response to file-reading request, demonstrating proper tool use with the system prompt's available tools. Reasoning content shows step-by-step thinking as instructed.

### 20. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 3/5** ⚠️ | Status: OK | 45.3 tok/s
> Model reasons correctly and begins agentic workflow, but tool call XML is malformed (mismatched tags, wrong syntax). Content is relevant to the task but the tool invocation would fail in practice.

### 21. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 3/5** ⚠️ | Status: OK | 65.3 tok/s
> Model correctly recognizes missing conversation context (expected for no-cache turn3 test) and attempts tool use, but the tool call XML is malformed with mismatched tags. Content is coherent but doesn't produce a unit test.

### 22. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 5/5** ✅ | Status: OK | 78.2 tok/s
> Model correctly uses read_file tool call as expected for an agent/tool-use scenario, with appropriate reasoning. Content is coherent and on-task.

### 23. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 4/5** 👍 | Status: OK | 54.7 tok/s
> Model reasons correctly, plans appropriate first step (explore codebase), and emits a tool call. The tool call XML is slightly malformed (mismatched tags, function call syntax wrong) but the intent and approach are sound for an agent-cached multi-turn scenario.

### 24. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 3/5** ⚠️ | Status: OK | 66.7 tok/s
> Model correctly recognizes missing conversation context and attempts to use tools, but the tool call XML is malformed (mismatched tags, wrong format). Content is coherent and reasoning is sound, but doesn't fulfill the test's intent of exercising cached multi-turn agent behavior.

### 25. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 2/5** ❌ | Status: OK | 77.9 tok/s
> Model spent entire 50-token budget on reasoning/thinking content with no visible output. For a short-output test expecting a concise description, producing zero content is a poor result.

### 26. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 5/5** ✅ | Status: OK | 50.3 tok/s
> Long-output test: 1992/2000 tokens generated, well-structured detailed recipe with ingredients, steps, tips, and variations as requested. Content is coherent, well-formatted with markdown, and not repetitive.

### 27. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 2/5** ❌ | Status: OK | 116.0 tok/s
> Logprobs test expects logprobs data but logprobs_count=0 — no logprobs were returned despite --max-logprobs 5. Content itself is correct (1+1=2) but the test feature (logprobs) failed.

### 28. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 5/5** ✅ | Status: OK | 85.0 tok/s
> Excellent summary: well-structured 3 paragraphs covering ML definition, learning types, and practical applications. Coherent, accurate, no repetition. Reasoning shows thoughtful drafting process.

### 29. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 5/5** ✅ | Status: OK | 114.0 tok/s
> Excellent summary: well-structured 3 paragraphs covering definition, learning types, and challenges. Coherent, on-topic, no repetition. KV-quantized inference performed well at 113 tok/s.

### 30. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 5/5** ✅ | Status: OK | 113.4 tok/s
> Excellent response: well-structured top-3 bottleneck analysis with specific, relevant investigation steps covering GPU memory/KV cache, JSON serialization/SSE, and concurrency contention. Directly addresses the architecture described in the system prompt. 116 tok/s at 3652 tokens with clean stop.

### 31. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 5/5** ✅ | Status: OK | 116.5 tok/s
> Excellent response: well-structured top-3 bottleneck analysis with specific, relevant investigation steps covering GPU memory/KV cache, serialization/SSE, and concurrency contention. Coherent, on-topic, and demonstrates deep architectural understanding. 116.77 tok/s at 3652 tokens with prefill-step-size 4096.

### 32. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 5/5** ✅ | Status: OK | 116.8 tok/s
> Excellent response: well-structured top-3 bottleneck analysis with specific, relevant investigation steps covering GPU memory/KV cache, serialization/SSE, and concurrency contention. Coherent, on-topic, and demonstrates deep architectural understanding matching the system prompt context.

### 33. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 5/5** ✅ | Status: OK | 117.2 tok/s
> Correct, coherent short poem about the moon with nice imagery and rhyme. Reasoning shows extensive drafting process. Status OK, proper finish_reason.

### 34. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 4/5** 👍 | Status: OK | 115.4 tok/s
> Raw mode test: <think> tags appear in content (expected in raw mode since server doesn't extract them). Correct answer (391), clear step-by-step. Minor: verbose thinking block, but content is coherent and correct.

### 35. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 5/5** ✅ | Status: OK | 114.0 tok/s
> Stop sequence '3.' correctly triggered - output ends at '2. Banana' without containing '3.', finish_reason is 'stop' as expected.

### 36. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 2/5** ❌ | Status: OK | 68.0 tok/s
> Content is empty — model spent all 253 tokens in reasoning_content without producing visible output. The stop sequences ['```', 'END'] were never reached in actual content. Test expects visible code block + END, got nothing.

### 37. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 2/5** ❌ | Status: OK | 86.9 tok/s
> Content is empty — model spent all tokens in reasoning_content without producing visible output. The stop sequence '\\n' likely fired immediately on the first newline of visible content, so the answer never appeared. The test expects a one-sentence answer about Paris.

### 38. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 2/5** ❌ | Status: OK | 79.2 tok/s
> Content is empty — model spent all 742 tokens in reasoning_content without producing visible output. The stop sequence '\\n\\n' likely fired immediately on the first double newline of the actual response, truncating all visible content. Test intent was to verify stop sequence halts output at first double newline between paragraphs, but getting zero visible content indicates the stop fired too early (possibly on thinking output leaking or on the first line of response).

### 39. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 5/5** ✅ | Status: OK | 113.3 tok/s
> Stop sequence 'Python' correctly truncated the content right before the first mention of 'Python'. Content cuts off at '1. **' which is exactly where 'Python' would appear. Stop string is absent from content, finish_reason is 'stop', and the model correctly listed programming languages before being stopped.

### 40. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 2/5** ❌ | Status: OK | 107.5 tok/s
> Stop sequence test: stop=[\

### 41. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 5/5** ✅ | Status: OK | 113.0 tok/s
> Stop sequence '3.' correctly truncated output after item 2, content doesn't contain the stop string, finish_reason=stop as expected.

### 42. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 3/5** ⚠️ | Status: OK | 68.9 tok/s
> Status OK but content is empty — model spent all tokens in reasoning_content before hitting stop sequence. Stop sequences (```, DONE) appear correctly absent from output, but no visible content was produced.

### 43. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 5/5** ✅ | Status: OK | 66.0 tok/s
> Stop sequence test passed: CLI --stop \

### 44. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 5/5** ✅ | Status: OK | 74.5 tok/s
> Stop sequence '3.' correctly truncated output after city #2. Content is coherent, finish_reason='stop', and the stop string does not appear in content. CLI --stop and API stop were deduplicated properly.

### 45. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 2/5** ❌ | Status: OK | 75.8 tok/s
> Content is empty — model spent entire 4096 token budget in reasoning_content with repetitive looping (endless 'Wait, I'll check if...' cycles) and never produced visible output. Stop sequence '3.' was never tested against actual content.

### 46. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 2/5** ❌ | Status: OK | 118.0 tok/s
> Content is only '[\

### 47. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 2/5** ❌ | Status: OK | 76.8 tok/s
> Stop sequence ',' fired too early, truncating content to just 'Here is a fictional person profile for Alice' — the guided-json object was never produced. The model spent most tokens reasoning but the visible output is incomplete and doesn't contain the expected JSON profile.

### 48. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 2/5** ❌ | Status: OK | 111.2 tok/s
> Stop sequence test: output should be JSON matching the guided-json schema, but model produced freeform markdown text instead of a JSON object. The stop string '}' never triggered on a JSON brace because no JSON was generated. The guided-json constraint (prompt injection) failed to produce structured output.

### 49. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 2/5** ❌ | Status: OK | 115.3 tok/s
> Content is empty — model spent entire 4096 token budget in reasoning_content with severe repetitive looping ('Wait, I'll output/stop' repeated dozens of times). Never produced visible output. Stop sequence 'age' is moot since no content was emitted.

### 50. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 5/5** ✅ | Status: OK | 117.7 tok/s
> Stop sequence 'In conclusion' correctly truncated the output — content ends with paragraph 2 and does not contain the stop string. Well-written, coherent essay content.

### 51. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 5/5** ✅ | Status: OK | 148.6 tok/s
> Stop sequence 'Step 3' correctly truncated output after Step 2 — content does not contain 'Step 3', finish_reason is 'stop', and the visible content is coherent and on-topic.

### 52. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 2/5** ❌ | Status: OK | 88.9 tok/s
> Content is empty; model spent entire 256-token budget on reasoning_content without producing visible output. Stop sequence 'XYZZY_NEVER_MATCH' correctly didn't fire, but the test expects a substantive TCP/UDP explanation in content.

### 53. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 2/5** ❌ | Status: OK | 100.0 tok/s
> Stop sequence test: content is empty because model spent all tokens in reasoning, but content_preview shows 'Thinking Process:' which starts visible output with no stop hit. The real issue: stop=['The','I','A'] should have truncated visible content, but there IS no visible content - the model never exited <think> tags, so it spent its entire budget reasoning without producing output. Score 2 for empty content with stop test not meaningfully exercised.

### 54. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 2/5** ❌ | Status: OK | 117.3 tok/s
> Stop sequence '**' should have truncated output before any bold markdown appeared, but the content shows the model's visible output was cut mid-sentence at 'a rate of' — the stop fired correctly before '**3.8'. However, the content is incomplete and not useful as a response. The real issue is the model spent most tokens (558) on reasoning_content, leaving visible content as just a truncated fragment. Stop sequence worked correctly but the result is a poor, incomplete answer.

### 55. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 5/5** ✅ | Status: OK | 98.9 tok/s
> Stop sequence '</li>' correctly truncated output before the closing tag. Content shows valid HTML structure starting with <ul> and <li>Apple, stopping exactly at '</li>' without including it.

### 56. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 2/5** ❌ | Status: OK | 77.0 tok/s
> Content is empty — model spent entire token budget in reasoning (thinking) and never produced visible output with the bullet point. The stop sequence '•' appears in reasoning_content but not in visible content, which is empty. Test expected visible bullet-point list; got nothing.

### 57. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 2/5** ❌ | Status: OK | 96.8 tok/s
> Content is empty — model spent entire 4096 token budget on reasoning_content without producing visible output. The stop sequences ['3.', 'three', 'Third', 'III'] were meant to test truncation at the third item, but the model never got past thinking to generate actual content.

### 58. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 4/5** 👍 | Status: OK | 118.8 tok/s
> Pirate persona is well-executed and content is on-topic. Stop sequence 'Arrr' not present in output. However, content is very short (only 1 sentence) suggesting the stop sequence may have truncated a longer response early, though this is correct behavior.

### 59. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 2/5** ❌ | Status: OK | 97.8 tok/s
> Stop sequence '4.' should have truncated output after item 3, but content shows only 3 items which looks correct. However, checking the raw content: it ends at item 3 with '3. Boosts mental health and reduces stress.' — the stop fired correctly before '4.' appeared. Wait, re-reading: content has exactly 3 items and no '4.' present, so stop worked. But the reasoning_content contains '4.' multiple times — stop sequences should only apply to visible content, not reasoning. Content is correct, on-topic, uses numbered list as requested. Score 4 — stop sequence worked correctly, content is good but truncated earlier than natural completion.

### 60. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 2/5** ❌ | Status: OK | 112.9 tok/s
> Stop sequence test: output contains only 2 items but the stop string '3.' should have truncated before '3.' appears. The content '1. Horizon\\n2. Whisper' looks correctly truncated, however the reasoning_content shows the full list including '3.' multiple times. The visible content is properly truncated but only 2 items with 274 completion_tokens means the model spent almost all tokens reasoning. The stop fired correctly on visible content, so this is borderline - but 274 tokens for 2 words of output is very inefficient. Score 3 would apply for correct stop behavior, but the massive token waste on reasoning pushes to 2-3 range.

### 61. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 2/5** ❌ | Status: OK | 77.3 tok/s
> Stop sequence '3.' should have truncated output before item 3, but content only shows 2 items (Rose, Tulip) — stop fired correctly. However, the content is truncated at '2. Tulip' without the stop string appearing, which is correct stop behavior. Wait — re-reading: content is '1. Rose\\n2. Tulip' which means stop fired on '3.' correctly, truncating before it appeared. This is correct stop sequence behavior. Score 4.

### 62. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 2/5** ❌ | Status: OK | 80.3 tok/s
> Stop sequence '3.' should have truncated output before item 3, but content only shows 2 items (Rose, Tulip) — stop fired correctly. However, the visible content is only 2 flowers instead of being cut at '3.' in the numbered list. Actually, stop='3.' fired after '2. Tulip\\n' when '3.' appeared, which is correct behavior. But completion_tokens=329 suggests the model spent most tokens on reasoning_content, and visible content is minimal (only 2 items). The stop sequence worked correctly (content doesn't contain '3.'), but the model wasted its budget on thinking. Score 3 for correct stop behavior but minimal visible output.

### 63. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 2/5** ❌ | Status: OK | 80.2 tok/s
> Model spent entire 100-token budget in reasoning_content with no visible content output. Stop sequence '2.' never triggered in visible content since none was produced. The test expects a numbered list where generation stops at '2.' — instead the model got stuck thinking.

### 64. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 5/5** ✅ | Status: OK | 76.4 tok/s
> Valid JSON with correct keys and accurate values. response_format json_object test passes: output is valid JSON, contains all required keys, factually correct.

### 65. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 2/5** ❌ | Status: OK | 118.1 tok/s
> response_format requested json_schema with a specific object schema (name/age/hobbies), but the model returned free-form markdown text instead of valid JSON. The content is high-quality prose but completely fails to meet the expected structured JSON output format.

### 66. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 5/5** ✅ | Status: OK | 118.5 tok/s
> Correct, concise response with accurate facts (Guido van Rossum, 1991). response_format=text properly returned plain text. finish_reason=stop as expected.

### 67. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 5/5** ✅ | Status: OK | 106.6 tok/s
> Correct answer (150 miles), clear step-by-step solution, reasoning_content properly extracted with think tags, finish_reason=stop as expected.

### 68. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 5/5** ✅ | Status: OK | 114.5 tok/s
> Raw mode correctly preserves <think> tags in content. Answer is correct (150 miles), step-by-step, coherent, and well-formatted.

### 69. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 5/5** ✅ | Status: OK | 114.5 tok/s
> Excellent 4-line poem about the ocean with ABAB rhyme scheme (shore/more, sand/land). Content is coherent, on-topic, and meets the expected outcome. Seed test produces valid output.

### 70. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 5/5** ✅ | Status: OK | 118.0 tok/s
> Correct 4-line poem about the ocean with ABAB rhyme scheme (shore/more, sand/land). Content is coherent, creative, and meets the expected outcome. Seed parameter accepted successfully.

### 71. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 2/5** ❌ | Status: OK | 117.2 tok/s
> max_completion_tokens was set to 100 but completion_tokens=1174, indicating the server failed to enforce the token limit. The model interpreted 'max_completion_tokens: 100' as a conversational instruction rather than the server capping output at 100 tokens.

### 72. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 5/5** ✅ | Status: OK | 116.9 tok/s
> Comprehensive, well-structured explanation of French Revolution causes and consequences. 2300 tokens generated within 4096 max_tokens limit, status OK, finish_reason=stop. Content is accurate, coherent, and well-organized with clear headings.

### 73. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 2/5** ❌ | Status: OK | 121.0 tok/s
> Content is empty; model spent entire 4096 token budget in reasoning_content which devolved into a repetitive loop ('Wait, I'll output/stop' repeated dozens of times). Developer role mapping worked (no error) but model failed to produce visible output.

### 74. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 5/5** ✅ | Status: OK | 119.5 tok/s
> Correct, well-structured Python function with multiple approaches. Developer role handled properly, clean output with reasoning.

### 75. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 5/5** ✅ | Status: OK | 120.9 tok/s
> Friendly, coherent greeting response with reasoning content showing thought process. Model loaded and generated successfully with good performance (104.9 tok/s).

### 76. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 5/5** ✅ | Status: OK | 104.9 tok/s
> Friendly, coherent greeting response with reasoning_content showing think tags extracted correctly. --very-verbose flag worked (verbose output). Content is natural and appropriate.

### 77. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 2/5** ❌ | Status: OK | 108.9 tok/s
> Tool call auto test expects finish_reason=\

### 78. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 2/5** ❌ | Status: OK | 100.7 tok/s
> Tool call test: expected finish_reason=\

### 79. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 2/5** ❌ | Status: OK | 121.6 tok/s
> Tool call test expects finish_reason=\

### 80. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 2/5** ❌ | Status: OK | 102.8 tok/s
> Tool call test expected finish_reason=\

### 81. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 2/5** ❌ | Status: OK | 121.0 tok/s
> Tool call test expects finish_reason=\

### 82. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 2/5** ❌ | Status: OK | 102.3 tok/s
> Tool call test expects finish_reason=\

### 83. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 2/5** ❌ | Status: OK | 123.4 tok/s
> Tool call test expects finish_reason=\

### 84. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 2/5** ❌ | Status: OK | 106.5 tok/s
> Tool call test: expected finish_reason=\

### 85. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 5/5** ✅ | Status: OK | 121.6 tok/s
> tool-call-none test: no tools provided, model correctly declined to make tool calls, finish_reason=stop, coherent response explaining inability to access real-time data

### 86. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 5/5** ✅ | Status: OK | 112.5 tok/s
> Minimal prompt 'Hi' produced a friendly, concise greeting response. Reasoning content shows coherent thought process. Clean finish_reason=stop.

### 87. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 5/5** ✅ | Status: OK | 106.2 tok/s
> Correctly repeated the phrase 8 times and ended with DONE as instructed. Content is exact and coherent.

### 88. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 2/5** ❌ | Status: OK | 116.5 tok/s
> Content is empty — model spent entire 4028-token budget on reasoning_content without producing any visible output. The special-chars test expects the model to actually repeat the characters in content, not just deliberate about how to do it.

### 89. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 5/5** ✅ | Status: OK | 119.0 tok/s
> All four translations are accurate and properly formatted as a numbered list. French, Spanish, Japanese, and Arabic are all correct with proper punctuation.

### 90. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 5/5** ✅ | Status: OK | 117.4 tok/s
> Correct Sieve of Eratosthenes implementation, clean Python code with no explanation text, meets test spec perfectly.

### 91. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 2/5** ❌ | Status: OK | 103.4 tok/s
> Model spent entire 32768 token budget in repetitive reasoning loop (checking if each Swift keyword is needed hundreds of times) and produced no actual content output. The reasoning contains correct Swift code drafts but the infinite loop prevented any code from being emitted.

### 92. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 5/5** ✅ | Status: OK | 96.2 tok/s
> Correct math ($21.27), clear step-by-step work as requested, proper discount application and tax calculation with appropriate rounding.

### 93. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 5/5** ✅ | Status: OK | 117.4 tok/s
> Excellent long-form technical blog post covering all requested topics (how MoE works, efficiency, comparison to dense models) with concrete examples (Mixtral 8x7B, Switch Transformer, DeepSeek), mathematical formulations, comparison tables, and well-structured sections. Content is coherent, detailed, and on-topic throughout 3743 tokens.

### 94. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 5/5** ✅ | Status: OK | 117.3 tok/s
> Output is exactly 3 lines with one word each (apple, banana, cherry), no numbering or punctuation. Perfectly meets the strict-format test spec.

### 95. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 103.8 tok/s

---
**Summary**: 53/96 passed (score ≥ 4), 37 failed (score ≤ 2)

<!-- AI_SCORES [{"i": 0, "s": 5}, {"i": 1, "s": 5}, {"i": 2, "s": 5}, {"i": 3, "s": 5}, {"i": 4, "s": 5}, {"i": 5, "s": 5}, {"i": 6, "s": 5}, {"i": 7, "s": 2}, {"i": 8, "s": 2}, {"i": 9, "s": 5}, {"i": 10, "s": 5}, {"i": 11, "s": 5}, {"i": 12, "s": 5}, {"i": 13, "s": 5}, {"i": 14, "s": 5}, {"i": 15, "s": 5}, {"i": 16, "s": 5}, {"i": 17, "s": 2}, {"i": 18, "s": 3}, {"i": 19, "s": 5}, {"i": 20, "s": 3}, {"i": 21, "s": 3}, {"i": 22, "s": 5}, {"i": 23, "s": 4}, {"i": 24, "s": 3}, {"i": 25, "s": 2}, {"i": 26, "s": 5}, {"i": 27, "s": 2}, {"i": 28, "s": 5}, {"i": 29, "s": 5}, {"i": 30, "s": 5}, {"i": 31, "s": 5}, {"i": 32, "s": 5}, {"i": 33, "s": 5}, {"i": 34, "s": 4}, {"i": 35, "s": 5}, {"i": 36, "s": 2}, {"i": 37, "s": 2}, {"i": 38, "s": 2}, {"i": 39, "s": 5}, {"i": 40, "s": 2}, {"i": 41, "s": 5}, {"i": 42, "s": 3}, {"i": 43, "s": 5}, {"i": 44, "s": 5}, {"i": 45, "s": 2}, {"i": 46, "s": 2}, {"i": 47, "s": 2}, {"i": 48, "s": 2}, {"i": 49, "s": 2}, {"i": 50, "s": 5}, {"i": 51, "s": 5}, {"i": 52, "s": 2}, {"i": 53, "s": 2}, {"i": 54, "s": 2}, {"i": 55, "s": 5}, {"i": 56, "s": 2}, {"i": 57, "s": 2}, {"i": 58, "s": 4}, {"i": 59, "s": 2}, {"i": 60, "s": 2}, {"i": 61, "s": 2}, {"i": 62, "s": 2}, {"i": 63, "s": 2}, {"i": 64, "s": 5}, {"i": 65, "s": 2}, {"i": 66, "s": 5}, {"i": 67, "s": 5}, {"i": 68, "s": 5}, {"i": 69, "s": 5}, {"i": 70, "s": 5}, {"i": 71, "s": 2}, {"i": 72, "s": 5}, {"i": 73, "s": 2}, {"i": 74, "s": 5}, {"i": 75, "s": 5}, {"i": 76, "s": 5}, {"i": 77, "s": 2}, {"i": 78, "s": 2}, {"i": 79, "s": 2}, {"i": 80, "s": 2}, {"i": 81, "s": 2}, {"i": 82, "s": 2}, {"i": 83, "s": 2}, {"i": 84, "s": 2}, {"i": 85, "s": 5}, {"i": 86, "s": 5}, {"i": 87, "s": 5}, {"i": 88, "s": 2}, {"i": 89, "s": 5}, {"i": 90, "s": 5}, {"i": 91, "s": 2}, {"i": 92, "s": 5}, {"i": 93, "s": 5}, {"i": 94, "s": 5}, {"i": 95, "s": 3}] -->
