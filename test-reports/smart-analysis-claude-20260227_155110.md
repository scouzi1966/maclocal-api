# Per-Test AI Analysis

### 0. 
**Score: 4/5** 👍 | Status:  | 0.0 tok/s
> Most tests pass well (sampling, system prompts, JSON, lists, essays, code, math all score 5). Two notable issues: (1) seed-42 limerick tests spend entire 4096 token budget in reasoning loops without producing content (score 2 each), and (2) several stop-sequence tests yield empty content due to stop strings firing before visible output on this thinking model (stop-newline, stop-double-newline, stop-multi, stop-immediate all have empty content). short-output max_tokens=50 also produces empty content. Tool call tests don't produce actual tool_calls with finish_reason=tool_calls. Overall the model generates high-quality responses at 80-120 tok/s but the thinking-model reasoning overhead causes failures in constrained-output scenarios.

### 1. mlx-community/Qwen3.5-35B-A3B-4bit @ greedy @ greedy
**Score: 3/5** ⚠️ | Status: OK | 109.0 tok/s

### 2. mlx-community/Qwen3.5-35B-A3B-4bit @ default @ default
**Score: 3/5** ⚠️ | Status: OK | 110.3 tok/s

### 3. mlx-community/Qwen3.5-35B-A3B-4bit @ high-temp @ high-temp
**Score: 3/5** ⚠️ | Status: OK | 114.3 tok/s

### 4. mlx-community/Qwen3.5-35B-A3B-4bit @ top-p @ top-p
**Score: 3/5** ⚠️ | Status: OK | 112.9 tok/s

### 5. mlx-community/Qwen3.5-35B-A3B-4bit @ top-k @ top-k
**Score: 3/5** ⚠️ | Status: OK | 108.5 tok/s

### 6. mlx-community/Qwen3.5-35B-A3B-4bit @ min-p @ min-p
**Score: 3/5** ⚠️ | Status: OK | 108.8 tok/s

### 7. mlx-community/Qwen3.5-35B-A3B-4bit @ combined-samplers @ combined-samplers
**Score: 3/5** ⚠️ | Status: OK | 110.5 tok/s

### 8. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run1 @ seed-42-run1
**Score: 3/5** ⚠️ | Status: OK | 103.2 tok/s

### 9. mlx-community/Qwen3.5-35B-A3B-4bit @ seed-42-run2 @ seed-42-run2
**Score: 3/5** ⚠️ | Status: OK | 113.2 tok/s

### 10. mlx-community/Qwen3.5-35B-A3B-4bit @ no-penalty @ no-penalty
**Score: 3/5** ⚠️ | Status: OK | 79.6 tok/s

### 11. mlx-community/Qwen3.5-35B-A3B-4bit @ with-penalty @ with-penalty
**Score: 3/5** ⚠️ | Status: OK | 90.4 tok/s

### 12. mlx-community/Qwen3.5-35B-A3B-4bit @ repetition-penalty @ repetition-penalty
**Score: 3/5** ⚠️ | Status: OK | 96.3 tok/s

### 13. mlx-community/Qwen3.5-35B-A3B-4bit @ pirate @ pirate
**Score: 3/5** ⚠️ | Status: OK | 117.1 tok/s

### 14. mlx-community/Qwen3.5-35B-A3B-4bit @ scientist @ scientist
**Score: 3/5** ⚠️ | Status: OK | 118.1 tok/s

### 15. mlx-community/Qwen3.5-35B-A3B-4bit @ eli5 @ eli5
**Score: 3/5** ⚠️ | Status: OK | 111.8 tok/s

### 16. mlx-community/Qwen3.5-35B-A3B-4bit @ json-output @ json-output
**Score: 3/5** ⚠️ | Status: OK | 104.5 tok/s

### 17. mlx-community/Qwen3.5-35B-A3B-4bit @ numbered-list @ numbered-list
**Score: 3/5** ⚠️ | Status: OK | 106.4 tok/s

### 18. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-simple @ guided-json-simple
**Score: 3/5** ⚠️ | Status: OK | 116.8 tok/s

### 19. mlx-community/Qwen3.5-35B-A3B-4bit @ guided-json-nested @ guided-json-nested
**Score: 3/5** ⚠️ | Status: OK | 115.8 tok/s

### 20. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn1 @ agent-no-cache-turn1
**Score: 3/5** ⚠️ | Status: OK | 55.0 tok/s

### 21. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn2 @ agent-no-cache-turn2
**Score: 3/5** ⚠️ | Status: OK | 68.6 tok/s

### 22. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-no-cache-turn3 @ agent-no-cache-turn3
**Score: 3/5** ⚠️ | Status: OK | 82.8 tok/s

### 23. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn1 @ agent-cached-turn1
**Score: 3/5** ⚠️ | Status: OK | 58.2 tok/s

### 24. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn2 @ agent-cached-turn2
**Score: 3/5** ⚠️ | Status: OK | 65.2 tok/s

### 25. mlx-community/Qwen3.5-35B-A3B-4bit @ agent-cached-turn3 @ agent-cached-turn3
**Score: 3/5** ⚠️ | Status: OK | 76.5 tok/s

### 26. mlx-community/Qwen3.5-35B-A3B-4bit @ short-output @ short-output
**Score: 3/5** ⚠️ | Status: OK | 33.2 tok/s

### 27. mlx-community/Qwen3.5-35B-A3B-4bit @ long-output @ long-output
**Score: 3/5** ⚠️ | Status: OK | 115.2 tok/s

### 28. mlx-community/Qwen3.5-35B-A3B-4bit @ logprobs @ logprobs
**Score: 3/5** ⚠️ | Status: OK | 85.7 tok/s

### 29. mlx-community/Qwen3.5-35B-A3B-4bit @ small-kv @ small-kv
**Score: 3/5** ⚠️ | Status: OK | 115.4 tok/s

### 30. mlx-community/Qwen3.5-35B-A3B-4bit @ kv-quantized @ kv-quantized
**Score: 3/5** ⚠️ | Status: OK | 118.7 tok/s

### 31. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-default @ prefill-default
**Score: 3/5** ⚠️ | Status: OK | 121.3 tok/s

### 32. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-large-4096 @ prefill-large-4096
**Score: 3/5** ⚠️ | Status: OK | 121.7 tok/s

### 33. mlx-community/Qwen3.5-35B-A3B-4bit @ prefill-small-256 @ prefill-small-256
**Score: 3/5** ⚠️ | Status: OK | 121.5 tok/s

### 34. mlx-community/Qwen3.5-35B-A3B-4bit @ no-streaming @ no-streaming
**Score: 3/5** ⚠️ | Status: OK | 120.5 tok/s

### 35. mlx-community/Qwen3.5-35B-A3B-4bit @ raw-mode @ raw-mode
**Score: 3/5** ⚠️ | Status: OK | 115.7 tok/s

### 36. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-single @ stop-single
**Score: 3/5** ⚠️ | Status: OK | 73.0 tok/s

### 37. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi @ stop-multi
**Score: 3/5** ⚠️ | Status: OK | 92.9 tok/s

### 38. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-newline @ stop-newline
**Score: 3/5** ⚠️ | Status: OK | 83.3 tok/s

### 39. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-double-newline @ stop-double-newline
**Score: 3/5** ⚠️ | Status: OK | 119.9 tok/s

### 40. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-word @ stop-word
**Score: 3/5** ⚠️ | Status: OK | 112.1 tok/s

### 41. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-period @ stop-period
**Score: 3/5** ⚠️ | Status: OK | 116.8 tok/s

### 42. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-only @ stop-cli-only
**Score: 3/5** ⚠️ | Status: OK | 70.2 tok/s

### 43. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-multi @ stop-cli-multi
**Score: 3/5** ⚠️ | Status: OK | 69.3 tok/s

### 44. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-merge @ stop-cli-api-merge
**Score: 3/5** ⚠️ | Status: OK | 77.3 tok/s

### 45. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-cli-api-dedup @ stop-cli-api-dedup
**Score: 3/5** ⚠️ | Status: OK | 79.7 tok/s

### 46. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-non-streaming @ stop-non-streaming
**Score: 3/5** ⚠️ | Status: OK | 121.9 tok/s

### 47. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-value @ stop-guided-json-value
**Score: 3/5** ⚠️ | Status: OK | 77.0 tok/s

### 48. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-comma @ stop-guided-json-comma
**Score: 3/5** ⚠️ | Status: OK | 113.0 tok/s

### 49. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-guided-json-brace @ stop-guided-json-brace
**Score: 3/5** ⚠️ | Status: OK | 117.5 tok/s

### 50. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-json-object-key @ stop-json-object-key
**Score: 3/5** ⚠️ | Status: OK | 122.1 tok/s

### 51. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-long-phrase @ stop-long-phrase
**Score: 3/5** ⚠️ | Status: OK | 149.6 tok/s

### 52. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-multi-word @ stop-multi-word
**Score: 3/5** ⚠️ | Status: OK | 90.2 tok/s

### 53. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-no-match @ stop-no-match
**Score: 3/5** ⚠️ | Status: OK | 101.3 tok/s

### 54. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-immediate @ stop-immediate
**Score: 3/5** ⚠️ | Status: OK | 113.6 tok/s

### 55. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-special-chars @ stop-special-chars
**Score: 3/5** ⚠️ | Status: OK | 100.8 tok/s

### 56. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-html-tag @ stop-html-tag
**Score: 3/5** ⚠️ | Status: OK | 74.7 tok/s

### 57. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-unicode @ stop-unicode
**Score: 3/5** ⚠️ | Status: OK | 97.9 tok/s

### 58. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-four-max @ stop-four-max
**Score: 3/5** ⚠️ | Status: OK | 121.3 tok/s

### 59. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-pirate @ stop-system-pirate
**Score: 3/5** ⚠️ | Status: OK | 60.1 tok/s

### 60. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-system-numbered @ stop-system-numbered
**Score: 3/5** ⚠️ | Status: OK | 89.3 tok/s

### 61. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-high-temp @ stop-high-temp
**Score: 3/5** ⚠️ | Status: OK | 65.7 tok/s

### 62. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run1 @ stop-seed-run1
**Score: 3/5** ⚠️ | Status: OK | 70.0 tok/s

### 63. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-seed-run2 @ stop-seed-run2
**Score: 3/5** ⚠️ | Status: OK | 69.5 tok/s

### 64. mlx-community/Qwen3.5-35B-A3B-4bit @ stop-low-max-tokens @ stop-low-max-tokens
**Score: 3/5** ⚠️ | Status: OK | 51.2 tok/s

### 65. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-json @ response-format-json
**Score: 3/5** ⚠️ | Status: OK | 106.8 tok/s

### 66. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-schema @ response-format-schema
**Score: 3/5** ⚠️ | Status: OK | 113.0 tok/s

### 67. mlx-community/Qwen3.5-35B-A3B-4bit @ response-format-text @ response-format-text
**Score: 3/5** ⚠️ | Status: OK | 97.0 tok/s

### 68. mlx-community/Qwen3.5-35B-A3B-4bit @ think-normal @ think-normal
**Score: 3/5** ⚠️ | Status: OK | 106.9 tok/s

### 69. mlx-community/Qwen3.5-35B-A3B-4bit @ think-raw @ think-raw
**Score: 3/5** ⚠️ | Status: OK | 104.3 tok/s

### 70. mlx-community/Qwen3.5-35B-A3B-4bit @ streaming-seeded @ streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 114.0 tok/s

### 71. mlx-community/Qwen3.5-35B-A3B-4bit @ non-streaming-seeded @ non-streaming-seeded
**Score: 3/5** ⚠️ | Status: OK | 114.4 tok/s

### 72. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 105.6 tok/s

### 73. mlx-community/Qwen3.5-35B-A3B-4bit @ max-completion-tokens @ max-completion-tokens
**Score: 3/5** ⚠️ | Status: OK | 114.2 tok/s

### 74. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 113.5 tok/s

### 75. mlx-community/Qwen3.5-35B-A3B-4bit @ developer-role @ developer-role
**Score: 3/5** ⚠️ | Status: OK | 117.2 tok/s

### 76. mlx-community/Qwen3.5-35B-A3B-4bit @ verbose @ verbose
**Score: 3/5** ⚠️ | Status: OK | 102.2 tok/s

### 77. mlx-community/Qwen3.5-35B-A3B-4bit @ very-verbose @ very-verbose
**Score: 3/5** ⚠️ | Status: OK | 98.4 tok/s

### 78. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 97.2 tok/s

### 79. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-auto @ tool-call-auto
**Score: 3/5** ⚠️ | Status: OK | 118.5 tok/s

### 80. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 99.5 tok/s

### 81. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-xml @ tool-call-xml
**Score: 3/5** ⚠️ | Status: OK | 117.3 tok/s

### 82. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 97.3 tok/s

### 83. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-multi @ tool-call-multi
**Score: 3/5** ⚠️ | Status: OK | 119.3 tok/s

### 84. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 99.5 tok/s

### 85. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-complex @ tool-call-complex
**Score: 3/5** ⚠️ | Status: OK | 117.6 tok/s

### 86. mlx-community/Qwen3.5-35B-A3B-4bit @ tool-call-none @ tool-call-none
**Score: 3/5** ⚠️ | Status: OK | 107.5 tok/s

### 87. mlx-community/Qwen3.5-35B-A3B-4bit @ minimal-prompt @ minimal-prompt
**Score: 3/5** ⚠️ | Status: OK | 95.3 tok/s

### 88. mlx-community/Qwen3.5-35B-A3B-4bit @ long-prompt @ long-prompt
**Score: 3/5** ⚠️ | Status: OK | 112.9 tok/s

### 89. mlx-community/Qwen3.5-35B-A3B-4bit @ special-chars @ special-chars
**Score: 3/5** ⚠️ | Status: OK | 116.5 tok/s

### 90. mlx-community/Qwen3.5-35B-A3B-4bit @ multilingual @ multilingual
**Score: 3/5** ⚠️ | Status: OK | 111.8 tok/s

### 91. mlx-community/Qwen3.5-35B-A3B-4bit @ code-python @ code-python
**Score: 3/5** ⚠️ | Status: OK | 91.7 tok/s

### 92. mlx-community/Qwen3.5-35B-A3B-4bit @ code-swift @ code-swift
**Score: 3/5** ⚠️ | Status: OK | 118.3 tok/s

### 93. mlx-community/Qwen3.5-35B-A3B-4bit @ math @ math
**Score: 3/5** ⚠️ | Status: OK | 118.0 tok/s

### 94. mlx-community/Qwen3.5-35B-A3B-4bit @ long-form @ long-form
**Score: 3/5** ⚠️ | Status: OK | 117.2 tok/s

### 95. mlx-community/Qwen3.5-35B-A3B-4bit @ strict-format @ strict-format
**Score: 3/5** ⚠️ | Status: OK | 101.5 tok/s

---
**Summary**: 1/96 passed (score ≥ 4), 0 failed (score ≤ 2)

<!-- AI_SCORES [{"i": 0, "s": 4}, {"i": 1, "s": 3}, {"i": 2, "s": 3}, {"i": 3, "s": 3}, {"i": 4, "s": 3}, {"i": 5, "s": 3}, {"i": 6, "s": 3}, {"i": 7, "s": 3}, {"i": 8, "s": 3}, {"i": 9, "s": 3}, {"i": 10, "s": 3}, {"i": 11, "s": 3}, {"i": 12, "s": 3}, {"i": 13, "s": 3}, {"i": 14, "s": 3}, {"i": 15, "s": 3}, {"i": 16, "s": 3}, {"i": 17, "s": 3}, {"i": 18, "s": 3}, {"i": 19, "s": 3}, {"i": 20, "s": 3}, {"i": 21, "s": 3}, {"i": 22, "s": 3}, {"i": 23, "s": 3}, {"i": 24, "s": 3}, {"i": 25, "s": 3}, {"i": 26, "s": 3}, {"i": 27, "s": 3}, {"i": 28, "s": 3}, {"i": 29, "s": 3}, {"i": 30, "s": 3}, {"i": 31, "s": 3}, {"i": 32, "s": 3}, {"i": 33, "s": 3}, {"i": 34, "s": 3}, {"i": 35, "s": 3}, {"i": 36, "s": 3}, {"i": 37, "s": 3}, {"i": 38, "s": 3}, {"i": 39, "s": 3}, {"i": 40, "s": 3}, {"i": 41, "s": 3}, {"i": 42, "s": 3}, {"i": 43, "s": 3}, {"i": 44, "s": 3}, {"i": 45, "s": 3}, {"i": 46, "s": 3}, {"i": 47, "s": 3}, {"i": 48, "s": 3}, {"i": 49, "s": 3}, {"i": 50, "s": 3}, {"i": 51, "s": 3}, {"i": 52, "s": 3}, {"i": 53, "s": 3}, {"i": 54, "s": 3}, {"i": 55, "s": 3}, {"i": 56, "s": 3}, {"i": 57, "s": 3}, {"i": 58, "s": 3}, {"i": 59, "s": 3}, {"i": 60, "s": 3}, {"i": 61, "s": 3}, {"i": 62, "s": 3}, {"i": 63, "s": 3}, {"i": 64, "s": 3}, {"i": 65, "s": 3}, {"i": 66, "s": 3}, {"i": 67, "s": 3}, {"i": 68, "s": 3}, {"i": 69, "s": 3}, {"i": 70, "s": 3}, {"i": 71, "s": 3}, {"i": 72, "s": 3}, {"i": 73, "s": 3}, {"i": 74, "s": 3}, {"i": 75, "s": 3}, {"i": 76, "s": 3}, {"i": 77, "s": 3}, {"i": 78, "s": 3}, {"i": 79, "s": 3}, {"i": 80, "s": 3}, {"i": 81, "s": 3}, {"i": 82, "s": 3}, {"i": 83, "s": 3}, {"i": 84, "s": 3}, {"i": 85, "s": 3}, {"i": 86, "s": 3}, {"i": 87, "s": 3}, {"i": 88, "s": 3}, {"i": 89, "s": 3}, {"i": 90, "s": 3}, {"i": 91, "s": 3}, {"i": 92, "s": 3}, {"i": 93, "s": 3}, {"i": 94, "s": 3}, {"i": 95, "s": 3}] -->
