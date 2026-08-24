import unittest

from mlx_model_test_oracle import (
    evaluate_expectations,
    expectation_for_prompt,
    extract_score_payload,
)


class ModelTestOracleTests(unittest.TestCase):
    def test_extracts_score_reason_with_escaped_quotes_from_cli_output(self):
        payload = extract_score_payload(
            'analysis noise\n{"score":5,"reason":"finish_reason=\\"tool_calls\\" is correct"}\n'
        )

        self.assertEqual(
            payload,
            {"score": 5, "reason": 'finish_reason="tool_calls" is correct'},
        )

    def test_variant_expectation_is_not_applied_to_global_baseline(self):
        config = {
            "prompt_idx": 0,
            "num_baseline": 1,
            "tools": [{"function": {"name": "get_weather"}}],
            "expect": {"tool_calls": [{"name": "get_weather"}]},
        }
        self.assertEqual(expectation_for_prompt(config), {"tool_calls": []})

        config["prompt_idx"] = 1
        self.assertEqual(expectation_for_prompt(config), config["expect"])

    def test_accepts_exact_ordered_distinct_tool_calls(self):
        expectation = {
            "finish_reason": "tool_calls",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "London"}},
                {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}},
            ],
        }
        actual = [
            {"function": {"name": "get_weather", "arguments": '{"city":"London"}'}},
            {"function": {"name": "get_time", "arguments": '{"timezone":"Asia/Tokyo"}'}},
        ]

        _, failures = evaluate_expectations(
            expectation,
            content="",
            finish_reason="tool_calls",
            logprobs_count=0,
            tool_calls=actual,
        )

        self.assertEqual(failures, [])

    def test_rejects_duplicate_substitution_and_wrong_argument(self):
        expectation = {
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "London"}},
                {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}},
            ]
        }
        actual = [
            {"function": {"name": "get_time", "arguments": '{"timezone":"Europe/London"}'}},
            {"function": {"name": "get_time", "arguments": '{"timezone":"Asia/Tokyo"}'}},
        ]

        _, failures = evaluate_expectations(
            expectation,
            content="",
            finish_reason="tool_calls",
            logprobs_count=0,
            tool_calls=actual,
        )

        self.assertTrue(any("tool_calls[0].name" in failure for failure in failures))
        self.assertTrue(any("arguments.city" in failure for failure in failures))

    def test_rejects_fenced_or_schema_invalid_json(self):
        expectation = {
            "valid_json": True,
            "json_schema": {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            },
        }

        valid_json, failures = evaluate_expectations(
            expectation,
            content='```json\n{"age":"37"}\n```',
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
        )

        self.assertFalse(valid_json)
        self.assertTrue(failures)

        valid_json, failures = evaluate_expectations(
            expectation,
            content='{"age":"37"}',
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
        )
        self.assertTrue(valid_json)
        self.assertIn("$.age expected type integer", failures)

    def test_exact_content_proves_deterministic_stop_prefix(self):
        expectation = {
            "finish_reason": "stop",
            "content_equals": "AFM_PREFIX",
            "content_not_contains": ["<AFM_STOP>", "AFM_SUFFIX"],
        }

        _, failures = evaluate_expectations(
            expectation,
            content="AFM_PREFIX",
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
        )
        self.assertEqual(failures, [])

        _, failures = evaluate_expectations(
            expectation,
            content="AFM_PREFIX AFM_SUFFIX",
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
        )
        self.assertTrue(any("content expected exactly" in failure for failure in failures))
        self.assertTrue(any("must not contain 'AFM_SUFFIX'" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
