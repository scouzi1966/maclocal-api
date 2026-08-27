import unittest

from mlx_model_test_oracle import (
    classify_result_likelihood,
    evaluate_expectations,
    evaluation_lane,
    expectation_for_prompt,
    extract_score_payload,
)


class ModelTestOracleTests(unittest.TestCase):
    def test_evaluation_lane_distinguishes_native_and_cross_family_parser_use(self):
        self.assertEqual(
            evaluation_lane(
                model="mlx-community/Qwen3.8-27B-4bit",
                afm_args="--tool-call-parser qwen3_xml",
            ),
            "native_protocol",
        )
        self.assertEqual(
            evaluation_lane(
                model="mlx-community/Muse-Glimmer-30B-4bit",
                afm_args="--tool-call-parser qwen3_xml",
            ),
            "forced_parser_compatibility",
        )
        self.assertEqual(
            evaluation_lane(model="test/model", has_expectation=False),
            "model_agent_behavior",
        )

    def test_failure_classification_is_likelihood_not_component_ownership(self):
        self.assertEqual(
            classify_result_likelihood(
                model="test/model",
                label="stop-single",
                status="OK",
                failures=["content mismatch"],
            ),
            "engine/runtime likely",
        )
        self.assertEqual(
            classify_result_likelihood(
                model="test/model",
                label="tool-call-auto",
                status="OK",
                failures=["tool call missing"],
            ),
            "parser/model boundary needs triage",
        )

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

    def test_accepts_distinct_tool_calls_in_either_order(self):
        expectation = {
            "finish_reason": "tool_calls",
            "tool_calls": [
                {"name": "get_weather", "arguments": {"city": "London"}},
                {"name": "get_time", "arguments": {"timezone": "Asia/Tokyo"}},
            ],
        }
        actual = [
            {"function": {"name": "get_time", "arguments": '{"timezone":"Asia/Tokyo"}'}},
            {"function": {"name": "get_weather", "arguments": '{"city":"London"}'}},
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

        self.assertIn("tool_calls missing expected function 'get_weather'", failures)
        self.assertTrue(any("arguments.timezone" in failure for failure in failures))

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

    def test_per_prompt_expectation_and_cache_telemetry(self):
        config = {
            "prompt_idx": 2,
            "num_baseline": 0,
            "expect_by_prompt": [
                {},
                {},
                {"cached_input_tokens_min": 10},
            ],
        }
        expectation = expectation_for_prompt(config)

        _, failures = evaluate_expectations(
            expectation,
            content="ok",
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
            cached_input_tokens=9,
        )

        self.assertEqual(expectation, {"cached_input_tokens_min": 10})
        self.assertEqual(
            failures,
            ["cached_input_tokens expected >= 10, got 9"],
        )


if __name__ == "__main__":
    unittest.main()
