import json
import tempfile
import unittest
from pathlib import Path

from assemble_smart_report import assemble_per_test_report
from mlx_model_test_oracle import (
    classify_result_likelihood,
    evaluate_expectations,
    evaluation_lane,
    expectation_for_prompt,
    extract_score_payload,
    failed_run_records,
    skipped_run_records,
    transport_failure_record,
)


class ModelTestOracleTests(unittest.TestCase):
    def test_per_test_report_assembler_handles_metadata_skip_and_score_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.jsonl"
            scores = root / "scores"
            scores.mkdir()
            results.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"_meta": True},
                        {
                            "model": "model/a",
                            "label": "skipped",
                            "status": "SKIP",
                            "overall_status": "skip",
                            "skip_reason": "tools unavailable",
                        },
                        {
                            "model": "model/a",
                            "label": "working",
                            "status": "OK",
                            "tokens_per_sec": 12.34,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (scores / "score_1.txt").write_text(
                'noise\n{"score":5,"reason":"meets intent"}\n',
                encoding="utf-8",
            )

            report = assemble_per_test_report(scores, results)

        self.assertIn("### 0. model/a @ skipped", report)
        self.assertIn("> tools unavailable.", report)
        self.assertIn("### 1. model/a @ working", report)
        self.assertIn("**Score: 5/5**", report)
        self.assertIn("> meets intent", report)
        self.assertIn('<!-- AI_SCORES [{"i": 1, "s": 5}] -->', report)

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
        self.assertEqual(
            classify_result_likelihood(
                model="test/model",
                label="response-format-json",
                status="OK",
                failures=["invalid best-effort JSON"],
            ),
            "model behavior likely",
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

        _, failures = evaluate_expectations(
            {"cached_input_tokens_max": 0},
            content="ok",
            finish_reason="stop",
            logprobs_count=0,
            tool_calls=[],
            cached_input_tokens=1,
        )
        self.assertEqual(
            failures,
            ["cached_input_tokens expected <= 0, got 1"],
        )

    def test_server_load_failure_records_every_prompt_with_engine_metadata(self):
        records = failed_run_records(
            {
                "model": "test/model",
                "label": "tool-call-auto",
                "prompts": ["first", "second"],
                "temperature": 0.0,
                "max_tokens": 20,
                "system": "system",
                "instructions": "instructions",
                "requires": ["tools"],
                "afm_args": "--no-think",
                "media": ["fixture.png"],
                "stop": ["END"],
                "expect": {"tool_calls": [{"name": "weather"}]},
                "tools": [{"type": "function", "function": {"name": "weather"}}],
            },
            error="server died",
            load_time_s=7,
        )

        self.assertEqual([record["prompt"] for record in records], ["first", "second"])
        self.assertTrue(all(record["overall_status"] == "fail" for record in records))
        self.assertTrue(
            all(record["failure_classification"] == "engine/runtime likely" for record in records)
        )
        self.assertTrue(all(record["evaluation_lane"] == "native_protocol" for record in records))
        self.assertTrue(all(record["required_capabilities"] == ["tools"] for record in records))
        self.assertTrue(all(record["media"] == ["fixture.png"] for record in records))
        self.assertTrue(all(record["stop"] == ["END"] for record in records))
        self.assertTrue(all(record["tools"][0]["function"]["name"] == "weather" for record in records))
        self.assertTrue(all(record["expect"]["tool_calls"][0]["name"] == "weather" for record in records))

    def test_client_failure_for_baseline_is_always_native_protocol(self):
        record = transport_failure_record(
            {
                "model": "test/model",
                "label": "behavior-quality",
                "prompt_idx": 0,
                "num_baseline": 1,
                "temperature": 0.7,
                "max_tokens": 20,
                "tools": [{"type": "function", "function": {"name": "weather"}}],
                "expect_by_prompt": [{"content_equals": "unused"}],
            },
            prompt="ordinary baseline",
            error="client JSON invalid",
            load_time_s=2.5,
        )

        self.assertTrue(record["is_baseline"])
        self.assertEqual(record["transport_status"], "fail")
        self.assertEqual(record["assertion_status"], "not_run")
        self.assertEqual(record["evaluation_lane"], "native_protocol")
        self.assertEqual(record["failure_classification"], "engine/runtime likely")
        self.assertEqual(record["expect"], {"tool_calls": []})
        self.assertEqual(record["load_time_s"], 2.5)

    def test_capability_skip_preserves_per_prompt_configuration(self):
        records = skipped_run_records(
            {
                "model": "test/model",
                "label": "structured",
                "prompts": ["first", "second"],
                "temperature": 0.0,
                "max_tokens": 20,
                "response_format": {"type": "json_schema"},
                "media": ["fixture.png"],
                "expect_by_prompt": [
                    {"content_equals": "one"},
                    {"content_equals": "two"},
                ],
            },
            reason="structured unavailable",
        )

        self.assertEqual([record["expect"]["content_equals"] for record in records], ["one", "two"])
        self.assertTrue(all(record["transport_status"] == "not_run" for record in records))
        self.assertTrue(all(record["response_format"] == {"type": "json_schema"} for record in records))
        self.assertTrue(all(record["media"] == ["fixture.png"] for record in records))


if __name__ == "__main__":
    unittest.main()
