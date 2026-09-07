"""CPU-only checks for separated integrity, cache, lexical, and semantic evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1] / "feature-mlx-concurrent-batch"
sys.path.insert(0, str(ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load("response_contracts")
prefix = load("validate_multiturn_prefix")


class AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class Content:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class StreamResponse:
    status = 200
    headers = {}
    raw_headers = ()

    def __init__(self, chunks):
        self.content = Content(chunks)

    def raise_for_status(self):
        pass


class ResponseContractTests(unittest.IsolatedAsyncioTestCase):
    def test_lexical_observation_does_not_define_quality_failure(self):
        evidence = contracts.observe_lexical(
            "He says I should stay. My voice is quiet.", ["said", "voice"]
        )
        self.assertFalse(evidence["ok"])
        self.assertEqual(evidence["classification"], "review_evidence")
        self.assertEqual(evidence["missing"], ["said"])
        self.assertEqual(evidence["observed"], ["voice"])

    def test_integrity_requires_complete_transport_and_output(self):
        result = contracts.evaluate_integrity_contract(
            dict(
                visible_text="\ufffd" * 6,
                completion_tokens=2,
                finish_reason=None,
                done_observed=False,
                parse_error_count=1,
            ),
            min_tokens=5,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["failures"],
            [
                "excess_replacement_characters",
                "completion_below_minimum",
                "missing_finish_reason",
                "missing_sse_done",
                "sender_parse_error",
            ],
        )

    def test_cache_contract_is_explicit_and_bounded(self):
        result = contracts.evaluate_cache_contract(
            dict(prompt_tokens=100, cached_tokens=90),
            dict(expected_prompt_tokens=100, minimum_cached_tokens=95),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["failures"], ["cached_tokens_below_minimum"])

        result = contracts.evaluate_cache_contract(
            dict(prompt_tokens=100, cached_tokens=95)
        )
        self.assertTrue(result["ok"])

    def test_integrity_accepts_nonempty_single_character_visible_output(self):
        result = contracts.evaluate_integrity_contract(
            dict(
                visible_text="Y",
                completion_tokens=1,
                finish_reason="stop",
                done_observed=True,
                parse_error_count=0,
            )
        )
        self.assertTrue(result["ok"])

    def test_malformed_token_metadata_is_a_deterministic_failure(self):
        result = contracts.evaluate_integrity_contract(
            dict(
                visible_text="visible",
                completion_tokens="many",
                finish_reason="stop",
                done_observed=True,
                parse_error_count="none",
            )
        )
        self.assertEqual(
            result["failures"],
            [
                "invalid_completion_token_metadata",
                "invalid_parse_error_metadata",
            ],
        )

    def test_negative_token_metadata_is_invalid(self):
        result = contracts.evaluate_integrity_contract(
            dict(
                visible_text="visible",
                completion_tokens=-1,
                finish_reason="stop",
                done_observed=True,
                parse_error_count=0,
            )
        )
        self.assertEqual(
            result["failures"],
            ["invalid_completion_token_metadata"],
        )

        result = contracts.evaluate_cache_contract(
            dict(prompt_tokens=-1, cached_tokens=-1)
        )
        self.assertEqual(
            result["failures"],
            [
                "invalid_prompt_token_metadata",
                "invalid_cached_token_metadata",
            ],
        )

        result = contracts.evaluate_cache_contract(
            dict(prompt_tokens="many", cached_tokens="none")
        )
        self.assertEqual(
            result["failures"],
            [
                "invalid_prompt_token_metadata",
                "invalid_cached_token_metadata",
            ],
        )

    async def test_explicit_cache_contract_participates_in_deterministic_score(self):
        conversation = dict(
            name="fixture",
            system="system",
            turns=[
                dict(
                    user="fixture",
                    expected=[],
                    cache_contract=dict(minimum_cached_tokens=11),
                )
            ],
        )
        response = dict(
            text="valid",
            visible_text="valid",
            reasoning_text="",
            combined_text="valid",
            completion_tokens=10,
            prompt_tokens=10,
            cached_tokens=10,
            pp_tok_s=10,
            tg_tok_s=10,
            ttft=0.1,
            wall_s=1,
            finish_reason="stop",
            done_observed=True,
            parse_error_count=0,
        )
        with patch.object(prefix, "send_request", AsyncMock(return_value=response)), \
             patch.object(
                 prefix.aiohttp,
                 "ClientSession",
                 return_value=AsyncContext(object()),
             ), contextlib.redirect_stdout(io.StringIO()) as output:
            passed, failed, rows = await prefix.run_batch(1, [conversation])

        self.assertEqual((passed, failed), (0, 1))
        self.assertEqual(rows[0]["status"], "CONTRACT_FAILURE")
        self.assertEqual(rows[0]["contract_failures"], ["cached_tokens_below_minimum"])
        self.assertIn("cached_tokens_below_minimum", output.getvalue())

    async def test_sender_separates_reasoning_visible_and_stream_completion(self):
        body = [
            b'data: {"choices":[{"delta":{"content":"visible","reasoning_content":"reason"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(StreamResponse(body))
            ),
            [],
            max_tokens=8,
        )
        self.assertEqual(result["text"], "visible")
        self.assertEqual(result["visible_text"], "visible")
        self.assertEqual(result["reasoning_text"], "reason")
        self.assertEqual(result["combined_text"], "visiblereason")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["done_observed"])
        self.assertEqual(result["parse_error_count"], 0)

    async def test_metadata_only_sse_chunk_without_choices_is_valid(self):
        body = [
            b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34,'
            b'"prompt_tokens_details":{"cached_tokens":7}}}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(StreamResponse(body))
            ),
            [],
            max_tokens=8,
        )
        self.assertEqual(result["parse_error_count"], 0)
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 34)
        self.assertEqual(result["cached_tokens"], 7)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["done_observed"])

    async def test_malformed_falsy_choices_remain_parse_errors(self):
        body = [
            b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34},"choices":null}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(StreamResponse(body))
            ),
            [],
            max_tokens=8,
        )
        self.assertEqual(result["parse_error_count"], 1)
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertEqual(result["completion_tokens"], 34)

    async def test_malformed_usage_container_is_deterministic_metadata_failure(self):
        body = [
            b'data: {"choices":[{"delta":{"content":"visible"}}]}\n',
            b'data: {"usage":null,"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(StreamResponse(body))
            ),
            [],
            max_tokens=8,
        )
        integrity = contracts.evaluate_integrity_contract(result)
        cache = contracts.evaluate_cache_contract(result)
        self.assertEqual(result["parse_error_count"], 0)
        self.assertEqual(
            integrity["failures"],
            ["invalid_completion_token_metadata"],
        )
        self.assertEqual(
            cache["failures"],
            [
                "invalid_prompt_token_metadata",
                "invalid_cached_token_metadata",
            ],
        )

    async def test_malformed_cached_token_details_are_deterministic(self):
        body = [
            b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34,'
            b'"prompt_tokens_details":null},'
            b'"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ]
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(StreamResponse(body))
            ),
            [],
            max_tokens=8,
        )
        cache = contracts.evaluate_cache_contract(result)
        self.assertEqual(result["parse_error_count"], 0)
        self.assertEqual(cache["failures"], ["invalid_cached_token_metadata"])

    async def test_invalid_utf8_is_a_sender_parse_error(self):
        result = await prefix.send_request(
            SimpleNamespace(
                post=lambda *_args, **_kwargs: AsyncContext(
                    StreamResponse([b"data: \xff\xfe\n", b"data: [DONE]\n"])
                )
            ),
            [],
            max_tokens=8,
        )
        self.assertEqual(result["visible_text"], "")
        self.assertEqual(result["parse_error_count"], 1)
        self.assertTrue(result["done_observed"])

    async def test_semantic_judge_requires_actual_json_booleans(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            judge.write_text(
                "import json\n"
                "print(json.dumps({'requirements': "
                "[{'id': 'setting', 'passed': 'false'}]}))\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}"},
            ):
                result = await contracts.evaluate_semantic_contract(
                    [{"role": "user", "content": "fixture"}],
                    {"visible_text": "response"},
                    [{"id": "setting", "description": "Establishes setting."}],
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("passed flag is not boolean", result["error"])

    async def test_semantic_judge_must_return_exactly_configured_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            judge.write_text(
                "import json\n"
                "print(json.dumps({'requirements': [\n"
                "    {'id': 'setting', 'passed': True},\n"
                "    {'id': 'unrequested', 'passed': True}\n"
                "]}))\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}"},
            ):
                result = await contracts.evaluate_semantic_contract(
                    [{"role": "user", "content": "fixture"}],
                    {"visible_text": "response"},
                    [{"id": "setting", "description": "Establishes setting."}],
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("unknown requirements: unrequested", result["error"])

    async def test_semantic_judge_receives_structured_multiturn_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            received = Path(directory) / "received.json"
            judge.write_text(
                "import json, sys\n"
                "payload = json.load(sys.stdin)\n"
                f"with open({json.dumps(str(received))}, 'w', encoding='utf-8') as output:\n"
                "    output.write(json.dumps(payload))\n"
                "print(json.dumps({'requirements': "
                "[{'id': 'continuity', 'passed': True}]}))\n",
                encoding="utf-8",
            )
            transcript = [
                {"role": "system", "content": "system context"},
                {"role": "user", "content": "first request"},
                {"role": "assistant", "content": "prior visible answer"},
                {"role": "user", "content": "continue that answer"},
            ]
            with patch.dict(
                "os.environ",
                {"AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}"},
            ):
                result = await contracts.evaluate_semantic_contract(
                    transcript,
                    {"visible_text": "current visible answer"},
                    [{"id": "continuity", "description": "Continues the story."}],
                )
                payload = json.loads(received.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["ok"])
        self.assertEqual(payload["messages"], transcript)
        self.assertEqual(payload["prompt"], "continue that answer")
        self.assertEqual(payload["response"], "current visible answer")

    async def test_hanging_semantic_judge_fails_closed_after_timeout(self):
        with patch.dict(
            "os.environ",
            {
                "AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} -c 'import time; time.sleep(1)'",
                "AFM_SEMANTIC_JUDGE_TIMEOUT_S": "0.05",
            },
        ):
            result = await contracts.evaluate_semantic_contract(
                [{"role": "user", "content": "fixture"}],
                {"visible_text": "response"},
                [{"id": "setting", "description": "Establishes setting."}],
            )

        self.assertEqual(result["status"], "error")
        self.assertIn("TimeoutError", result["error"])

    async def test_semantic_judge_output_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            judge.write_text("print('12345')\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}",
                    "AFM_SEMANTIC_JUDGE_MAX_OUTPUT_BYTES": "4",
                },
            ):
                result = await contracts.evaluate_semantic_contract(
                    [{"role": "user", "content": "fixture"}],
                    {"visible_text": "response"},
                    [{"id": "setting", "description": "Establishes setting."}],
                )

        self.assertEqual(result["status"], "error")
        self.assertIn("stdout exceeded the output limit", result["error"])

    async def test_external_semantic_judge_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            judge.write_text(
                "import json, sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(json.dumps({'requirements': [\n"
                "    {'id': 'setting', 'passed': False, 'evidence': 'fixture'}\n"
                "], 'summary': 'fixture judged'}))\n",
                encoding="utf-8",
            )
            conversation = dict(
                name="fixture",
                system="system",
                turns=[
                    dict(
                        user="fixture",
                        expected=[],
                        semantic_requirements=[
                            {"id": "setting", "description": "Establishes setting."}
                        ],
                    )
                ],
            )
            response = dict(
                text="valid visible response",
                visible_text="valid visible response",
                reasoning_text="",
                combined_text="valid visible response",
                completion_tokens=10,
                prompt_tokens=10,
                cached_tokens=0,
                pp_tok_s=10,
                tg_tok_s=10,
                ttft=0.1,
                wall_s=1,
                finish_reason="stop",
                done_observed=True,
                parse_error_count=0,
            )
            with patch.dict(
                "os.environ",
                {"AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}"},
            ), patch.object(
                prefix,
                "send_request",
                AsyncMock(return_value=response),
            ), patch.object(
                prefix.aiohttp,
                "ClientSession",
                return_value=AsyncContext(object()),
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                passed, failed, rows = await prefix.run_batch(1, [conversation])

            self.assertEqual((passed, failed), (1, 0))
            self.assertEqual(rows[0]["semantic_review"]["status"], "evaluated")
            self.assertFalse(rows[0]["semantic_review"]["ok"])
            self.assertTrue(rows[0]["ok"])
            self.assertIn("semantic=ok:false(setting:fail)", output.getvalue())

    async def test_review_evidence_is_visible_on_contract_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = Path(directory) / "semantic-judge.py"
            judge.write_text(
                "import json, sys\n"
                "json.load(sys.stdin)\n"
                "print(json.dumps({'requirements': "
                "[{'id': 'setting', 'passed': False}]}))\n",
                encoding="utf-8",
            )
            conversation = dict(
                name="fixture",
                system="system",
                turns=[
                    dict(
                        user="fixture",
                        expected=["needle"],
                        semantic_requirements=[
                            {"id": "setting", "description": "Establishes setting."}
                        ],
                    )
                ],
            )
            response = dict(
                text="other",
                visible_text="other",
                reasoning_text="needle only in reasoning",
                combined_text="otherneedle only in reasoning",
                completion_tokens=10,
                prompt_tokens=10,
                cached_tokens=0,
                pp_tok_s=10,
                tg_tok_s=10,
                ttft=0.1,
                wall_s=1,
                finish_reason="stop",
                done_observed=False,
                parse_error_count=0,
            )
            with patch.dict(
                "os.environ",
                {"AFM_SEMANTIC_JUDGE_COMMAND": f"{sys.executable} {judge}"},
            ), patch.object(
                prefix,
                "send_request",
                AsyncMock(return_value=response),
            ), patch.object(
                prefix.aiohttp,
                "ClientSession",
                return_value=AsyncContext(object()),
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                passed, failed, rows = await prefix.run_batch(1, [conversation])

            self.assertEqual((passed, failed), (0, 1))
            self.assertEqual(rows[0]["contract_failures"], ["missing_sse_done"])
            self.assertEqual(rows[0]["lexical_observation"]["missing"], ["needle"])
            self.assertIn("lexical-evidence-missing=['needle']", output.getvalue())
            self.assertIn("semantic=ok:false(setting:fail)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
