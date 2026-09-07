"""CPU-only checks for separated integrity, cache, lexical, and semantic evidence."""
import contextlib
import importlib.util
import io
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
            self.assertIn("semantic=evaluated", output.getvalue())


if __name__ == "__main__":
    unittest.main()
