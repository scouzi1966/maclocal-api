#!/usr/bin/env python3
"""CPU-only SDK dispatch fixtures; no server or inference subprocess."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "feature-codex-optimize-api/test-openai-compat-evals.py"
SPEC = importlib.util.spec_from_file_location("sdk_eval", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def models(caps, name="exact-checkpoint"):
    return {"data": [{"id": name}], "models": [{"model": name, "capabilities": caps}]}


class CapabilityTests(unittest.TestCase):
    def run_suite(self, payload):
        reports = []
        checks = ("run_openai_python_nonstream", "run_openai_python_stream",
                  "run_openai_python_nonstream_logprobs", "run_openai_python_stream_logprobs")
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", [str(PATH), "--model", "exact-checkpoint"]))
            stack.enter_context(patch.object(MODULE, "http_get_json", return_value=payload))
            stack.enter_context(patch("shutil.which", return_value=None))
            stack.enter_context(patch.object(MODULE, "log"))
            stack.enter_context(patch.object(MODULE, "write_report", side_effect=lambda results, path: reports.extend(results)))
            mocks = []
            for name in checks:
                mock = stack.enter_context(patch.object(MODULE, name, return_value={
                    "name": name.removeprefix("run_"), "ok": True, "elapsed_s": 0,
                }))
                mock.__name__ = name
                mocks.append(mock)
            self.assertEqual(MODULE.main(), 0)
            return reports, [mock.call_count for mock in mocks]

    def test_dwarfstar_runs_core_sdk_and_records_two_skips(self):
        reports, calls = self.run_suite(models(["dwarfstar_runtime", "text", "streaming"]))
        self.assertEqual(calls, [1, 1, 0, 0])
        self.assertEqual([r["status"] for r in reports if "status" in r], ["SKIP", "SKIP"])
        self.assertEqual(len(reports), 5)

    def test_mlx_logprobs_runs_all_checks(self):
        reports, calls = self.run_suite(models(["mlx_runtime", "text", "logprobs"]))
        self.assertEqual(calls, [1, 1, 1, 1])
        self.assertFalse(any(r.get("status") == "SKIP" for r in reports))

    def test_capabilities_not_engine_name_determine_skip(self):
        _, calls = self.run_suite(models(["mlx_runtime", "text"]))
        self.assertEqual(calls, [1, 1, 0, 0])
        _, calls = self.run_suite(models(["dwarfstar_runtime", "logprobs"]))
        self.assertEqual(calls, [1, 1, 1, 1])

    def test_unknown_missing_and_malformed_metadata_do_not_skip(self):
        for payload in ({"data": [{"id": "plain-openai"}]}, models([]),
                        models("dwarfstar_runtime"), models([None]),
                        models(["third_party_runtime"])):
            with self.subTest(payload=payload):
                _, calls = self.run_suite(payload)
                self.assertEqual(calls, [1, 1, 1, 1])

    def test_exact_model_precedes_unrelated_capabilities(self):
        payload = models(["mlx_runtime", "logprobs"])
        payload["models"].append({"model": "other", "capabilities": ["dwarfstar_runtime"]})
        _, calls = self.run_suite(payload)
        self.assertEqual(calls, [1, 1, 1, 1])

    def test_alias_only_resolves_unambiguous_model(self):
        payload = models(["dwarfstar_runtime"], name="canonical-name")
        self.assertEqual(MODULE.advertised_capabilities(payload, "alias"), ["dwarfstar_runtime"])
        payload["models"].append({"model": "other", "capabilities": ["mlx_runtime"]})
        self.assertIsNone(MODULE.advertised_capabilities(payload, "alias"))

    def test_transport_failure_is_not_capability_skip(self):
        with patch.object(MODULE, "http_get_json", side_effect=OSError("connection refused")):
            with self.assertRaises(OSError):
                MODULE.run_models_probe("http://fixture/v1", "exact-checkpoint")


if __name__ == "__main__":
    unittest.main()
