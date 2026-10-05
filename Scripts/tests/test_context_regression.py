import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("gate", Path(__file__).parents[1] / "check-context-regression.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ContextRegressionTests(unittest.TestCase):
    def fixture(self, speed=100):
        metadata = dict(checkpoint="same", community_revision="a" * 40, binary_sha256="a" * 64,
                        config_sha256="b" * 64, server_arguments=["mlx", "--mtp"],
                        tuning_overrides={},
                        experiment=dict(trials_per_context=2, in_process_warm_context_pass=True,
                                        warm_marker_epoch="warm"))
        rows = [dict(context_size="2k", prompt_tokens=2048, generation_tokens=128,
                     generated_text="answer", generation_tps=speed, prompt_tps_e2e=1000)
                for _ in range(2)]
        texts = [dict(prompt_sha256=str(i), generated_text="answer") for i in range(2)]
        return metadata, rows, texts

    def check(self, old, new):
        with patch.object(gate, "load_run", side_effect=[old, new]):
            return gate.compare(Path("old"), Path("new"), 3)

    def test_equal_passes(self):
        self.assertTrue(self.check(self.fixture(), self.fixture())["passed"])

    def test_diagnostic_runs_rejected_on_either_side(self):
        for side in (0, 1):
            for nested in (False, True):
                pair = [self.fixture(), self.fixture()]
                metadata = pair[side][0]
                target = metadata["experiment"] if nested else metadata
                target["diagnostic_only"] = True
                with self.assertRaisesRegex(ValueError, "Diagnostic"):
                    self.check(*pair)

    def test_changed_checkpoint_config_rejected(self):
        old, new = self.fixture(), self.fixture()
        old[0]["config_sha256"] = "b" * 64
        new[0]["config_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.check(old, new)

    def test_missing_identity_cannot_match_another_missing_identity(self):
        for key in ("config_sha256", "binary_sha256", "server_arguments"):
            old, new = self.fixture(), self.fixture()
            del old[0][key]
            del new[0][key]
            with self.assertRaises(ValueError):
                self.check(old, new)

    def test_changed_server_settings_are_not_a_regression_control(self):
        new = self.fixture()
        new[0]["server_arguments"] = ["mlx", "--mtp", "--mtp-depth", "7"]
        with self.assertRaisesRegex(ValueError, "startup configuration"):
            self.check(self.fixture(), new)

    def test_tuning_overrides_rejected_even_when_identical(self):
        old, new = self.fixture(), self.fixture()
        for run in (old, new):
            run[0]["tuning_overrides"] = dict(verify_async_ladder=0)
        with self.assertRaisesRegex(ValueError, "Tuning overrides"):
            self.check(old, new)

    def test_missing_provenance_is_not_proof_of_clean_execution(self):
        for key in ("tuning_overrides", "community_revision"):
            old, new = self.fixture(), self.fixture()
            for run in (old, new):
                del run[0][key]
            with self.assertRaisesRegex(ValueError, "Missing"):
                self.check(old, new)
        for key in ("trials_per_context", "in_process_warm_context_pass", "warm_marker_epoch"):
            old, new = self.fixture(), self.fixture()
            for run in (old, new):
                del run[0]["experiment"][key]
            with self.assertRaisesRegex(ValueError, "schedule"):
                self.check(old, new)

    def test_local_checkpoint_requires_matching_manifest(self):
        old, new = self.fixture(), self.fixture()
        for run in (old, new):
            del run[0]["community_revision"]
            run[0]["checkpoint_manifest_sha256"] = "d" * 64
        self.assertTrue(self.check(old, new)["passed"])
        new[0]["checkpoint_manifest_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "manifests"):
            self.check(old, new)

    def test_twenty_percent_loss_fails(self):
        self.assertFalse(self.check(self.fixture(), self.fixture(80))["performance_passed"])

    def test_prefill_loss_fails_even_with_faster_decode(self):
        new = self.fixture(110)
        for row in new[1]:
            row["prompt_tps_e2e"] = 900
        self.assertFalse(self.check(self.fixture(), new)["passed"])

    def test_one_fast_trial_cannot_hide_regression(self):
        new = self.fixture(50)
        new[1][0]["generation_tps"] = 120
        self.assertFalse(self.check(self.fixture(), new)["performance_passed"])

    def test_output_change_requires_review(self):
        new = self.fixture()
        new[1][0]["generated_text"] = "different"
        result = self.check(self.fixture(), new)
        self.assertTrue(result["performance_passed"])
        self.assertFalse(result["passed"])

    def test_mismatched_workload_rejected(self):
        for location, key, value in [(0, "checkpoint", "other"), (0, "community_revision", "other")]:
            new = self.fixture()
            new[location][key] = value
            with self.assertRaises(ValueError):
                self.check(self.fixture(), new)
        new = self.fixture()
        new[1].pop()
        with self.assertRaises(ValueError):
            self.check(self.fixture(), new)

    def test_changed_warmup_rejected(self):
        new = copy.deepcopy(self.fixture())
        new[0]["experiment"]["in_process_warm_context_pass"] = False
        with self.assertRaises(ValueError):
            self.check(self.fixture(), new)

    def test_nan_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            gate.compare(Path("old"), Path("new"), float("nan"))

    def test_sampling_change_rejected(self):
        old, new = self.fixture(), self.fixture()
        old[0]["benchmark_arguments"] = ["--temperature", "0"]
        new[0]["benchmark_arguments"] = ["--temperature", "1"]
        with self.assertRaises(ValueError):
            self.check(old, new)

    def test_file_evidence_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta, rows, texts = self.fixture()
            usage = []
            for index, text in enumerate(texts):
                text["prompt"] = "prompt " + str(index)
                text["prompt_sha256"] = hashlib.sha256(text["prompt"].encode()).hexdigest()
                usage.append(dict(prompt_sha256=text["prompt_sha256"], usage=dict(
                    prompt_tokens=2048, prompt_tokens_details=dict(cached_tokens=0))))
            def save(name, value):
                (root / name).write_text(json.dumps(value))
            def save_lines(name, value):
                (root / name).write_text("\n".join(json.dumps(row) for row in value))
            save("metadata.json", meta)
            save("result.json", dict(status="completed"))
            command = ["python", "context.py", "output", "--contexts", "2", "--max-tokens", "128"]
            save("executed-test-command.json", command)
            save("server-command.json", ["/path/to/afm", "mlx", "--mtp"])
            save_lines("raw-trial-results.jsonl", rows)
            save_lines("paired-transcripts.jsonl", texts)
            save_lines("stream-usage.jsonl", usage)
            self.assertEqual(len(gate.load_run(root)[1]), 2)
            command[4] = "2,4"
            save("executed-test-command.json", command)
            with self.assertRaisesRegex(ValueError, "contexts"):
                gate.load_run(root)
            command[4] = "2"
            save("executed-test-command.json", command)
            usage[0]["usage"]["prompt_tokens_details"]["cached_tokens"] = 10
            save_lines("stream-usage.jsonl", usage)
            with self.assertRaisesRegex(ValueError, "cached"):
                gate.load_run(root)
            usage[0]["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            save_lines("stream-usage.jsonl", usage)
            rows[0]["generation_tps"] = float("nan")
            save_lines("raw-trial-results.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Invalid throughput"):
                gate.load_run(root)


if __name__ == "__main__":
    unittest.main()
