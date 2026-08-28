import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mlx_model_test_config import (
    ai_intent_for_result,
    capture_prompts_snapshot,
    expand_template_runs,
    materialize_verified_prompts_snapshot,
    parse_ai_intent_specs,
    parse_prompts_file,
    publish_report_atomically,
    results_metadata_declares_prompts,
)


class MLXModelTestConfigTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text(text, encoding="utf-8")
            return parse_prompts_file(path)

    def test_json_messages_decode_real_newlines_without_changing_literal_form(self):
        config = self.parse(
            'system: literal\\nseparator\n'
            '[@ decoded]\n'
            'system_json: "first\\nsecond"\n'
            'developer_json: "dev\\tmessage"\n'
            'instructions_json: "server\\ninstructions"\n'
            'Prompt\n'
        )

        self.assertEqual(config["defaults"]["system"], r"literal\nseparator")
        params = config["runs"][0]["params"]
        self.assertEqual(params["system"], "first\nsecond")
        self.assertEqual(params["developer"], "dev\tmessage")
        self.assertEqual(params["instructions"], "server\ninstructions")

    def test_duplicate_section_is_rejected_instead_of_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "duplicate section"):
            self.parse("[@ duplicate]\nOne\n[@ duplicate]\nTwo\n")

    def test_template_expansion_preserves_requirements(self):
        config = self.parse(
            "[@ structured]\n"
            "requires: structured, streaming\n"
            "Prompt\n"
        )

        expanded = expand_template_runs(config, ["org/one", "org/two"])

        self.assertEqual([run["model"] for run in expanded["runs"]], ["org/one", "org/two"])
        self.assertEqual(
            expanded["runs"][0]["params"]["requires"],
            ["structured", "streaming"],
        )

    def test_ai_intents_are_keyed_by_model_and_fall_back_to_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.txt"
            path.write_text(
                "# AI: model A intent\n"
                "[a/model @ shared]\nPrompt A\n"
                "# AI: model B intent\n"
                "[b/model @ shared]\nPrompt B\n"
                "# AI: template intent\n"
                "[@ common]\nPrompt template\n",
                encoding="utf-8",
            )
            specs = parse_ai_intent_specs(path)

        self.assertEqual(ai_intent_for_result(specs, "a/model", "shared"), ["model A intent"])
        self.assertEqual(ai_intent_for_result(specs, "b/model", "shared"), ["model B intent"])
        self.assertEqual(ai_intent_for_result(specs, "any/model", "common"), ["template intent"])

    def test_prompts_snapshot_is_recovered_after_digest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            snapshot = Path(directory) / "results.prompts.txt"
            snapshot.write_text("# AI: expected behavior\n", encoding="utf-8")
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "prompts_snapshot": snapshot.name,
                        "prompts_sha256": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            verified_copy = materialize_verified_prompts_snapshot(results)
            self.assertIsNotNone(verified_copy)
            self.assertEqual(verified_copy.read_bytes(), snapshot.read_bytes())
            self.assertEqual(verified_copy.parent, results.resolve().parent / "test-reports")
            verified_copy.unlink()

    def test_capture_prompts_snapshot_is_atomic_and_uses_captured_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.txt"
            results = Path(directory) / "run.jsonl"
            prompts.write_text("# AI: original intent\n", encoding="utf-8")

            snapshot, digest = capture_prompts_snapshot(prompts, results)
            prompts.write_text("# AI: changed later\n", encoding="utf-8")

            self.assertEqual(snapshot, Path(directory).resolve() / "run.prompts.txt")
            self.assertEqual(snapshot.read_text(encoding="utf-8"), "# AI: original intent\n")
            self.assertEqual(digest, hashlib.sha256(snapshot.read_bytes()).hexdigest())

    def test_capture_prompts_snapshot_rejects_destination_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.txt"
            results = Path(directory) / "run.jsonl"
            prompts.write_text("# AI: intent\n", encoding="utf-8")
            (Path(directory) / "run.prompts.txt").mkdir()

            with self.assertRaises(OSError):
                capture_prompts_snapshot(prompts, results)
            self.assertEqual(
                list(Path(directory).glob(".run.prompts.txt.*.tmp")),
                [],
            )

    def test_atomic_report_publish_rejects_destination_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_report = Path(directory) / ".report.tmp"
            report = Path(directory) / "report.md"
            temporary_report.write_text("complete report\n", encoding="utf-8")
            report.mkdir()

            with self.assertRaises(IsADirectoryError):
                publish_report_atomically(temporary_report, report)
            self.assertTrue(temporary_report.is_file())
            self.assertTrue(report.is_dir())

    def test_atomic_report_publish_replaces_file(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary_report = Path(directory) / ".report.tmp"
            report = Path(directory) / "report.md"
            temporary_report.write_text("complete report\n", encoding="utf-8")
            report.write_text("old report\n", encoding="utf-8")

            publish_report_atomically(temporary_report, report)

            self.assertFalse(temporary_report.exists())
            self.assertEqual(report.read_text(encoding="utf-8"), "complete report\n")

    def test_prompts_snapshot_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "prompts_snapshot": "../private.txt",
                        "prompts_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "must be a sibling"):
                materialize_verified_prompts_snapshot(results)

    def test_prompts_snapshot_rejects_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            snapshot = Path(directory) / "results.prompts.txt"
            snapshot.write_text("changed expectations\n", encoding="utf-8")
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "prompts_snapshot": snapshot.name,
                        "prompts_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                materialize_verified_prompts_snapshot(results)

    def test_prompts_snapshot_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            outside = Path(directory).parent / f"{Path(directory).name}-outside-prompts.txt"
            outside.write_text("private contents\n", encoding="utf-8")
            snapshot = Path(directory) / "results.prompts.txt"
            snapshot.symlink_to(outside)
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "prompts_snapshot": snapshot.name,
                        "prompts_sha256": digest,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            try:
                with self.assertRaisesRegex(ValueError, "unavailable or unsafe"):
                    materialize_verified_prompts_snapshot(results)
            finally:
                outside.unlink()

    def test_legacy_metadata_declares_prompts_without_dereferencing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "test_command": "mlx-model-test.sh --prompts /private/file.txt",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertTrue(results_metadata_declares_prompts(results))
            self.assertIsNone(materialize_verified_prompts_snapshot(results))

    def test_malformed_legacy_command_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results.jsonl"
            results.write_text(
                json.dumps(
                    {
                        "_meta": True,
                        "test_command": "mlx-model-test.sh --prompts 'unterminated",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "metadata is malformed"):
                results_metadata_declares_prompts(results)


if __name__ == "__main__":
    unittest.main()
