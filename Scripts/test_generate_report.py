import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("generate-report.py")


class GenerateReportTests(unittest.TestCase):
    def test_batch_scores_render_for_passed_and_failed_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            report_dir = root / "test-reports"
            report_dir.mkdir()
            results_path = root / "results.jsonl"
            records = [
                {"_meta": True, "test_command": "mlx-model-test.sh"},
                self.record("pass-case", "pass", "OK"),
                self.record("fail-case", "fail", "FAIL"),
            ]
            results_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            # Batch judges occasionally place the comment close on the next line.
            # The JSON array remains valid and should still populate every record.
            smart_timestamp = "20260825_010203"
            (report_dir / f"smart-analysis-codex-{smart_timestamp}.md").write_text(
                "## Analysis\n\n<!-- AI_SCORES "
                '[{"i":0,"s":5},{"i":1,"s":2}]\n-->\n',
                encoding="utf-8",
            )

            env = os.environ.copy()
            env.update(
                {
                    "RESULTS_FILE": str(results_path),
                    "REPORT_OUTPUT_DIR": str(root),
                    "REPORT_TIMESTAMP": "20260825_040506",
                    "SMART_TIMESTAMP": smart_timestamp,
                    "AFM_REPORT_NO_OPEN": "1",
                }
            )
            subprocess.run([sys.executable, str(SCRIPT)], env=env, check=True)

            report = (
                report_dir / "mlx-model-report-20260825_040506.html"
            ).read_text(encoding="utf-8")
            self.assertEqual(report.count("🤖 AI Scores"), 2)
            self.assertEqual(report.count("<strong>codex</strong>"), 2)
            self.assertIn("5/5 ✅", report)
            self.assertIn("2/5 ❌", report)
            self.assertEqual(report.count('class="response-section" id="resp-'), 2)

    def test_full_prompt_is_rendered_without_truncation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            report_dir = root / "test-reports"
            report_dir.mkdir()
            results_path = root / "results.jsonl"
            long_prompt = "A" * 600 + " FULL_PROMPT_TAIL"
            record = self.record("long-prompt", "pass", "OK")
            record["prompt"] = long_prompt
            results_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "RESULTS_FILE": str(results_path),
                    "REPORT_OUTPUT_DIR": str(root),
                    "REPORT_TIMESTAMP": "20260825_050607",
                    "AFM_REPORT_NO_OPEN": "1",
                }
            )
            subprocess.run([sys.executable, str(SCRIPT)], env=env, check=True)

            report = (
                report_dir / "mlx-model-report-20260825_050607.html"
            ).read_text(encoding="utf-8")
            self.assertIn(long_prompt, report)
            self.assertNotIn("A" * 500 + "...", report)

    @staticmethod
    def record(label, overall_status, status):
        return {
            "model": "test/model",
            "label": label,
            "prompt": f"Prompt for {label}",
            "content": "response",
            "content_preview": "response",
            "reasoning_content": "",
            "overall_status": overall_status,
            "status": status,
            "assertion_status": "pass" if overall_status == "pass" else "fail",
            "assertion_failures": [] if overall_status == "pass" else ["expected failure"],
            "completion_tokens": 2,
            "prompt_tokens": 3,
            "total_tokens": 5,
            "tokens_per_sec": 10.0,
            "load_time_s": 1.0,
            "gen_time_s": 0.2,
            "temperature": 0.0,
            "max_tokens": 20,
            "finish_reason": "stop",
            "afm_args": "--no-think",
        }


if __name__ == "__main__":
    unittest.main()
