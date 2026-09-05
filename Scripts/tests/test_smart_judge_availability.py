#!/usr/bin/env python3
"""CPU-only smart-judge fixtures; no model, network, or real judge is invoked."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Scripts/mlx-model-test.sh"


def run_reanalysis(fake_judge_script, records):
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        bindir = work / "bin"
        bindir.mkdir()
        fake_codex = bindir / "codex"
        fake_codex.write_text(fake_judge_script, encoding="utf-8")
        fake_codex.chmod(0o755)

        results = work / "results.jsonl"
        results.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        environment = dict(os.environ)
        environment["PATH"] = f"{bindir}{os.pathsep}{environment.get('PATH', '')}"
        # CPU-only reanalysis must not require the generation-time OpenAI SDK.
        environment.pop("PYTHONPATH", None)
        environment["AFM_TEST_WORK_ROOT"] = str(work / "judge-work")
        completed = subprocess.run(
            [
                str(SCRIPT),
                "--reanalyse",
                str(results),
                "--smart",
                "1:codex",
                "--no-report",
            ],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        reports = [
            (path, path.read_text(encoding="utf-8"))
            for path in (work / "test-reports").glob("smart-analysis-codex-*.md")
        ]
        return completed, reports


class SmartJudgeAvailabilityTests(unittest.TestCase):
    def test_failing_judge_stops_before_synthesizing_scores(self):
        records = [
            {"model": "fixture/model", "label": "case", "status": "OK"},
            {"model": "fixture/model", "label": "unreached", "status": "OK"},
        ]
        completed, reports = run_reanalysis(
            "#!/bin/sh\n"
            "echo 'usage limit fixture' >&2\n"
            "exit 7\n",
            records,
        )

        self.assertEqual(completed.returncode, 7)
        self.assertIn("codex judge exited with status 7", completed.stderr)
        self.assertIn("result 0", completed.stderr)
        self.assertIn("[1/2]", completed.stdout)
        self.assertNotIn("[2/2]", completed.stdout)
        self.assertEqual(reports, [])

    def test_reasonless_score_is_rejected_before_the_next_result(self):
        records = [
            {"model": "fixture/model", "label": "case", "status": "OK"},
            {"model": "fixture/model", "label": "unreached", "status": "OK"},
        ]
        completed, reports = run_reanalysis(
            "#!/bin/sh\n"
            "echo '{\"score\":5}'\n",
            records,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "codex judge returned no valid score payload for result 0",
            completed.stderr,
        )
        self.assertIn("[1/2]", completed.stdout)
        self.assertNotIn("[2/2]", completed.stdout)
        self.assertEqual(reports, [])

    def test_genuine_score_and_reason_produce_a_complete_report(self):
        records = [
            {"model": "fixture/model", "label": "first", "status": "OK"},
            {
                "model": "fixture/model",
                "label": "skipped",
                "status": "SKIP",
                "overall_status": "skip",
            },
            {"model": "fixture/model", "label": "second", "status": "OK"},
        ]
        completed, reports = run_reanalysis(
            "#!/bin/sh\n"
            "echo '{\"score\":4,\"reason\":\"fixture reason\"}'\n",
            records,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("[3/3] second... score=4", completed.stdout)
        self.assertEqual(len(reports), 1)
        report = reports[0][1]
        self.assertIn("> fixture reason", report)
        self.assertIn(
            '<!-- AI_SCORES [{"i": 0, "s": 4}, {"i": 2, "s": 4}] -->',
            report,
        )


if __name__ == "__main__":
    unittest.main()
