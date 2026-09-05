#!/usr/bin/env python3
"""Assemble per-result AI scores into the markdown consumed by the HTML report."""

import json
import os
import sys
from pathlib import Path

from mlx_model_test_oracle import extract_score_payload


def assemble_per_test_report(scores_dir: str | Path, results_file: str | Path) -> str:
    scores_path = Path(scores_dir)
    scores = []
    report_lines = ["# Per-Test AI Analysis", ""]
    result_idx = 0

    with Path(results_file).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            result = json.loads(line)
            if result.get("_meta"):
                continue

            model = result.get("model", "")
            label = result.get("label", "")
            label_suffix = f" @ {label}" if label else ""
            name = (
                model
                if not label_suffix or model.endswith(label_suffix)
                else model + label_suffix
            )

            if result.get("status") == "SKIP" or result.get("overall_status") == "skip":
                skip_reason = result.get("skip_reason") or "Required capability unavailable"
                report_lines.append(f"### {result_idx}. {name}")
                report_lines.append("**Not scored** ⏭️ | Status: SKIP")
                report_lines.append(f"> {skip_reason}.")
                report_lines.append("")
                result_idx += 1
                continue

            score_file = scores_path / f"score_{result_idx}.txt"
            payload = None
            if score_file.exists():
                payload = extract_score_payload(score_file.read_text(encoding="utf-8").strip())
            if payload is None:
                raise ValueError(
                    f"Result {result_idx} ({name}) has no genuine judge payload; "
                    "refusing to synthesize a fallback score"
                )

            score_value = payload.get("score")
            if type(score_value) is not int or not 1 <= score_value <= 5:
                raise ValueError(
                    f"Result {result_idx} ({name}) has invalid judge score: {score_value!r}"
                )

            reason = payload.get("reason", "")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"Result {result_idx} ({name}) has no genuine judge reason; "
                    "refusing to publish its score"
                )
            scores.append({"i": result_idx, "s": score_value})

            tokens_per_second = result.get("tokens_per_sec", 0) or 0
            status = result.get("status", "")
            emoji = {5: "✅", 4: "👍", 3: "⚠️", 2: "❌", 1: "💥"}.get(
                score_value, "❓"
            )
            report_lines.append(f"### {result_idx}. {name}")
            report_lines.append(
                f"**Score: {score_value}/5** {emoji} | Status: {status} | "
                f"{float(tokens_per_second):.1f} tok/s"
            )
            if reason:
                report_lines.append(f"> {reason}")
            report_lines.append("")
            result_idx += 1

    total = len(scores)
    pass_count = sum(1 for score in scores if score["s"] >= 4)
    fail_count = sum(1 for score in scores if score["s"] <= 2)
    report_lines.append("---")
    report_lines.append(
        f"**Summary**: {pass_count}/{total} passed (score ≥ 4), "
        f"{fail_count} failed (score ≤ 2)"
    )
    report_lines.append("")
    report_lines.append("<!-- AI_SCORES " + json.dumps(scores) + " -->")
    return "\n".join(report_lines)


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: assemble_smart_report.py <scores-dir> <results-jsonl>",
            file=sys.stderr,
        )
        return os.EX_USAGE
    print(assemble_per_test_report(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
