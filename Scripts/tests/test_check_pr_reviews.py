#!/usr/bin/env python3
"""Offline fixtures for automatic-review inspection; no network or tokens."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "check_pr_reviews.py"
SPEC = importlib.util.spec_from_file_location("check_pr_reviews", MODULE_PATH)
check_pr_reviews = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_pr_reviews)
sys.modules[check_pr_reviews.__name__] = check_pr_reviews


def state(
    *,
    comments=(),
    reviews=(),
    threads=(),
):
    return check_pr_reviews.normalize_state(
        {
            "number": 283,
            "title": "Fixture",
            "url": "https://example.invalid/pr/283",
            "state": "OPEN",
            "merged": False,
            "mergeable": "MERGEABLE",
            "reviewDecision": "REVIEW_REQUIRED",
            "comments": {"nodes": list(comments)},
            "reviews": {"nodes": list(reviews)},
            "reviewThreads": {"nodes": list(threads)},
        },
        {"sourcery-ai"},
    )


def thread(resolved=False, outdated=False):
    return {
        "url": "https://example.invalid/thread",
        "isResolved": resolved,
        "isOutdated": outdated,
        "comments": {
            "nodes": [
                {
                    "author": {"login": "sourcery-ai"},
                    "body": "fixture finding",
                    "path": "README.md",
                    "line": 1,
                }
            ]
        },
    }


class CheckPrReviewsTests(unittest.TestCase):
    def test_parses_and_rejects_pr_references(self):
        self.assertEqual(
            check_pr_reviews.parse_pr("scouzi1966/maclocal-api#283"),
            ("scouzi1966", "maclocal-api", 283),
        )
        with self.assertRaisesRegex(ValueError, "owner/repository#number"):
            check_pr_reviews.parse_pr("scouzi1966/maclocal-api")

    def test_rate_limited_guide_is_visible_but_not_a_finding(self):
        review_state = state(
            comments=[{"author": {"login": "sourcery-ai"}, "body": "Reviewer guide"}],
            reviews=[
                {
                    "author": {"login": "sourcery-ai"},
                    "state": "COMMENTED",
                    "body": "Sorry, you've used your own review budget.",
                }
            ],
        )

        outcome, findings = check_pr_reviews.evaluate_state(review_state)

        self.assertTrue(review_state["rate_limited"])
        self.assertEqual(review_state["automatic_comment_count"], 1)
        self.assertEqual(review_state["automatic_review_count"], 1)
        self.assertEqual((outcome, findings), ("PASS", []))

    def test_unresolved_and_changes_requested_reviews_fail(self):
        review_state = state(
            reviews=[
                {
                    "author": {"login": "sourcery-ai"},
                    "state": "CHANGES_REQUESTED",
                    "body": "Please revise",
                }
            ],
            threads=[thread()],
        )

        outcome, findings = check_pr_reviews.evaluate_state(review_state)

        self.assertEqual(outcome, "FAIL")
        self.assertEqual(len(findings), 2)

    def test_require_review_fails_missing_and_rate_limited_reviews(self):
        missing = state()
        limited = state(
            reviews=[
                {
                    "author": {"login": "sourcery-ai"},
                    "state": "COMMENTED",
                    "body": "rate limit hit",
                }
            ]
        )

        self.assertEqual(
            check_pr_reviews.evaluate_state(missing, require_review=True)[0], "FAIL"
        )
        self.assertEqual(
            check_pr_reviews.evaluate_state(limited, require_review=True)[0], "FAIL"
        )

    def test_outdated_thread_policy_is_explicit(self):
        review_state = state(threads=[thread(outdated=True)])

        self.assertEqual(
            check_pr_reviews.evaluate_state(review_state)[0], "FAIL"
        )
        self.assertEqual(
            check_pr_reviews.evaluate_state(review_state, allow_outdated=True)[0],
            "PASS",
        )

    def test_cli_prints_json_and_exit_status(self):
        fixture = state(
            reviews=[
                {
                    "author": {"login": "sourcery-ai"},
                    "state": "COMMENTED",
                    "body": "Reviewer guide",
                }
            ]
        )
        with patch.object(
            check_pr_reviews,
            "fetch_pr_review_state",
            return_value={
                "number": 283,
                "title": "Fixture",
                "url": "https://example.invalid/pr/283",
                "state": "OPEN",
                "merged": False,
                "mergeable": "MERGEABLE",
                "reviewDecision": "REVIEW_REQUIRED",
                "comments": {"nodes": []},
                "reviews": {"nodes": []},
                "reviewThreads": {"nodes": []},
            },
        ), patch.object(check_pr_reviews, "github_token", return_value="fixture"), patch.object(
            check_pr_reviews, "normalize_state", return_value=fixture
        ):
            code = check_pr_reviews.main(["scouzi1966/maclocal-api#283", "--json"])

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
