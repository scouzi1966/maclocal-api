#!/usr/bin/env python3
"""Inspect automatic PR reviews before treating a change as review-ready."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


GRAPHQL_URL = "https://api.github.com/graphql"
PR_PATTERN = re.compile(r"^(?P<owner>[^/]+)/(?P<name>[^#]+)#(?P<number>[0-9]+)$")
RATE_LIMIT_MARKERS = (
    "review budget",
    "rate limit hit",
    "used your own review budget",
)

QUERY = """
query ReviewState($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      title
      url
      state
      merged
      mergeable
      reviewDecision
      comments(first: 100) {
        nodes {
          databaseId
          createdAt
          author { login }
          body
          url
        }
      }
      reviews(first: 100) {
        nodes {
          databaseId
          submittedAt
          author { login }
          state
          body
          url
        }
      }
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          comments(first: 100) {
            nodes {
              databaseId
              createdAt
              author { login }
              body
              path
              line
              url
            }
          }
        }
      }
    }
  }
}
"""


def parse_pr(value: str) -> tuple[str, str, int]:
    match = PR_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("PR must be specified as owner/repository#number")
    return match.group("owner"), match.group("name"), int(match.group("number"))


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    executable = os.environ.get("GH_EXECUTABLE", "gh")
    try:
        completed = subprocess.run(
            [executable, "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    token = completed.stdout.strip()
    return token or None


def fetch_pr_review_state(
    owner: str,
    repository: str,
    number: int,
    token: str | None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if not token:
        raise RuntimeError(
            "GitHub GraphQL authentication is required; set GITHUB_TOKEN or log in with gh"
        )
    payload = json.dumps(
        {
            "query": QUERY,
            "variables": {"owner": owner, "name": repository, "number": number},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "maclocal-api-review-check",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            document = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API HTTP {error.code}: {detail}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GitHub API request failed: {error}") from error

    if document.get("errors"):
        messages = "; ".join(
            str(item.get("message", item)) for item in document["errors"]
        )
        raise RuntimeError(f"GitHub GraphQL errors: {messages}")
    pull_request = document.get("data", {}).get("repository", {}).get("pullRequest")
    if pull_request is None:
        raise RuntimeError("GitHub API returned no pull request")
    return pull_request


def author_login(node: dict[str, Any]) -> str:
    return str((node.get("author") or {}).get("login") or "")


def is_selected_bot(node: dict[str, Any], bots: set[str]) -> bool:
    return author_login(node).lower() in bots


def normalize_state(
    pull_request: dict[str, Any],
    bots: set[str],
) -> dict[str, Any]:
    comments = pull_request.get("comments", {}).get("nodes", [])
    reviews = pull_request.get("reviews", {}).get("nodes", [])
    threads = pull_request.get("reviewThreads", {}).get("nodes", [])
    automatic_comments = [item for item in comments if is_selected_bot(item, bots)]
    automatic_reviews = [item for item in reviews if is_selected_bot(item, bots)]
    automatic_threads = []
    for thread in threads:
        thread_comments = thread.get("comments", {}).get("nodes", [])
        if any(is_selected_bot(item, bots) for item in thread_comments):
            automatic_threads.append(
                {
                    **thread,
                    "url": next(
                        (item.get("url") for item in thread_comments if item.get("url")),
                        None,
                    ),
                }
            )

    automatic_text = " ".join(
        str(item.get("body") or "")
        for item in automatic_comments + automatic_reviews
    ).lower()
    rate_limited = any(marker in automatic_text for marker in RATE_LIMIT_MARKERS)
    changes_requested = any(
        str(item.get("state") or "").upper() == "CHANGES_REQUESTED"
        for item in automatic_reviews
    )

    return {
        "pr": {
            "number": pull_request.get("number"),
            "title": pull_request.get("title"),
            "url": pull_request.get("url"),
            "state": pull_request.get("state"),
            "merged": pull_request.get("merged"),
            "mergeable": pull_request.get("mergeable"),
            "review_decision": pull_request.get("reviewDecision"),
        },
        "selected_bots": sorted(bots),
        "automatic_comment_count": len(automatic_comments),
        "automatic_review_count": len(automatic_reviews),
        "automatic_review_states": sorted(
            {str(item.get("state") or "UNKNOWN") for item in automatic_reviews}
        ),
        "automatic_threads": [
            {
                "url": thread.get("url"),
                "resolved": bool(thread.get("isResolved")),
                "outdated": bool(thread.get("isOutdated")),
                "comments": thread.get("comments", {}).get("nodes", []),
            }
            for thread in automatic_threads
        ],
        "rate_limited": rate_limited,
        "changes_requested": changes_requested,
    }


def evaluate_state(
    state: dict[str, Any],
    *,
    require_review: bool = False,
    allow_outdated: bool = False,
) -> tuple[str, list[str]]:
    findings: list[str] = []
    if state["changes_requested"]:
        findings.append("Selected automatic reviewer requested changes")

    unresolved = []
    for thread in state["automatic_threads"]:
        if thread["resolved"]:
            continue
        if allow_outdated and thread["outdated"]:
            continue
        unresolved.append(thread)
    if unresolved:
        findings.append(
            f"Selected automatic reviewer has {len(unresolved)} unresolved thread(s)"
        )

    observed = bool(state["automatic_comment_count"] or state["automatic_review_count"])
    if require_review and not observed:
        findings.append("No selected automatic review or guide comment was observed")
    if require_review and state["rate_limited"]:
        findings.append(
            "Selected automatic review was rate-limited; request a completed review"
        )

    return ("FAIL" if findings else "PASS"), findings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("pr", help="Pull request as owner/repository#number")
    parser.add_argument(
        "--bot",
        action="append",
        default=["sourcery-ai"],
        help="Automatic reviewer login; repeatable (default: sourcery-ai)",
    )
    parser.add_argument(
        "--require-review",
        action="store_true",
        help="Fail when no selected review was observed or the review was rate-limited",
    )
    parser.add_argument(
        "--allow-outdated",
        action="store_true",
        help="Do not fail for unresolved threads marked outdated",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="GitHub API timeout in seconds (default: 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        owner, repository, number = parse_pr(args.pr)
        state = normalize_state(
            fetch_pr_review_state(
                owner,
                repository,
                number,
                github_token(),
                timeout=args.timeout,
            ),
            {bot.lower() for bot in args.bot},
        )
        outcome, findings = evaluate_state(
            state,
            require_review=args.require_review,
            allow_outdated=args.allow_outdated,
        )
        state["outcome"] = outcome
        state["findings"] = findings
        if args.json:
            print(json.dumps(state, indent=2, sort_keys=True))
        else:
            print(f"PR: {state['pr']['url']}")
            print(f"Automatic comments: {state['automatic_comment_count']}")
            print(f"Automatic reviews: {state['automatic_review_count']}")
            print(f"Automatic threads: {len(state['automatic_threads'])}")
            print(f"Rate limited: {state['rate_limited']}")
            print(f"Outcome: {outcome}")
            for finding in findings:
                print(f"- {finding}")
        return 0 if outcome == "PASS" else 1
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
