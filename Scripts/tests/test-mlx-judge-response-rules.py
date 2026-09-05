#!/usr/bin/env python3
"""CPU prompt-contract fixtures; no model invocation or inferred judge scores."""
from pathlib import Path
import re
import subprocess
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "mlx-model-test.sh").read_text()


def heredoc_assignment(name):
    match = re.search(
        rf"^ +{name}=\"\$\(cat <<'{name}_END'\n.*?^{name}_END\n\)\"",
        SOURCE, re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise AssertionError(f"Missing production assignment: {name}")
    return match.group()


def assembled_prompt(name):
    shared = heredoc_assignment("JUDGE_RESPONSE_RULES")
    if name == "AFM_SCORE_PROMPT":
        # Include only the production literal assignment, not the model invocation.
        start = SOURCE.index('            AFM_SCORE_PROMPT="You are')
        end = SOURCE.index('\n            AFM_SCORE_PROMPT="$JUDGE_RESPONSE_RULES', start)
        assignment = SOURCE[start:end]
    else:
        assignment = heredoc_assignment(name)
    prepend = re.search(
        rf'^ +{name}="\$JUDGE_RESPONSE_RULES\n\n\${name}"$',
        SOURCE, re.MULTILINE,
    )
    if not prepend:
        raise AssertionError(f"Shared policy not wired into {name}")
    script = shared + "\n" + assignment + "\n" + prepend.group()
    script += f'\nprintf "%s" "${name}"\n'
    return subprocess.check_output(["bash", "-eu", "-c", script], text=True)


class JudgeResponseRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = {name: assembled_prompt(name) for name in (
            "ANALYSIS_PROMPT", "PER_TEST_PROMPT", "AFM_SCORE_PROMPT",
        )}

    def check_all(self, *clauses):
        for name, prompt in self.prompts.items():
            with self.subTest(mode=name):
                for clause in clauses:
                    self.assertIn(clause, prompt)

    def test_all_modes_prepend_identical_policy(self):
        policies = []
        for prompt in self.prompts.values():
            self.assertTrue(prompt.startswith("RESPONSE CONTRACT PRECEDENCE"))
            policies.append(prompt.split("\n\nYou are", 1)[0])
        self.assertEqual(len(set(policies)), 1)

    def test_tool_only_contract_requires_valid_expected_calls(self):
        self.check_all('finish_reason="tool_calls"', "expected function names/arguments",
                       "penalty or score-3 cap", "or unexpected calls are not exempt")

    def test_immediate_stop_requires_explicit_empty_expectation(self):
        self.check_all('content_equals=""', 'with finish_reason="stop"',
                       'finish_reason="stop" alone does not establish')

    def test_failure_and_baseline_guards_apply_before_exceptions(self):
        self.check_all("status=FAIL remains score 1", "never overrides a failed assertion",
                       "For is_baseline=true, ignore the enclosing",
                       "judge only the result's own prompt")
        for prompt in self.prompts.values():
            self.assertLess(prompt.index("For is_baseline=true"),
                            prompt.index("EXPECTED TOOL-ONLY RESPONSE"))

    def test_unexpected_empty_stays_evidence_based(self):
        self.check_all("Do not automatically score 3", "state unknown quality",
                       "Keep the cause unattributed unless evidence establishes",
                       "neither successful behavior nor a harness capture bug")
        self.assertNotIn("the test harness failed to capture output", SOURCE)
        self.assertNotIn("If BOTH content and reasoning_content", SOURCE)

    def test_reasoning_only_does_not_prove_budget_or_engine_health(self):
        self.check_all("possible budget exhaustion", "Token counts alone do not prove",
                       "above take precedence over this heuristic")
        self.assertNotIn("it means the max_tokens was too low", SOURCE)

    def test_intent_can_explicitly_require_truncated_json(self):
        self.check_all("output contracts take precedence over generic formatting rules",
                       "expect truncated invalid JSON")


if __name__ == "__main__":
    unittest.main()
