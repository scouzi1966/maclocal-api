import unittest
from mlx_model_test_oracle import paired_judge_evidence


class PairedJudgeEvidenceTests(unittest.TestCase):
    def fixture(self, label="streaming-seeded"):
        return dict(label=label, model="checkpoint", prompt="poem", status="OK",
                    transport_status="pass", overall_status="pass",
                    assertion_status="not_configured", assertion_failures=[],
                    content="actual response", temperature=0.7, max_tokens=200,
                    max_completion_tokens=None, system_prompt="", developer_prompt=None,
                    server_instructions=None, afm_args="--mtp --seed 123")

    def test_matches_only_same_checkpoint_prompt_and_variant(self):
        result = self.fixture()
        peer = dict(result, label="non-streaming-seeded", content="actual response")
        records = [peer, dict(peer, model="other"), dict(peer, prompt="other"),
                   dict(peer, is_baseline=True), dict(peer, status="SKIP")]
        self.assertEqual(paired_judge_evidence(result, records), [peer])
        self.assertEqual(paired_judge_evidence(dict(result, is_baseline=True), records), [])

    def test_missing_pair_does_not_manufacture_evidence(self):
        self.assertEqual(paired_judge_evidence(dict(label="streaming-seeded"), []), [])
        self.assertEqual(paired_judge_evidence(dict(label="unrelated"), [{}]), [])

    def test_stop_seed_pair_is_symmetric(self):
        first = self.fixture("stop-seed-run1")
        second = dict(first, label="stop-seed-run2")
        self.assertEqual(paired_judge_evidence(first, [second]), [second])
        self.assertEqual(paired_judge_evidence(second, [first]), [first])

    def test_both_records_must_have_successful_transport_and_assertions(self):
        result = self.fixture()
        peer = dict(result, label="non-streaming-seeded")
        for field, value in (("status", "FAIL"), ("transport_status", "fail"),
                             ("overall_status", "fail"), ("assertion_status", "fail"),
                             ("assertion_failures", ["wrong answer"]), ("error", "timeout"),
                             ("content", None)):
            with self.subTest(field=field):
                self.assertEqual(paired_judge_evidence(result, [dict(peer, **{field: value})]), [])
                self.assertEqual(paired_judge_evidence(dict(result, **{field: value}), [peer]), [])

    def test_runtime_and_request_mismatches_are_not_paired(self):
        result = self.fixture()
        peer = dict(result, label="non-streaming-seeded")
        for field, value in (("temperature", 0), ("seed", 42), ("stop", ["STOP"]),
                             ("max_tokens", 10), ("system_prompt", "different"),
                             ("developer_prompt", "different"), ("tools", []),
                             ("afm_args", "--mtp --seed 42"), ("top_p", 0.5)):
            with self.subTest(field=field):
                self.assertEqual(paired_judge_evidence(result, [dict(peer, **{field: value})]), [])

    def test_missing_configuration_cannot_pair_even_when_both_omit_it(self):
        for field in ("temperature", "max_tokens", "afm_args", "system_prompt", "status"):
            result = self.fixture()
            del result[field]
            self.assertEqual(paired_judge_evidence(result, [dict(result, label="non-streaming-seeded")]), [])

    def test_streaming_pair_allows_only_transport_difference(self):
        result = dict(self.fixture(), stream=True)
        peer = dict(result, label="non-streaming-seeded", stream=False,
                    afm_args="--mtp --no-streaming --seed 123")
        self.assertEqual(paired_judge_evidence(result, [peer]), [peer])
        self.assertEqual(paired_judge_evidence(peer, [result]), [result])
        self.assertEqual(paired_judge_evidence(result, [dict(peer, afm_args="--seed 123 --no-streaming")]), [])

    def test_repeated_seed_pair_does_not_allow_streaming_difference(self):
        result = self.fixture("seed-42-run1")
        peer = dict(result, label="seed-42-run2", afm_args="--mtp --seed 123 --no-streaming")
        self.assertEqual(paired_judge_evidence(result, [peer]), [])

    def test_malformed_arguments_do_not_pair(self):
        result = self.fixture()
        result['afm_args'] = '"unterminated'
        self.assertEqual(paired_judge_evidence(result, [dict(result, label="non-streaming-seeded")]), [])


if __name__ == "__main__":
    unittest.main()
