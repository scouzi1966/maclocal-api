import unittest
from mlx_model_test_oracle import paired_judge_evidence


class PairedJudgeEvidenceTests(unittest.TestCase):
    def test_matches_only_same_checkpoint_prompt_and_variant(self):
        result = dict(label="streaming-seeded", model="checkpoint", prompt="poem")
        peer = dict(result, label="non-streaming-seeded", content="actual response")
        records = [peer, dict(peer, model="other"), dict(peer, prompt="other"),
                   dict(peer, is_baseline=True), dict(peer, status="SKIP")]
        self.assertEqual(paired_judge_evidence(result, records), [peer])
        self.assertEqual(paired_judge_evidence(dict(result, is_baseline=True), records), [])

    def test_missing_pair_does_not_manufacture_evidence(self):
        self.assertEqual(paired_judge_evidence(dict(label="streaming-seeded"), []), [])
        self.assertEqual(paired_judge_evidence(dict(label="unrelated"), [{}]), [])

    def test_stop_seed_pair_is_symmetric(self):
        first = dict(label="stop-seed-run1", model="m", prompt="p")
        second = dict(first, label="stop-seed-run2")
        self.assertEqual(paired_judge_evidence(first, [second]), [second])
        self.assertEqual(paired_judge_evidence(second, [first]), [first])


if __name__ == "__main__":
    unittest.main()
