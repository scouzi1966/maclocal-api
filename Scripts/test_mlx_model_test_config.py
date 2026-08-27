import json
import tempfile
import unittest
from pathlib import Path

from mlx_model_test_config import expand_template_runs, parse_prompts_file


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


if __name__ == "__main__":
    unittest.main()
