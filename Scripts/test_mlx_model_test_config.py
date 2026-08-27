import json
import tempfile
import unittest
from pathlib import Path

from mlx_model_test_config import (
    ai_intent_for_result,
    expand_template_runs,
    parse_ai_intent_specs,
    parse_prompts_file,
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


if __name__ == "__main__":
    unittest.main()
