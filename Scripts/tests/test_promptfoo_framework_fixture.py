"""CPU-only contract checks for the read-before-edit framework fixture.

Uses the dataset's explicit indentation convention, without requiring PyYAML
or invoking Promptfoo, a server, or a model.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'Scripts/feature-promptfoo-agentic/datasets/agentic/framework-tool-schemas.yaml'
DESCRIPTION = 'pi: prefer read before edit for existing file changes'


def fixture_block():
    blocks = re.split(r'(?=^- description:)', DATASET.read_text(), flags=re.MULTILINE)
    return next(block for block in blocks
                if block.startswith(f'- description: "{DESCRIPTION}"'))


def tool_contract(block):
    system = re.search(r'^    system_prompt: "([^"]+)"$', block, re.MULTILINE).group(1)
    advertised = re.search(r'with only (.+?)\.', system).group(1)
    names = {name.strip() for name in re.split(r',|\band\b', advertised) if name.strip()}
    tools = set(re.findall(r'^          name: (\w+)$', block, re.MULTILINE))
    return names, tools


class PromptfooFrameworkFixtureTests(unittest.TestCase):
    def test_advertised_tools_equal_supplied_tools(self):
        advertised, supplied = tool_contract(fixture_block())
        self.assertEqual(supplied, {'read', 'write', 'edit'})
        self.assertEqual(advertised, supplied)

    def test_contract_detects_original_unavailable_bash_advertisement(self):
        original = fixture_block().replace('read, write, and edit.',
                                           'read, write, edit, and bash.')
        advertised, supplied = tool_contract(original)
        self.assertEqual(advertised - supplied, {'bash'})

    def test_read_first_requirement_and_assertion_are_preserved(self):
        block = fixture_block()
        self.assertIn('Inspect before editing.', block)
        self.assertIn('parsed.tool_calls.length === 1', block)
        self.assertIn("parsed.tool_calls[0].function.name === 'read'", block)


if __name__ == '__main__':
    unittest.main()
