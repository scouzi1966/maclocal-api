"""Resource staging must follow SwiftPM when local defaults are overridden."""
from pathlib import Path
import re
import subprocess
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / "swiftpm-reliable.sh"


class PathPrecedenceTests(unittest.TestCase):
    def resolve(self, function, *arguments):
        body = re.search(rf"{function}\(\) \{{.*?\n\}}", WRAPPER.read_text(), re.S).group()
        return subprocess.check_output(
            ["bash", "-c", 'ROOT_DIR=/consumer\n' + body + f'\n{function} "$@"',
             "fixture", *arguments], text=True).strip()

    def test_provider_overrides_generated_package(self):
        self.assertEqual(self.resolve("test_package_root", "--package-path", "/generated/package",
                                      "test", "--package-path", "/provider with spaces"),
                         "/provider with spaces")

    def test_provider_scratch_overrides_consumer_default(self):
        self.assertEqual(self.resolve("test_scratch_path", "--scratch-path", "/consumer/.build",
                                      "--scratch-path=/provider-tests"), "/provider-tests")

    def test_last_separate_argument_overrides_equals_form(self):
        self.assertEqual(self.resolve("test_package_root", "--package-path=/first",
                                      "--package-path", "/last"), "/last")

    def test_default_paths(self):
        self.assertEqual(self.resolve("test_package_root", "test"), "/consumer")
        self.assertEqual(self.resolve("test_scratch_path", "test"), "/consumer/.build")


if __name__ == "__main__":
    unittest.main()
