"""Exercise the wrapper's actual package-name expression without building."""
from pathlib import Path
import re
import subprocess
import unittest

WRAPPER = Path(__file__).resolve().parents[1] / 'swiftpm-reliable.sh'


class SwiftPMPackageNameTests(unittest.TestCase):
    def extract(self, manifest):
        line = next(line for line in WRAPPER.read_text().splitlines() if line.strip().startswith('package_name="$('))
        expression = re.search(r"sed -nE '([^']+)'", line).group(1)
        output = subprocess.check_output(['sed', '-nE', expression], input=manifest, text=True)
        return output.splitlines()[0] if output.splitlines() else ''

    def test_helper_target_does_not_name_the_package_bundle(self):
        self.assertEqual(self.extract('''let helper = Target.target(
    name: "Cmlx",
)
let package = Package(
    name: "AFMKit",
    targets: [.testTarget(
        name: "AFMKitMLXTests")]
)
'''), 'AFMKit')

    def test_consumer_manifest(self):
        self.assertEqual(self.extract((WRAPPER.parents[1] / 'Package.swift').read_text()), 'MacLocalAPI')

    def test_no_package_fails_closed(self):
        self.assertEqual(self.extract('let helper = Target.target(\n    name: "Cmlx")\n'), '')


if __name__ == '__main__':
    unittest.main()
