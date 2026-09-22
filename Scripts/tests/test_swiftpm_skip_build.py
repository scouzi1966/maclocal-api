"""Exercise skip-build rejection before the wrapper can invalidate products."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


WRAPPER = Path(__file__).resolve().parents[1] / "swiftpm-reliable.sh"


class SkipBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "Scripts"
        scripts.mkdir()
        self.wrapper = scripts / WRAPPER.name
        shutil.copy2(WRAPPER, self.wrapper)
        self.state = self.root / ".build-reliable-state"
        self.state.mkdir()
        self.binary = self.root / ".build/release/afm"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"preserve release executable")
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        swift = self.fake_bin / "swift"
        swift.write_text("#!/bin/sh\necho 'fixture compiler reached' >&2\nexit 77\n")
        swift.chmod(0o755)

    def invoke(self, *arguments):
        return subprocess.run(
            ["bash", str(self.wrapper), "test", *arguments],
            env={**os.environ, "PATH": f"{self.fake_bin}:{os.environ['PATH']}"},
            capture_output=True, text=True,
        )

    def assert_preserved(self):
        self.assertEqual(self.binary.read_bytes(), b"preserve release executable")

    def test_release_build_then_skip_build_is_rejected_without_mutation(self):
        stamp = self.state / "last-operation-release"
        stamp.write_text("build\n")
        result = self.invoke("-c", "release", "--skip-build")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Existing products were preserved", result.stderr)
        self.assertNotIn("fixture compiler reached", result.stderr)
        self.assertEqual(stamp.read_text(), "build\n")
        self.assert_preserved()

    def test_missing_test_build_is_rejected(self):
        result = self.invoke("--configuration=release", "--skip-build")
        self.assertEqual(result.returncode, 2)
        self.assert_preserved()

    def test_default_debug_checks_its_own_stamp(self):
        (self.state / "last-operation-release").write_text("test\n")
        result = self.invoke("--skip-build")
        self.assertEqual(result.returncode, 2)
        self.assertIn("for debug", result.stderr)
        self.assert_preserved()

    def test_existing_test_build_reaches_normal_validation(self):
        (self.state / "last-operation-release").write_text("test\n")
        result = self.invoke("--skip-build", "--configuration", "release")
        self.assertNotIn("Cannot use --skip-build", result.stderr)
        self.assertIn("Unable to run the selected Swift compiler", result.stderr)
        self.assert_preserved()

    def test_ordinary_test_still_reaches_normal_validation(self):
        (self.state / "last-operation-release").write_text("build\n")
        result = self.invoke("-c", "release")
        self.assertIn("Unable to run the selected Swift compiler", result.stderr)
        self.assert_preserved()


if __name__ == "__main__":
    unittest.main()
