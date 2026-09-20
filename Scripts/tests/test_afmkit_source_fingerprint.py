"""Regression coverage for writable AFMKit/DS4 source fingerprints."""

from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "afmkit-source-fingerprint.sh"


def run(*args, cwd=None):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def initialize_repository(path):
    path.mkdir(parents=True)
    run("git", "init", "-q", cwd=path)
    run("git", "config", "user.email", "fingerprint@example.invalid", cwd=path)
    run("git", "config", "user.name", "Fingerprint Test", cwd=path)


class AFMKitSourceFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)

        self.ds4_source = root / "ds4-source"
        initialize_repository(self.ds4_source)
        (self.ds4_source / "ds4.c").write_text("int ds4_value = 1;\n")
        (self.ds4_source / ".gitignore").write_text("build/\n")
        run("git", "add", ".", cwd=self.ds4_source)
        run("git", "commit", "-qm", "Initial DS4", cwd=self.ds4_source)

        self.afmkit = root / "AFMKit"
        initialize_repository(self.afmkit)
        (self.afmkit / "Sources").mkdir()
        (self.afmkit / "Sources" / "AFMKit.swift").write_text("public enum AFMKit {}\n")
        (self.afmkit / "Package.swift").write_text("// swift-tools-version: 6.0\n")
        run("git", "add", ".", cwd=self.afmkit)
        run("git", "commit", "-qm", "Initial AFMKit", cwd=self.afmkit)
        run(
            "git", "-c", "protocol.file.allow=always", "submodule", "add", "-q",
            str(self.ds4_source), "vendor/ds4", cwd=self.afmkit,
        )
        run("git", "commit", "-qam", "Add DS4", cwd=self.afmkit)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def fingerprint(self):
        return run(str(SCRIPT), str(self.afmkit), "workspace:test")

    def test_tracked_ds4_edit_changes_fingerprint(self):
        baseline = self.fingerprint()
        (self.afmkit / "vendor/ds4/ds4.c").write_text("int ds4_value = 2;\n")
        self.assertNotEqual(self.fingerprint(), baseline)

    def test_untracked_ds4_file_changes_fingerprint(self):
        baseline = self.fingerprint()
        (self.afmkit / "vendor/ds4/new-provider.c").write_text("int new_provider;\n")
        self.assertNotEqual(self.fingerprint(), baseline)

    def test_ignored_ds4_build_output_does_not_change_fingerprint(self):
        baseline = self.fingerprint()
        output = self.afmkit / "vendor/ds4/build/generated.o"
        output.parent.mkdir()
        output.write_bytes(b"generated")
        self.assertEqual(self.fingerprint(), baseline)


if __name__ == "__main__":
    unittest.main()
