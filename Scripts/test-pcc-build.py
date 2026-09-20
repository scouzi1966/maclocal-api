#!/usr/bin/env python3
"""Offline orchestration tests: no Swift build, keychain access, signing, or inference."""
import argparse
import contextlib
import importlib.util
import io
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("pcc_build", pathlib.Path(__file__).with_name("build-pcc-app.py"))
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PCCBuildTests(unittest.TestCase):
    def setUp(self):
        root = builder.ROOT / ".build-pcc-script-tests"
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.profile = self.root / "fixture profile.provisionprofile"
        self.profile.write_text("inert fixture")
        self.binary = self.root / "fixture afm"
        self.binary.write_text("inert fixture")
        self.args = argparse.Namespace(profile=str(self.profile), identity="fixture-identity",
                                       binary=self.binary, output=self.root / "AFM.app")

    def test_existing_binary_only_preflights_and_packages(self):
        with patch.object(builder, "require_platform"), patch.object(builder.subprocess, "run") as run:
            builder.build_and_sign(self.args)
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1] + ["--check-only"])
        self.assertEqual(calls[1][calls[1].index("--profile") + 1], str(self.profile))
        self.assertEqual(calls[1][calls[1].index("--binary") + 1], str(self.binary))

    def test_build_uses_wrapper_and_fresh_scratch_before_signing(self):
        self.args.binary = None

        def fake_run(command, **kwargs):
            if "--scratch-path" in command:
                scratch = pathlib.Path(command[command.index("--scratch-path") + 1])
                binary = scratch / "out/Products/Release/afm"
                binary.parent.mkdir(parents=True)
                binary.write_text("inert fixture build")

        with patch.object(builder, "ROOT", self.root), patch.object(builder, "require_platform"), \
             patch.object(builder.subprocess, "run", side_effect=fake_run) as run:
            builder.build_and_sign(self.args)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertTrue(commands[0][0].endswith("check-afmkit-consumer-boundary.sh"))
        self.assertTrue(commands[1][0].endswith("swiftpm-reliable.sh"))
        self.assertEqual(commands[1][1:6], ["build", "-c", "release", "--product", "afm"])
        self.assertIn(str(self.root / ".build-pcc"), commands[1][-1])
        self.assertEqual(commands[2][-1], "--check-only")

    def test_failed_preflight_never_signs(self):
        with patch.object(builder, "require_platform"), \
             patch.object(builder.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["fixture"])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build_and_sign(self.args)
        self.assertEqual(run.call_count, 1)

    def test_failed_build_never_packages_or_uses_stale_binary(self):
        self.args.binary = None
        with patch.object(builder, "ROOT", self.root), patch.object(builder, "require_platform"), \
             patch.object(builder.subprocess, "run", side_effect=[None, subprocess.CalledProcessError(1, ["fixture"])]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                builder.build_and_sign(self.args)
        self.assertEqual(run.call_count, 2)

    def test_existing_output_fails_before_any_subprocess(self):
        self.args.output.mkdir()
        with patch.object(builder, "require_platform"), patch.object(builder.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                builder.build_and_sign(self.args)
        run.assert_not_called()

    def test_old_macos_fails_before_credentials_or_build(self):
        with patch.object(builder.platform, "system", return_value="Darwin"), \
             patch.object(builder.platform, "machine", return_value="arm64"), \
             patch.object(builder.platform, "mac_ver", return_value=("26.5", (), "")), \
             patch.object(builder.getpass, "getpass") as prompt:
            with self.assertRaisesRegex(ValueError, "macOS 27"):
                builder.build_and_sign(self.args)
        prompt.assert_not_called()

    def test_hidden_prompts_when_credentials_omitted(self):
        self.args.profile = self.args.identity = None
        with patch.object(builder, "require_platform"), \
             patch.object(builder.getpass, "getpass", side_effect=[str(self.profile), "fixture-identity"]) as prompt, \
             patch.object(builder.subprocess, "run"):
            builder.build_and_sign(self.args)
        self.assertEqual(prompt.call_count, 2)

    def test_failure_summary_does_not_echo_credential_command(self):
        output = io.StringIO()
        failure = subprocess.CalledProcessError(1, ["signer", "private-fixture-identity", "private-fixture-profile"])
        with patch.object(builder, "build_and_sign", side_effect=failure), contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit):
                builder.main([])
        self.assertNotIn("private-fixture", output.getvalue())

    def test_prompt_refuses_echo_fallback(self):
        with patch.object(builder.getpass, "getpass", side_effect=builder.getpass.GetPassWarning("fixture")):
            with self.assertRaisesRegex(ValueError, "private terminal"):
                builder.local_prompt("fixture prompt")

    def test_ambiguous_build_outputs_are_rejected(self):
        for name in ["out/Products/Release/afm", "arm64-apple-macosx/release/afm"]:
            path = self.root / name
            path.parent.mkdir(parents=True)
            path.write_text("inert fixture")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            builder.built_binary(self.root)


if __name__ == "__main__":
    unittest.main()
