"""Synthetic provenance regressions; no model payloads or inference involved."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'test-external-benchmarks.py'
spec = importlib.util.spec_from_file_location('external_benchmarks', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        work = SCRIPT.parent.parent / '.build/provenance-tests'
        work.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.binary = self.source / 'afm'
        self.binary.write_bytes(b'original executable')
        self.binary.chmod(0o755)
        self.bundle = self.source / 'AFMKit_AFMKitMLX.bundle/Contents/Resources'
        self.bundle.mkdir(parents=True)
        (self.bundle / 'default.metallib').write_bytes(b'original shaders')
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'weights')

    def pin(self):
        return module.pin_runtime(self.binary, self.root / 'runtime')

    def test_pin_preserves_resources_and_survives_source_replacement(self):
        pinned, manifest = self.pin()
        self.binary.write_bytes(b'replacement')
        (self.bundle / 'default.metallib').write_bytes(b'replacement shaders')
        self.assertEqual(pinned.read_bytes(), b'original executable')
        self.assertEqual(module.runtime_identity(pinned.parent), manifest)
        self.assertFalse(pinned.stat().st_mode & 0o222)
        self.assertTrue(pinned.stat().st_mode & 0o111)
        self.assertIn('AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib', manifest)

    def test_binary_symlink_resolves_its_actual_sibling_resources(self):
        link = self.root / 'command'
        link.symlink_to(self.binary)
        pinned, manifest = module.pin_runtime(link, self.root / 'runtime')
        self.assertEqual(pinned.name, 'afm')
        self.assertIn('AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib', manifest)

    def test_copy_mutation_is_rejected(self):
        original = module.shutil.copy2
        def changing_copy(source, target, **kwargs):
            result = original(source, target, **kwargs)
            if Path(source) == self.binary:
                self.binary.write_bytes(b'changed during copy')
            return result
        with patch.object(module.shutil, 'copy2', side_effect=changing_copy):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                self.pin()

    def test_checkpoint_content_fingerprint_changes_at_same_path(self):
        first = module.checkpoint_identity(self.model)
        (self.model / 'model.safetensors').write_bytes(b'changed')
        second = module.checkpoint_identity(self.model)
        self.assertTrue(first['weight_payloads_verified'])
        self.assertNotEqual(first['manifest_sha256'], second['manifest_sha256'])
        self.assertNotEqual(first['files']['model.safetensors']['sha256'],
                            second['files']['model.safetensors']['sha256'])

    def test_metadata_mode_does_not_read_or_claim_weight_verification(self):
        original = module.sha256_file
        def reject_weight_read(path):
            self.assertNotEqual(path.suffix, '.safetensors')
            return original(path)
        with patch.object(module, 'sha256_file', side_effect=reject_weight_read):
            identity = module.checkpoint_identity(self.model, 'metadata')
        self.assertFalse(identity['weight_payloads_verified'])
        self.assertNotIn('sha256', identity['files']['model.safetensors'])
        self.assertIn('sha256', identity['files']['config.json'])

    def test_weight_rewrite_with_restored_mtime_still_detected(self):
        first = module.checkpoint_identity(self.model, 'metadata')
        weight = self.model / 'model.safetensors'
        stat = weight.stat()
        weight.write_bytes(b'changed')
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = module.checkpoint_identity(self.model, 'metadata')
        self.assertNotEqual(module.checkpoint_mutation_identity(first),
                            module.checkpoint_mutation_identity(second))

    def test_provenance_rejects_changed_pinned_binary_and_command(self):
        pinned, manifest = self.pin()
        args = SimpleNamespace(binary=pinned, model=self.model, llmprobe=self.root / 'probe.mjs',
                               context_harness=self.root)
        harness = {'manifest_sha256': 'fixture'}
        metadata = dict(pinned_binary=str(pinned), runtime_manifest=manifest,
                        checkpoint=module.checkpoint_identity(self.model),
                        harnesses={'llmprobe': harness, 'context': harness})
        with patch.object(module, 'harness_identity', return_value=harness):
            self.assertEqual(module.verify_provenance(args, metadata)['binary_sha256'], manifest['afm'])
            args.binary = self.binary
            with self.assertRaisesRegex(RuntimeError, 'command'):
                module.verify_provenance(args, metadata)
            args.binary = pinned
            pinned.chmod(0o755)
            pinned.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'resources changed'):
                module.verify_provenance(args, metadata)

    def test_checkpoint_mutation_invalidates_phase(self):
        pinned, manifest = self.pin()
        args = SimpleNamespace(binary=pinned, model=self.model)
        metadata = dict(pinned_binary=str(pinned), runtime_manifest=manifest,
                        checkpoint=module.checkpoint_identity(self.model))
        (self.model / 'config.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(RuntimeError, 'Checkpoint changed'):
            module.verify_provenance(args, metadata)

    def test_server_refuses_unrecorded_command_before_launch(self):
        pinned, manifest = self.pin()
        args = SimpleNamespace(binary=pinned, pinned_binary_sha256=manifest['afm'], port=9999)
        with patch.object(module.socket, 'socket') as socket, \
                patch.object(module, 'server_command', return_value=[str(self.binary)]), \
                patch.object(module.subprocess, 'Popen') as launch:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'recorded pinned executable'):
                with module.server(args, 'context', self.root):
                    self.fail('Unrecorded executable was accepted')
            launch.assert_not_called()

    def git(self, *arguments):
        return subprocess.check_output(['git', '-C', str(self.root), *arguments], stderr=subprocess.DEVNULL)

    def test_harness_identity_captures_dirty_untracked_and_ignored_entrypoint(self):
        self.git('init')
        tracked = self.root / 'harness.py'
        tracked.write_text('original')
        (self.root / '.gitignore').write_text('dist/\nsource/\nmodel/\n')
        self.git('add', 'harness.py', '.gitignore')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
        distribution = self.root / 'dist'
        distribution.mkdir()
        entrypoint = distribution / 'probe.mjs'
        entrypoint.write_text('original build')
        first = module.harness_identity(self.root, entrypoint)
        tracked.write_text('dirty')
        second = module.harness_identity(self.root, entrypoint)
        self.assertEqual(first['revision'], second['revision'])
        self.assertNotEqual(first['manifest_sha256'], second['manifest_sha256'])
        self.assertIn('harness.py', second['status'])
        (self.root / 'extra.py').write_text('untracked')
        third = module.harness_identity(self.root, entrypoint)
        self.assertIn('extra.py', third['files'])
        entrypoint.write_text('new ignored build')
        fourth = module.harness_identity(self.root, entrypoint)
        self.assertNotEqual(third['manifest_sha256'], fourth['manifest_sha256'])
        self.assertIn('dist/probe.mjs', fourth['files'])


if __name__ == '__main__':
    unittest.main()
