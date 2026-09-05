"""CPU fixture servers exercise real runner capability routing and summaries."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest

from test_promptfoo_cli_phases import FAKE_AFM, ROOT

HELPER = ROOT / 'Scripts/feature-promptfoo-agentic/capability-routing.py'
SPEC = importlib.util.spec_from_file_location('routing', HELPER)
ROUTING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTING)
SERVER = FAKE_AFM.replace('self.send_response(200)',
    "self.send_response(503 if self.path.endswith('/models') and os.environ.get('FIXTURE_DISCOVERY_FAIL') else 200)")
SERVER = SERVER.replace("self.wfile.write(b'{}')", "self.wfile.write(os.environ['FIXTURE_MODELS'].encode())")
PROMPTFOO = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert Path(os.environ['FIXTURE_SERVER_MARKER']).exists()
invocation = json.loads(Path(os.environ['FIXTURE_SERVER_MARKER']).read_text())
Path(args[args.index('-o')+1]).write_text(json.dumps({'results': {
  'stats': {'successes': 1, 'failures': 0, 'errors': 0},
  'results': [{'success': True, 'testCase': {'description': 'CPU fixture'},
               'response': {'metadata': {'argv': invocation}}}]}}))
'''


class PromptfooCapabilitiesTests(unittest.TestCase):
    def run_fixture(self, mode, caps, discovery_fail=False):
        parent = ROOT / '.build' / 'promptfoo-capability-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            work = Path(directory)
            for name, content in [('afm', SERVER), ('promptfoo', PROMPTFOO)]:
                path = work / name
                path.write_text(content)
                path.chmod(0o755)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env = {key: value for key, value in os.environ.items() if not key.startswith('AFM_')}
            env.update(PATH=str(work) + os.pathsep + os.environ['PATH'],
                       AFM_BINARY=str(work/'afm'), AFM_MODEL='fixture/model',
                       AFM_PROMPTFOO_PORT=str(port), AFM_PROMPTFOO_OUT_DIR=str(work/'reports'),
                       AFM_PROMPTFOO_LOAD_TIMEOUT_SECONDS='10',
                       FIXTURE_SERVER_MARKER=str(work/'server-alive'),
                       FIXTURE_MODELS=json.dumps({'models': [{'model': 'fixture/model', 'capabilities': caps}]}))
            if discovery_fail: env['FIXTURE_DISCOVERY_FAIL'] = '1'
            result = subprocess.run(['zsh', str(ROOT/'Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh'), mode],
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertFalse((work/'server-alive').exists())
            if discovery_fail:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((work/'reports/capability-skips.json').exists())
                return
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return {path.name: json.loads(path.read_text()) for path in (work/'reports').glob('*.json')}

    def test_ds4_all_keeps_eight_native_suites_and_no_forced_profiles(self):
        reports = self.run_fixture('all', ['dwarfstar_runtime', 'tools', 'streaming', 'prefix_cache'])
        summary = reports['promptfoo-summary-fixture_model.json']
        self.assertEqual(len(summary['capabilitySkips']), 19)
        self.assertEqual(sum(item['cases'] for item in summary['categories'].values()), 8)
        for name, report in reports.items():
            if 'results' not in report: continue
            self.assertNotIn('adaptive', name)
            args = report['results']['results'][0]['response']['metadata']['argv']
            self.assertNotIn('--tool-call-parser', args)
            self.assertNotIn('--enable-grammar-constraints', args)

    def test_only_unsupported_suite_has_zero_passes_with_explicit_skip(self):
        reports = self.run_fixture('structured', ['dwarfstar_runtime', 'tools'])
        summary = reports['promptfoo-summary-fixture_model.json']
        self.assertEqual(len(summary['capabilitySkips']), 1)
        self.assertEqual(sum(item['successes'] for item in summary['categories'].values()), 0)

    def test_mlx_forced_profile_remains_enabled(self):
        reports = self.run_fixture('adaptive-xml-grammar', ['mlx_runtime', 'tools', 'structured'])
        row = reports['toolcall-adaptive-xml-grammar-fixture_model.json']['results']['results'][0]
        self.assertIn('afm_adaptive_xml', row['response']['metadata']['argv'])
        self.assertEqual(reports['capability-skips.json'], [])

    def test_discovery_transport_failure_never_becomes_skip(self):
        self.run_fixture('all', [], discovery_fail=True)

    def test_unknown_or_ambiguous_metadata_never_skips(self):
        for payload in ({}, {'models': [{'model': 'fixture', 'capabilities': []}]},
                        {'models': [{'model': 'a', 'capabilities': ['dwarfstar_runtime']},
                                    {'model': 'b', 'capabilities': ['mlx_runtime']}]},
                        {'models': [{'model': 'fixture', 'capabilities': [None]}]}):
            evidence = ROUTING.snapshot(payload, 'fixture')
            self.assertEqual(evidence['status'], 'unknown')
            self.assertEqual(ROUTING.missing_capabilities(evidence, ['structured']), [])


if __name__ == '__main__':
    unittest.main()
