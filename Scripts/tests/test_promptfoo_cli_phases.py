"""Exercise the real runner/provider with CPU-only fake AFM and Promptfoo CLIs."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
FAKE_AFM = r'''#!/usr/bin/env python3
import json, os, signal, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
marker = Path(os.environ['FIXTURE_SERVER_MARKER'])
args = sys.argv[1:]
if '--port' not in args:
    assert not marker.exists(), 'CLI and server must not own models simultaneously'
    print(json.dumps({'argv': args}))
    sys.exit(0)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
server = HTTPServer(('127.0.0.1', int(args[args.index('--port') + 1])), Handler)
marker.write_text(json.dumps(args))
def stopped(*_): raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
try: server.serve_forever()
finally:
    marker.unlink(missing_ok=True)
    server.server_close()
'''
FAKE_PROMPTFOO = r'''#!/usr/bin/env python3
import json, os, re, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
config = Path(args[args.index('-c') + 1])
dataset = config.parent / ('datasets/quality/structured-stress.yaml'
                          if 'stress' in config.name else 'datasets/structured-core.yaml')
pattern = args[args.index('--filter-pattern') + 1]
cases = re.split(r'(?=^- description:)', dataset.read_text(), flags=re.MULTILINE)
selected = []
for case in cases:
    match = re.match(r'- description: "(.*)"', case)
    if not match or not re.search(pattern, match[1]): continue
    cli = 'transport: cli-guided-json' in case
    marker = Path(os.environ['FIXTURE_SERVER_MARKER'])
    assert marker.exists() != cli, 'phase does not match actual case provider'
    assert args[args.index('-j') + 1] == '1'
    if cli:
        result = subprocess.run(['node', os.environ['FIXTURE_PROVIDER_SCRIPT']],
                                text=True, capture_output=True, check=True)
        invocation = json.loads(result.stdout)
    else:
        invocation = json.loads(marker.read_text())
    selected.append({'testCase': {'description': match[1]}, 'success': True,
                     'response': {'metadata': {'args': invocation}}})
assert selected, 'each structured phase must select cases'
output = Path(args[args.index('-o') + 1])
output.write_text(json.dumps({'results': {'stats': {'successes': len(selected),
                                                  'failures': 0, 'errors': 0},
                                        'results': selected}}))
if os.environ.get('FIXTURE_FAIL_API') == '1' and 'cli' not in output.name:
    sys.exit(1)
'''


class PromptfooCLIPhasesTests(unittest.TestCase):
    def run_fixture(self, suite, mode, fail_api=False):
        parent = ROOT / '.build' / 'promptfoo-cli-fixtures'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            work = Path(directory)
            fake_bin = work / 'bin'
            fake_bin.mkdir()
            for name, content in [('afm', FAKE_AFM), ('promptfoo', FAKE_PROMPTFOO)]:
                executable = fake_bin / name
                executable.write_text(content)
                executable.chmod(0o755)
            provider_script = work / 'invoke-provider.mjs'
            provider = ROOT / 'Scripts/feature-promptfoo-agentic/providers/afm_provider.mjs'
            provider_script.write_text(
                f'import Provider from {json.dumps(provider.as_uri())};\n'
                'const result = await new Provider({config: {transport: "cli-guided-json"}})'
                '.callApi("fixture", {vars: {schema: {type: "object"}}});\n'
                'process.stdout.write(JSON.stringify(JSON.parse(result.output).argv));\n')
            head = work / 'mtp head $(no-shell)'
            head.mkdir()
            support = work / 'support checkpoint $(no-shell).gguf'
            support.touch()
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith('AFM_')}
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            env.update(PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                       AFM_BINARY=str(fake_bin / 'afm'), AFM_MODEL='fixture/model',
                       AFM_NO_THINK='1', AFM_PROMPTFOO_PORT=str(port),
                       AFM_PROMPTFOO_OUT_DIR=str(work / 'reports'),
                       AFM_PROMPTFOO_LOAD_TIMEOUT_SECONDS='10',
                       FIXTURE_SERVER_MARKER=str(work / 'server-alive'),
                       FIXTURE_PROVIDER_SCRIPT=str(provider_script),
                       FIXTURE_FAIL_API='1' if fail_api else '0')
            if mode == 'mtp': env.update(AFM_MTP='1', AFM_MTP_MODEL=str(head))
            if mode == 'dspark': env.update(AFM_DSPARK_SUPPORT=str(support))
            if mode == 'invalid-head': env.update(AFM_MTP_MODEL=str(head))
            result = subprocess.run(
                ['zsh', str(ROOT / 'Scripts/feature-promptfoo-agentic/run-promptfoo-agentic.sh'), suite],
                env=env, text=True, capture_output=True, timeout=30)
            if mode == 'invalid-head':
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires AFM_MTP=1', result.stderr)
                return
            self.assertEqual(result.returncode, 1 if fail_api else 0,
                             result.stdout + result.stderr)
            reports = work / 'reports'
            api = json.loads((reports / f'{suite}-api-fixture_model.json').read_text())
            cli = json.loads((reports / f'{suite}-cli-fixture_model.json').read_text())
            rows = api['results']['results'] + cli['results']['results']
            self.assertEqual(len(rows), 4 if suite.endswith('stress') else 6)
            self.assertEqual(len({row['testCase']['description'] for row in rows}), len(rows))
            for row in rows:
                args = row['response']['metadata']['args']
                self.assertEqual(args.count('--no-think'), 1)
                self.assertEqual(args.count('--mtp'), int(mode == 'mtp'))
                self.assertEqual(args.count('--dspark-support'), int(mode == 'dspark'))
                if mode == 'mtp': self.assertEqual(args[args.index('--mtp-model') + 1], str(head))
                if mode == 'dspark': self.assertEqual(args[args.index('--dspark-support') + 1], str(support))
            summary = json.loads((reports / 'promptfoo-summary-fixture_model.json').read_text())
            category = summary['categories']['nativeProtocolConformance']
            self.assertEqual(category['cases'], len(rows))
            self.assertEqual(category['fileCount'], 2)
            self.assertFalse((work / 'server-alive').exists())

    def test_both_suites_preserve_modes_and_exclusive_model_ownership(self):
        for suite in ['structured', 'structured-stress']:
            for mode in ['off', 'mtp', 'dspark']:
                with self.subTest(suite=suite, mode=mode): self.run_fixture(suite, mode)

    def test_cli_runs_after_api_failure_and_failure_status_is_retained(self):
        self.run_fixture('structured', 'mtp', fail_api=True)

    def test_invalid_head_is_rejected_before_starting_server(self):
        self.run_fixture('structured', 'invalid-head')


if __name__ == '__main__': unittest.main()
