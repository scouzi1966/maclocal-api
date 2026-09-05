"""CPU CLI startup regressions. Set AFM_STARTUP_TEST_BINARY after rebuilding.

No downloads or inference: unsupported local config, malformed GGUF, and a
gateway-only server exercise fatal startup and graceful SIGINT. Logs remain
under .build/server-startup-fixtures (or AFM_STARTUP_TEST_REPORT_DIR).
Without a binary, only the source-wiring guard runs; that is not runtime proof.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BINARY = os.environ.get('AFM_STARTUP_TEST_BINARY')
TIMEOUT_SECONDS = 60
POLL_SECONDS = 0.1


class SourceWiringTests(unittest.TestCase):
    def test_fatal_errors_cross_all_three_server_run_loops(self):
        source = (ROOT/'Sources/AFMCLI/main.swift').read_text()
        self.assertEqual(source.count('let serverFailure = Mutex<(any Error)?>(nil)'), 3)
        self.assertEqual(source.count('serverFailure.withLock { $0 = error }\n                shouldKeepRunning = false'), 3)
        self.assertEqual(source.count('if let error = serverFailure.withLock({ $0 }) { throw error }'), 3)
        handler = source.split('func handleShutdown(', 1)[1].split('\n}', 1)[0]
        self.assertNotIn('serverFailure', handler)


@unittest.skipUnless(BINARY, 'Set AFM_STARTUP_TEST_BINARY to a rebuilt binary; source guards alone are not runtime proof')
class StartupSubprocessTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get('AFM_STARTUP_TEST_REPORT_DIR', ROOT/'.build/server-startup-fixtures'))
        parent.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix=self._testMethodName + '-', dir=parent))
        self.env = {key: value for key, value in os.environ.items() if not key.startswith('AFM_')}
        self.env.update(HF_HUB_OFFLINE='1', MACAFM_MLX_MODEL_CACHE=str(self.work))

    def run_rejection(self, args):
        command = [str(Path(BINARY).resolve()), *args]
        result = subprocess.run(command, env=self.env, stdin=subprocess.DEVNULL,
                                text=True, capture_output=True, timeout=TIMEOUT_SECONDS)
        (self.work/'stdout.log').write_text(result.stdout)
        (self.work/'stderr.log').write_text(result.stderr)
        (self.work/'result.json').write_text(json.dumps({'command': command, 'exit_code': result.returncode}, indent=2))
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('Server shutdown complete.', result.stdout)
        self.assertIn('Error', result.stderr)

    def test_mlx_unsupported_local_config_exits_nonzero(self):
        model = self.work/'unsupported-model'
        model.mkdir()
        (model/'config.json').write_text(json.dumps({'model_type': 'afm_startup_fixture_unsupported'}))
        # No weights or tokenizer exist; model construction must reject before inference.
        self.run_rejection(['mlx', '-m', str(model), '--prewarm', 'n'])

    def test_dwarfstar_malformed_local_gguf_exits_nonzero(self):
        model = self.work/'invalid.gguf'
        model.write_bytes(b'not a GGUF checkpoint')
        self.run_rejection(['mlx', '-m', str(model), '--mlx-runtime', 'dwarfstar', '--prewarm', 'n'])

    def test_gateway_bind_failure_exits_nonzero(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            port = occupied.getsockname()[1]
            self.run_rejection(['serve', '--gateway', '--prewarm', 'n', '--port', str(port)])

    def test_gateway_sigint_exits_zero(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        command = [str(Path(BINARY).resolve()), 'serve', '--gateway', '--prewarm', 'n', '--port', str(port)]
        with (self.work/'server.log').open('w') as log:
            proc = subprocess.Popen(command, env=self.env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + TIMEOUT_SECONDS
                while True:
                    self.assertIsNone(proc.poll(), 'Gateway exited before readiness; inspect retained server.log')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            self.fail('Gateway readiness timed out')
                        time.sleep(POLL_SECONDS)
                os.killpg(proc.pid, signal.SIGINT)
                self.assertEqual(proc.wait(timeout=TIMEOUT_SECONDS), 0)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                (self.work/'result.json').write_text(json.dumps({'command': command, 'exit_code': proc.returncode}, indent=2))


if __name__ == '__main__':
    unittest.main()
