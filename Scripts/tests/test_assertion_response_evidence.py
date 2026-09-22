"""Exercise production shell helpers against an actual HTTP error, without a model."""
import http.server
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest


ERROR_BODY = b'{"error":{"type":"server_error","message":"fixture generation failure"}}'


class ErrorHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(ERROR_BODY)))
        self.end_headers()
        self.wfile.write(ERROR_BODY)

    def log_message(self, *_args):
        pass


class AssertionResponseEvidenceTests(unittest.TestCase):
    def test_http_error_bodies_survive_all_production_helpers(self):
        source = (Path(__file__).resolve().parents[1] / "test-assertions.sh").read_text()
        curl = re.search(r"^curl\(\) \{.*?^\}", source, re.M | re.S).group()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ErrorHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for name, prefix in (("api_call", "json"), ("api_stream", "sse"), ("api_call_headers", "headers")):
                with self.subTest(helper=name), tempfile.TemporaryDirectory(prefix="afm-error-evidence-") as directory:
                    helper = re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.M | re.S).group()
                    env = dict(os.environ, RAW_REQUEST_DIR=directory, REQUEST_TIMEOUT="5",
                               BASE_URL=f"http://127.0.0.1:{server.server_port}",
                               TRANSPORT_FAILURE_FILE=f"{directory}/transport-failure",
                               REQUEST_FAILURE_FILE=f"{directory}/request-failure")
                    result = subprocess.run(
                        ["bash", "-c", f"set -euo pipefail\n{curl}\n{helper}\n{name} '{{\"messages\":[]}}'"],
                        env=env, capture_output=True, timeout=10, check=True)
                    records = [p for p in Path(directory).glob(f"{prefix}.*") if p.name.count(".") == 1]
                    self.assertEqual(len(records), 1)
                    record = records[0]
                    self.assertEqual(json.loads(Path(f"{record}.request.json").read_bytes()), {"messages": []})
                    if name == "api_call_headers":
                        self.assertEqual(Path(f"{record}.body").read_bytes(), ERROR_BODY)
                        self.assertIn(b"500", result.stdout.splitlines()[0])
                        self.assertIn("HTTP request failed", Path(env["REQUEST_FAILURE_FILE"]).read_text())
                    else:
                        self.assertEqual(result.stdout, ERROR_BODY)
                        self.assertEqual(record.read_bytes(), ERROR_BODY)
                        self.assertIn("500", Path(f"{record}.headers").read_text().splitlines()[0])
                    self.assertFalse(Path(env["TRANSPORT_FAILURE_FILE"]).exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
