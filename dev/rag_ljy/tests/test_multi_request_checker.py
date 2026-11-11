"""Exercise the client checker against real local HTTP, without models/databases."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_multi_request/check_requests.py'
spec = importlib.util.spec_from_file_location('checker', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def hits(text):
    return [{'rank': 1, 'document_id': text, 'chunk_id': text, 'title': text,
             'text': text, 'source_uri': 'test://doc', 'retrieval_ranks': {'dense': 1},
             'rerank_score': 0.9, 'rrf_score': 0.02}]


@pytest.mark.parametrize('corrupt', [False, True])
def test_http_checker_saves_results_and_detects_misrouted_responses(tmp_path, corrupt):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, content):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('X-Request-ID', uuid4().hex)
            self.end_headers()
            self.wfile.write(json.dumps(content).encode())

        def do_GET(self):
            self.respond({'status': 'ready', 'max_inflight': 8})

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['query']
            with self.server.lock:
                self.server.count += 1
                number = self.server.count
            time.sleep(0.06)
            self.respond(hits('wrong request' if corrupt and number > 2 else q))

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.lock, server.count = threading.Lock(), 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    queries = tmp_path / 'queries.jsonl'
    queries.write_text('\n'.join(json.dumps({'query_id': str(i), 'query': q})
                                 for i, q in enumerate(['nutrition', 'health'])))
    try:
        run = subprocess.run([sys.executable, str(SCRIPT), '--backend', 'milvus',
                              '--url', f'http://127.0.0.1:{server.server_port}',
                              '--queries', str(queries), '--stagger-ms', '5',
                              '--output-dir', str(tmp_path / 'results')],
                             capture_output=True, text=True, timeout=15)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert run.returncode == int(corrupt), run.stdout + run.stderr
    output = next((tmp_path / 'results').iterdir())
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == ('failed' if corrupt else 'passed')
    assert summary['checks']['simultaneous']['overlapping_http_pairs'] == 1
    assert summary['checks']['staggered']['overlapping_http_pairs'] == 1
    assert summary['started_at'].endswith('+08:00')
    for scenario in ('serial', 'simultaneous', 'staggered'):
        rows = json.loads((output / (scenario + '.json')).read_text())
        assert len(rows) == 2 and all(r['results'] for r in rows)


def test_validator_rejects_missing_ids_failed_requests_and_no_overlap():
    row = {'query_id': 'a', 'request_id': 'id', 'status': 200, 'results': hits('a'),
           'sent_ms': 0, 'completed_ms': 1}
    bad = {**row, 'request_id': None, 'status': 503}
    checks = checker.validate({'serial': [row], 'simultaneous': [bad]})
    assert checks['serial']['passed']
    assert not checks['simultaneous']['passed']
    assert len(checks['simultaneous']['failures']) == 3


def test_score_tolerance_and_ordered_identity():
    a, b = hits('a'), hits('a')
    b[0]['rerank_score'] += 1e-7
    assert checker.same_results(a, b)
    b[0]['chunk_id'] = 'b'
    assert not checker.same_results(a, b)
    assert not checker.same_results([], [])
