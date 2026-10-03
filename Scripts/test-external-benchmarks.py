#!/usr/bin/env python3
"""Run llmprobe and Context against a candidate, preserving failures and raw evidence."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

CONTEXTS = ['0.5k', '1k', '2k', '4k', '8k', '16k', '32k']
RUNS = 2
LOAD_TIMEOUT = 900
TEST_TIMEOUT = 10800
STOP_TIMEOUT = 30


def existing_path(path):
    # Validate the target without replacing a virtualenv's Python symlink:
    # invoking its resolved base interpreter loses the environment's packages.
    path.resolve(strict=True)
    return path.absolute()


def validate_context(root):
    """The upstream process can exit zero after partial runs; require every trial."""
    errors = []
    sections = re.split(r'Benchmarking ([\d.]+k)\.txt \.\.\.', (root / 'context.log').read_text())
    if sections[1::2] != CONTEXTS:
        errors.append('Missing or reordered context sizes')
    for context, body in zip(sections[1::2], sections[2::2]):
        runs = re.split(r'  Run (\d+)/(\d+)\.\.\.', body)
        if len(runs) != 1 + 3 * RUNS:
            errors.append(f'{context}: incomplete trial count')
        for i in range(1, len(runs), 3):
            number, total, sample = runs[i:i + 3]
            speed = re.findall(r'^\s+generation_tps: ([\d.]+)\s*$', sample, re.M)
            tokens = re.findall(r'^\s+Completion tokens:\s+(\d+)\s*$', sample, re.M)
            if total != str(RUNS) or not speed or float(speed[0]) <= 0 or not tokens or int(tokens[0]) <= 0:
                errors.append(f'{context} trial {number}: missing generation')
    artifacts = list(root.glob('benchmark_openai_*/benchmark_results.csv'))
    if len(artifacts) != 1:
        errors.append('Expected exactly one results CSV')
    else:
        with artifacts[0].open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if [r['context_size'] for r in rows] != CONTEXTS:
            errors.append('Incomplete CSV')
        for row in rows:
            for metric in ['generation_tps', 'prompt_tps_latency_adjusted', 'host_memory_gb']:
                value = float(row[metric])
                if not math.isfinite(value) or value <= 0:
                    errors.append(f'{row["context_size"]}: invalid {metric}')
        for context in CONTEXTS:
            response = artifacts[0].parent / f'response_{context}.txt'
            if not response.is_file() or not response.read_text().strip():
                errors.append(f'{context}: missing response')
    return {'passed': not errors, 'errors': errors, 'expected_contexts': CONTEXTS,
            'note': 'Completeness only; speed and response quality require review.'}


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def listener_owned_by(process, port):
    result = subprocess.run(['lsof', '-nP', '-t', f'-iTCP:{port}', '-sTCP:LISTEN'],
                            capture_output=True, text=True)
    owners = {int(value) for value in result.stdout.split()}
    if owners and owners != {process.pid}:
        raise RuntimeError(f'Port {port} belongs to another process; refusing to test it')
    return owners == {process.pid}


def run(command, log, cwd=None, owner=None):
    with log.open('w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   cwd=cwd, start_new_session=True)
        try:
            deadline = time.monotonic() + TEST_TIMEOUT
            while time.monotonic() < deadline:
                if owner is not None and owner.poll() is not None:
                    raise RuntimeError('Candidate server exited during the test')
                try:
                    return process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            raise TimeoutError('Benchmark timed out')
        finally:
            stop(process)


def server_command(args, phase):
    command = [str(args.binary), 'mlx', '-m', str(args.model), '--hostname', '127.0.0.1',
               '--port', str(args.port), '--enable-grammar-constraints']
    if args.mtp:
        command += ['--mtp', '--mtp-depth', str(args.mtp_depth)]
    if args.prefill_step_size is not None:
        command += ['--prefill-step-size', str(args.prefill_step_size)]
    if args.concurrent_capacity != 1:
        command += ['--concurrent', str(args.concurrent_capacity)]
    if phase == 'context':
        command.append('--no-think')
    return command


@contextlib.contextmanager
def server(args, phase, output):
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError(f'Port {args.port} is occupied; no existing process will be stopped')
    command = server_command(args, phase)
    (output / 'server-command.json').write_text(json.dumps(command, indent=2))
    with (output / 'server.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + LOAD_TIMEOUT
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited: {process.returncode}')
                if not listener_owned_by(process, args.port):
                    time.sleep(1)
                    continue
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=3) as response:
                        models = json.load(response)
                    (output / 'models.json').write_text(json.dumps(models, indent=2))
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise RuntimeError('Model load timed out')
            yield process
        finally:
            stop(process)


def context_worker():
    harness, output, *arguments = sys.argv[2:]
    sys.path.insert(0, harness)
    import benchmark_common as common
    import openai_benchmark
    original_stream = common.stream_chat
    original_directory = common.create_output_directory
    def stream(*args, **kwargs):
        kwargs['top_p'] = 1.0
        kwargs['extra_body'] = {**kwargs.get('extra_body', {}), 'chat_template_kwargs': {'enable_thinking': False}}
        return original_stream(*args, **kwargs)
    def directory(*args, **kwargs):
        kwargs['base_dir'] = output
        return original_directory(*args, **kwargs)
    common.stream_chat = stream
    common.create_output_directory = directory
    sys.argv = [sys.argv[0], *arguments]
    return openai_benchmark.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--llmprobe', type=Path, required=True, help='Path to bin/dist/llmprobe.mjs')
    parser.add_argument('--context-harness', type=Path, required=True)
    parser.add_argument('--context-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--phase', choices=['all', 'llmprobe', 'context'], default='all')
    parser.add_argument('--mtp', action='store_true', help='Exercise MTP in both external suites')
    parser.add_argument('--mtp-depth', type=int, default=3)
    parser.add_argument('--prefill-step-size', type=int)
    parser.add_argument('--concurrent-capacity', type=int, default=1,
                        help='AFM server capacity; Context still sends one request at a time')
    args = parser.parse_args()
    if args.mtp_depth < 1 or (args.prefill_step_size is not None and args.prefill_step_size < 1):
        parser.error('MTP depth and explicit prefill step size must be positive')
    if args.concurrent_capacity < 1:
        parser.error('Concurrent server capacity must be positive')
    for name in ['binary', 'model', 'llmprobe', 'context_harness', 'context_python']:
        setattr(args, name, existing_path(getattr(args, name)))
    if args.phase in ('all', 'context'):
        dependencies = subprocess.run(
            [str(args.context_python), '-c', 'import openai, matplotlib, numpy, psutil'],
            capture_output=True, text=True)
        if dependencies.returncode:
            parser.error('Context Python requires openai, matplotlib, numpy and psutil; install them before starting model tests')
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                'model': str(args.model), 'speculation': 'mtp' if args.mtp else 'off',
                'mtp_depth': args.mtp_depth if args.mtp else None,
                'prefill_step_size': args.prefill_step_size,
                'server_concurrent_capacity': args.concurrent_capacity,
                'qwen_environment': {k: v for k, v in os.environ.items() if k.startswith('AFM_QWEN_')},
                'harnesses': {}}
    for name, path in [('llmprobe', args.llmprobe.parent), ('context', args.context_harness)]:
        metadata['harnesses'][name] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results = {}
    phases = ['llmprobe', 'context'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        output = args.output / phase
        output.mkdir()
        base = f'http://127.0.0.1:{args.port}'
        if phase == 'llmprobe':
            command = ['node', str(args.llmprobe), base, '--model', str(args.model), '--full', '--no-bench',
                       '--reasoning', 'medium', '--save', str(output / 'llmprobe.json'),
                       '--html', str(output / 'llmprobe.html'), '--library', str(output / 'library')]
        else:
            command = [str(args.context_python), str(Path(__file__).resolve()), '--context-worker',
                       str(args.context_harness), str(output), '--model', str(args.model), '--base-url', base + '/v1',
                       '--temperature', '0', '--contexts', ','.join(v[:-1] for v in CONTEXTS),
                       '--context-type', 'prose', '--runs', str(RUNS), '--max-tokens', '128',
                       '--timeout', '3600', '--cold-prefill', '--no-batch', '--save-responses']
        (output / 'test-command.json').write_text(json.dumps(command, indent=2))
        print(f'Starting {phase}: {output}', flush=True)
        try:
            with server(args, phase, output) as owner:
                code = run(command, output / f'{phase}.log', args.context_harness if phase == 'context' else None, owner=owner)
            result = {'exit_code': code, 'passed': code == 0}
            if phase == 'context':
                completeness = validate_context(output)
                (output / 'completeness.json').write_text(json.dumps(completeness, indent=2))
                result['passed'] = result['passed'] and completeness['passed']
            elif not (output / 'llmprobe.json').is_file():
                result['passed'] = False
                result['error'] = 'Missing llmprobe report'
        except Exception as error:
            result = {'passed': False, 'error': str(error)}
        results[phase] = result
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        print(f'{phase}: {result}', flush=True)
    return int(not all(result['passed'] for result in results.values()))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(context_worker() if sys.argv[1:2] == ['--context-worker'] else main())
