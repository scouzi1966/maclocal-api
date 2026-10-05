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
import shutil
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
HASH_CHUNK_BYTES = 1024 * 1024
WEIGHT_SUFFIXES = {'.safetensors', '.gguf', '.bin', '.npz'}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_identity(path, content=True):
    before = path.stat()
    record = dict(size=before.st_size, mtime_ns=before.st_mtime_ns,
                  ctime_ns=before.st_ctime_ns, inode=before.st_ino, device=before.st_dev,
                  resolved_path=str(path.resolve(strict=True)))
    if content:
        record['sha256'] = sha256_file(path)
    after = path.stat()
    if any(getattr(before, field) != getattr(after, field)
           for field in ('st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_ino', 'st_dev', 'st_mode')):
        raise RuntimeError(f'Input changed while fingerprinting: {path}')
    return record


def checkpoint_identity(model, verification='sha256'):
    """Metadata mode never claims weight payloads have been content-verified."""
    files = {}
    for path in sorted(model.rglob('*')):
        relative = path.relative_to(model)
        if any(part.startswith('.') for part in relative.parts) or not path.is_file():
            continue
        files[str(relative)] = file_identity(
            path, content=verification == 'sha256' or path.suffix not in WEIGHT_SUFFIXES)
    if 'config.json' not in files or not any(Path(name).suffix in WEIGHT_SUFFIXES for name in files):
        raise ValueError('Checkpoint must contain config.json and local weight files')
    content = {name: {'sha256': record['sha256'], 'size': record['size']}
               if 'sha256' in record else record for name, record in files.items()}
    return dict(verification=verification, weight_payloads_verified=verification == 'sha256',
                manifest_sha256=manifest_digest(content), files=files)


def checkpoint_mutation_identity(identity):
    # Avoid rereading hundreds of GB between phases. Config/tokenizer hashes and
    # weight file identity, size, mtime and ctime still detect ordinary mutations.
    return {name: {key: value for key, value in record.items()
                   if key != 'sha256' or Path(name).suffix not in WEIGHT_SUFFIXES}
            for name, record in identity['files'].items()}


def harness_identity(path, entrypoint=None):
    root = Path(subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', '--show-toplevel'], text=True).strip())
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    names = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).split(b'\0')
    paths = {root / os.fsdecode(name) for name in names if name}
    # Generated distribution files may be ignored by Git but are what Node runs.
    if entrypoint is not None:
        paths.update(p for p in entrypoint.parent.rglob('*') if p.is_file())
    files = {str(p.relative_to(root)): file_identity(p) if p.is_file() else {'missing': True}
             for p in sorted(paths)}
    return dict(revision=revision, status=status, files=files,
                manifest_sha256=manifest_digest(files))


def runtime_identity(directory):
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def pin_runtime(binary, destination):
    """Copy the relocatable release layout, dereferencing mutable source links."""
    binary = binary.resolve(strict=True)
    destination.mkdir()
    sources = [binary] + [path for path in sorted(binary.parent.iterdir())
                          if path != binary and path.suffix in {'.bundle', '.dylib', '.metallib'}]
    before = {}
    for source in sources:
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        before.update({str(p.relative_to(binary.parent)): file_identity(p)
                       for p in paths if p.is_file()})
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)
    expected = {name: identity['sha256'] for name, identity in before.items()}
    if runtime_identity(destination) != expected:
        raise RuntimeError('Runtime changed while copying the pinned release layout')
    for name, identity in before.items():
        if file_identity(binary.parent / name) != identity:
            raise RuntimeError('Source runtime changed while pinning')
    for path in destination.rglob('*'):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    return destination / binary.name, expected


def verify_provenance(args, metadata):
    if str(args.binary) != metadata['pinned_binary']:
        raise RuntimeError('Server executable does not match the pinned command')
    runtime = runtime_identity(args.binary.parent)
    if runtime != metadata['runtime_manifest']:
        raise RuntimeError('Pinned executable or runtime resources changed')
    checkpoint = checkpoint_identity(args.model, verification='metadata')
    if checkpoint_mutation_identity(checkpoint) != checkpoint_mutation_identity(metadata['checkpoint']):
        raise RuntimeError('Checkpoint changed during benchmark qualification')
    harnesses = {}
    for name, path, entrypoint in [('llmprobe', args.llmprobe.parent, args.llmprobe),
                                    ('context', args.context_harness, None)]:
        harnesses[name] = harness_identity(path, entrypoint)
        if harnesses[name] != metadata['harnesses'][name]:
            raise RuntimeError(f'{name} harness changed during benchmark qualification')
    return dict(binary_sha256=runtime[args.binary.name], runtime_manifest=runtime,
                checkpoint_mutation_manifest_sha256=manifest_digest(checkpoint_mutation_identity(checkpoint)),
                harness_manifest_sha256={name: identity['manifest_sha256']
                                         for name, identity in harnesses.items()})


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
        if runs[1::3] != [str(number) for number in range(1, RUNS + 1)]:
            errors.append(f'{context}: missing, duplicate or reordered trial identifiers')
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


def validate_llmprobe(report, expect_vision=False):
    """A 100% headline can hide HTTP 500s classified as unsupported capabilities."""
    errors = []
    # Full conformance registry in llmprobe 3952b5d9 (conformance/index.ts,
    # shared.ts and surfaces.ts). These are case identities, NOT the assertion
    # denominator: JSON v2 omits successful assertions, and one case may assert
    # several MUSTs. Keep this coverage contract when upgrading the harness.
    shared = set('''basic streaming parity usage stream-usage finish-length
        limits-max-tokens limits-stop unicode errors tool-serialization
        tool-stream-reassembly tool-result-turn parallel-tools tool-choice-none
        parallel-tools-off tool-arg-types json-mode structured-outputs
        structured-markers structured-terminates tool-args-literal-delimiter
        vision logprobs seed top-p reasoning reasoning-cap reasoning-scratchpad
        reasoning-roundtrip prompt-caching prompt-cache-prefix rate-limit-headers
        concurrency concurrency-cache'''.split())
    expected = {'models-list', 'count-tokens', 'embeddings-basic', 'embeddings-dimensions',
                'completions-basic', 'images-generate', 'images-edit', 'audio-speech'}
    for surface in ('chat', 'responses', 'messages'):
        omitted = {'logprobs', 'seed'} if surface != 'chat' else set()
        if surface == 'responses':
            omitted.add('limits-stop')
        if surface == 'messages':
            omitted.update(('json-mode', 'structured-outputs', 'structured-markers', 'structured-terminates'))
        expected.update(f'{surface}-{name}' for name in shared - omitted)
    expected.update('''chat-n-choices chat-max-tokens-alias chat-assistant-prefill
        chat-stop-string chat-template-kwargs chat-sampling-extensions
        responses-reasoning-effort-none responses-event-order responses-previous-response-id
        responses-background responses-mcp-tools messages-event-order
        messages-max-tokens-required messages-stop-sequence-echo messages-assistant-prefill
        messages-thinking-budget messages-system-blocks messages-content-blocks
        messages-top-k messages-cache-control messages-error-envelope'''.split())
    if not isinstance(report, dict):
        return {'passed': False, 'errors': ['Malformed llmprobe report']}
    run = report.get('run', {})
    def object_value(value):
        return value if isinstance(value, dict) else {}
    run = object_value(run)
    phases = object_value(run.get('phases'))
    if (report.get('version') != 2 or not isinstance(run, dict)
            or run.get('depth') != 'full' or run.get('mode') != 'probe'
            or object_value(run.get('budget')).get('exhausted') is not False
            or any(object_value(phases.get(phase)).get('status') != 'measured'
                   for phase in ('coverage', 'conformance', 'capability', 'agentic', 'fidelity'))):
        errors.append('Missing or incomplete full probe run metadata')
    conformance = report.get('conformance', {})
    if not isinstance(conformance, dict):
        conformance = {}
    total, passed = conformance.get('total'), conformance.get('passed')
    if type(total) is not int or total <= 0 or type(passed) is not int or passed != total:
        errors.append('Incomplete or failing mandatory conformance')
    rows = conformance.get('results', [])
    cases = {}
    if not isinstance(rows, list):
        errors.append('Malformed conformance results')
        rows = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                or not isinstance(row.get('surface'), str)):
            errors.append('Malformed conformance case')
            continue
        if row['id'] in cases:
            errors.append(f'Duplicate conformance case: {row["id"]}')
        cases[row['id']] = row
        if row.get('outcome') not in ('pass', 'unsupported', 'inconclusive'):
            errors.append(f'{row["id"]}: failed, skipped or invalid outcome')
        failures = row.get('failures')
        if not isinstance(failures, list) or any(
                not isinstance(failure, dict) or failure.get('severity') == 'MUST'
                for failure in failures):
            errors.append(f'{row["id"]}: malformed or failed mandatory assertions')
    if expected - cases.keys():
        errors.append('Missing full conformance cases: ' + ', '.join(sorted(expected - cases.keys())))
    surfaces = conformance.get('bySurface', [])
    if (not isinstance(surfaces, list) or not surfaces
            or any(not isinstance(row, dict) or not isinstance(row.get('surface'), str)
                   or type(row.get('total')) is not int or row['total'] < 0
                   or type(row.get('passed')) is not int or row['passed'] != row['total']
                   for row in surfaces)):
        errors.append('Malformed or failing per-surface conformance')
    elif (len({row['surface'] for row in surfaces}) != len(surfaces)
          or sum(row['total'] for row in surfaces) != total
          or sum(row['passed'] for row in surfaces) != passed
          or {row.get('surface') for row in cases.values()
              if row.get('outcome') in ('pass', 'fail')} != {row['surface'] for row in surfaces}):
        errors.append('Conformance surface totals disagree with results/headline')
    if expect_vision:
        for case in ('chat-vision', 'responses-vision', 'messages-vision'):
            result = cases.get(case, {})
            if result.get('outcome') != 'pass':
                errors.append(f'{case}: expected checkpoint capability not passed: '
                              f'{result.get("reason", result.get("outcome", "missing"))}')
    return {'passed': not errors, 'errors': errors,
            'note': 'Protocol/capability coverage gate; agentic quality still needs separate review.'}


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


def checkpoint_has_vision(model):
    """Require both configuration and present indexed vision weight shards."""
    config = model / 'config.json'
    index = model / 'model.safetensors.index.json'
    if not config.is_file() or not index.is_file():
        return False
    if not json.loads(config.read_text()).get('vision_config'):
        return False
    weights = json.loads(index.read_text()).get('weight_map', {})
    shards = {value for key, value in weights.items()
              if re.search(r'(^|\.)(vision_tower|visual|vision_model)\.', key)}
    return bool(shards) and all((model / shard).is_file() for shard in shards)


def server_command(args, phase):
    command = [str(args.binary), 'mlx', '-m', str(args.model), '--hostname', '127.0.0.1',
               '--port', str(args.port), '--enable-grammar-constraints']
    if getattr(args, 'vlm', False) or checkpoint_has_vision(args.model):
        command.append('--vlm')
    if getattr(args, 'qwen_mtp_profile', None):
        command += ['--qwen-mtp-profile', args.qwen_mtp_profile]
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
    if (command[0] != str(args.binary)
            or sha256_file(Path(command[0])) != args.pinned_binary_sha256):
        raise RuntimeError('Server command does not use the recorded pinned executable')
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
    parser.add_argument('--checkpoint-verification', choices=['sha256', 'metadata'], default='sha256',
                        help='sha256 reads all checkpoint payloads once before testing; metadata only hashes non-weight files and records weight file stats (not content verification)')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--phase', choices=['all', 'llmprobe', 'context'], default='all')
    parser.add_argument('--mtp', action='store_true', help='Exercise MTP in both external suites')
    parser.add_argument('--mtp-depth', type=int, default=3)
    parser.add_argument('--vlm', action='store_true', help='Enable vision explicitly; indexed vision checkpoints are detected automatically')
    parser.add_argument('--qwen-mtp-profile', choices=['off', 'throughput-v1', 'throughput-v2'])
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
    if os.environ.get('MACAFM_MLX_METALLIB'):
        parser.error('Unset MACAFM_MLX_METALLIB; qualification uses the pinned runtime bundle')
    original_binary = str(args.binary)
    args.binary, runtime_manifest = pin_runtime(args.binary, args.output / 'runtime')
    args.pinned_binary_sha256 = runtime_manifest[args.binary.name]
    if not any((args.binary.parent / name).is_file() for name in (
            'default.metallib', 'AFMKit_AFMKitMLX.bundle/default.metallib',
            'AFMKit_AFMKitMLX.bundle/Contents/Resources/default.metallib')):
        parser.error('Candidate requires a sibling MLX metallib or resource bundle for relocation')
    metadata = {'binary_sha256': runtime_manifest[args.binary.name],
                'original_binary': original_binary, 'pinned_binary': str(args.binary),
                'runtime_manifest': runtime_manifest,
                'checkpoint': checkpoint_identity(args.model, args.checkpoint_verification),
                'checkpoint_mutation_check': 'non-weight SHA-256; weight size/inode/device/mtime/ctime before and after each phase',
                'model': str(args.model), 'speculation': 'mtp' if args.mtp else 'off',
                'mtp_depth': args.mtp_depth if args.mtp else None,
                'prefill_step_size': args.prefill_step_size,
                'server_concurrent_capacity': args.concurrent_capacity,
                'vision_enabled': args.vlm or checkpoint_has_vision(args.model),
                'qwen_mtp_profile': args.qwen_mtp_profile,
                'qwen_environment': {k: v for k, v in os.environ.items() if k.startswith('AFM_QWEN_')},
                'harnesses': {}}
    for name, path, entrypoint in [('llmprobe', args.llmprobe.parent, args.llmprobe),
                                    ('context', args.context_harness, None)]:
        metadata['harnesses'][name] = harness_identity(path, entrypoint)
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
            before = verify_provenance(args, metadata)
            (output / 'provenance-before.json').write_text(json.dumps(before, indent=2))
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
            else:
                coverage = validate_llmprobe(json.loads((output / 'llmprobe.json').read_text()),
                                            expect_vision=args.vlm or checkpoint_has_vision(args.model))
                result['coverage'] = coverage
                result['passed'] = result['passed'] and coverage['passed']
        except Exception as error:
            result = {'passed': False, 'error': str(error)}
        try:
            after = verify_provenance(args, metadata)
            (output / 'provenance-after.json').write_text(json.dumps(after, indent=2))
            result['provenance_verified'] = True
        except Exception as error:
            result.update(passed=False, provenance_verified=False, provenance_error=str(error))
        results[phase] = result
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
        print(f'{phase}: {result}', flush=True)
    return int(not all(result['passed'] for result in results.values()))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(context_worker() if sys.argv[1:2] == ['--context-worker'] else main())
