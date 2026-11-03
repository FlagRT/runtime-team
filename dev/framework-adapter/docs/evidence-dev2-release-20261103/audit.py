"""Audit committed source, offline wheel builds and P800 case controls.

Usage: python audit.py --repo REPO --build BUILD --rebuild REBUILD
    --hardware-wheel ORIGINAL_WHEEL --case CASE --output OUTPUT_JSON
Runs CPU controls only. Historical hardware wheel must be provided separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ('repo', 'build', 'rebuild', 'hardware-wheel', 'case', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    commit = '6ee91294e7ab028f2e804225482f58d24c8ccd5a'
    results = []
    for build in (args.build, args.rebuild):
        baseline = json.loads((build / 'local.baseline.json').read_text())
        assert baseline['overall'] == 'PASS'
        assert all(c['exit_code'] == 0 for c in baseline['commands'])
        count = 0
        for log in build.glob('test_*.log'):
            matches = re.findall(r'Ran (\d+) tests?', log.read_text())
            assert len(matches) == 1, log
            count += int(matches[0])
        assert count == 100, count
        for path, digest in baseline['source_sha256'].items():
            committed = subprocess.check_output(['git', '-C', str(args.repo), 'show', f'{commit}:{path}'])
            assert sha(committed) == digest, path
            assert sha((build / 'source' / path).read_bytes()) == digest, path
        wheel = build / 'wheels' / baseline['wheel']['filename']
        assert sha(wheel.read_bytes()) == baseline['wheel']['sha256']
        results.append({'package_controls': count, 'wheel_sha256': baseline['wheel']['sha256'],
                        'source_sha256': baseline['source_sha256'],
                        'python': baseline['python'], 'platform': baseline['platform'],
                        'dependencies': baseline['dependencies'],
                        'commands': [{'name': c['name'], 'exit_code': c['exit_code']} for c in baseline['commands']]})
    assert results[0]['source_sha256'] == results[1]['source_sha256']
    hardware_sha = sha(args.hardware_wheel.read_bytes())
    assert hardware_sha == 'de5da93ab72d00456993f74baf98a2b9aeaf1cae0a21844d2423ba4462f83008'
    with zipfile.ZipFile(args.hardware_wheel) as old:
        runtime = {n: old.read(n) for n in old.namelist() if n.startswith('flagrt_adapter/') and n.endswith('.py')}
    assert len(runtime) == 10
    for build in (args.build, args.rebuild):
        with zipfile.ZipFile(next((build / 'wheels').glob('*.whl'))) as new:
            payload = {n: new.read(n) for n in new.namelist() if n.startswith('flagrt_adapter/') and n.endswith('.py')}
            assert runtime == payload
    env = dict(os.environ, PYTHONPATH=str(args.build / 'installed'), PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(args.case / 'assets'),
                          '-p', 'test_*.py', '-v'], env=env, capture_output=True, text=True)
    (args.build / 'case-controls.log').write_text(run.stdout + run.stderr)
    assert run.returncode == 0 and 'Ran 34 tests' in run.stderr, run.stderr
    review_count = 0
    for path in args.case.glob('review*/model-reviews.json'):
        for row in json.loads(path.read_text())['reviews']:
            assert sha((path.parent / row['evidence']).read_bytes()) == row['evidence_sha256']
            review_count += 1
    for path in args.case.rglob('*.json'):
        json.loads(path.read_text())
    for path in args.case.rglob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    summary = dict(date='2026-11-03', overall='PASS', source_commit=commit, version='0.3.0.dev2',
                   scope='CPU build/install/controls and historical payload identity; no new accelerator runs',
                   builds=results, case_controls=34, model_review_evidence_hash_checks=review_count,
                   hardware_tested_wheel_sha256=hardware_sha, identical_runtime_files={n: sha(b) for n, b in runtime.items()},
                   limitations=['source candidate only, not PyPI or stable release', 'wheel archive hashes differ',
                                'new P800 env/v1 unqualified', 'training v3 target unqualified',
                                'public training CLI has no measured workload identity', 'complete three-level fallback unfinished'])
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('PASS: 100 package controls per build; 34 case controls; 10 identical runtime files;', review_count, 'review hashes')


if __name__ == '__main__':
    main()
