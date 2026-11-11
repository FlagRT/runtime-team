"""Offline release audit. No Git writes, networking or hardware execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

p = argparse.ArgumentParser(description=__doc__)
for name in ('package', 'team', 'build', 'other_build', 'measured_wheel', 'qwen_input', 'qwen_collection', 'output'):
    p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
a = p.parse_args()
for name, value in vars(a).items():
    setattr(a, name, value.resolve())
a.output.mkdir(parents=True, exist_ok=False)
summary = {'status': 'ERROR', 'scope': 'offline source/build/evidence controls; no new hardware run'}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def payload(wheel):
    with zipfile.ZipFile(wheel) as z:
        return {n: z.read(n) for n in z.namelist() if n.startswith('flagrt_adapter/') and n.endswith('.py')}

def test_count(directory):
    total = 0
    for log in directory.glob('test_*.log'):
        text = log.read_text()
        counts = re.findall(r'Ran (\d+) tests? in', text)
        assert len(counts) == 1 and '\nOK\n' in text, log.name
        total += int(counts[0])
    return total

try:
    commit = subprocess.check_output(['git', '-C', str(a.package), 'rev-parse', 'HEAD'], text=True).strip()
    measured = payload(a.measured_wheel)
    assert len(measured) == 16
    summary.update(commit=commit, version='0.3.0.dev4', hardware_tested_wheel_sha256=sha(a.measured_wheel.read_bytes()))
    summary['runtime_files'] = {n: sha(b) for n, b in measured.items()}
    for n, b in measured.items():
        assert (a.package / 'src' / n).read_bytes() == b
        assert subprocess.check_output(['git', '-C', str(a.package), 'show', commit + ':src/' + n]) == b
    summary['builds'] = []
    for directory in (a.build, a.other_build):
        baseline = json.loads((directory / 'local.baseline.json').read_text())
        assert baseline['overall'] == 'PASS' and all(c['exit_code'] == 0 for c in baseline['commands'])
        wheel, = (directory / 'wheels').glob('*.whl')
        assert sha(wheel.read_bytes()) == baseline['wheel']['sha256'] and payload(wheel) == measured
        for rel, expected in baseline['source_sha256'].items():
            assert sha(subprocess.check_output(['git', '-C', str(a.package), 'show', commit + ':' + rel])) == expected, rel
        count = test_count(directory)
        assert count == 178
        summary['builds'].append(dict(id=directory.name, wheel_sha256=baseline['wheel']['sha256'], package_controls=count,
                                     commands=len(baseline['commands']), source_snapshot_matches_commit=True))
    env = dict(os.environ, PYTHONPATH=str(a.build/'installed'), PYTHONDONTWRITEBYTECODE='1',
               FLAGRT_ADAPTER_INSTALLED_ONLY='1', FLAGRT_ADAPTER_INSTALL_ROOT=str(a.build/'installed'),
               QWEN_EMBEDDING_INPUT=str(a.qwen_input))
    cases = a.team/'dev/framework-adapter/cases'
    suites = [('ascend-official-unary/assets', 'test_*.py'), ('embedding', 'test_controls.py'),
              ('embedding-config', 'test_controls.py'), ('embedding-config', 'test_qwen.py'),
              ('operator-inventory', 'test_*.py')]
    summary['case_controls'] = []
    for i, (rel, pattern) in enumerate(suites):
        log = a.output/f'case-{i}.log'
        with log.open('x') as f:
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(cases/rel),
                                     '-p', pattern, '-v'], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
        assert result.returncode == 0, log.name
        count, = re.findall(r'Ran (\d+) tests? in', log.read_text())
        summary['case_controls'].append(dict(suite=rel+'/'+pattern, tests=int(count), status='PASS'))
    with (a.output/'qwen-verification.log').open('x') as f:
        result = subprocess.run([sys.executable, '-B', str(cases/'embedding-config/verify_qwen.py'),
             '--collection', str(a.qwen_collection), '--input', str(a.qwen_input),
             '--output', str(a.output/'qwen-reverified.json')], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, 'qwen evidence verification'
    qwen = json.loads((a.output/'qwen-reverified.json').read_text())
    assert qwen == json.loads((a.team/'dev/framework-adapter/docs/evidence-qwen-embedding-20261111/summary.json').read_text())
    summary['qwen_evidence'] = dict(files=qwen['verified_light_files'], archive_sha256=qwen['archive_sha256'],
                                  cpu_startup_paths=3, unchanged=True)
    summary['syntax_checked'] = 0
    for folder in ('embedding-config','embedding','operator-inventory','ppu-deepfm','ppu-models','ppu-unary','cann-shmem-validation'):
        for path in (cases/folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path)); summary['syntax_checked'] += 1
    summary['review_evidence_hashes_checked'] = 0
    evidence_dirs = [d for d in (a.team/'dev/framework-adapter/docs').glob('evidence-*')
                     if d.is_dir() and d.name[-8:].isdigit() and d.name[-8:] >= '20261104']
    for path in sorted(p for d in evidence_dirs for p in d.rglob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict): continue
        for record in data.get('records', []) if isinstance(data.get('records', []), list) else []:
            if isinstance(record,dict) and isinstance(record.get('evidence'), str) and 'evidence_sha256' in record:
                evidence = path.parent/record['evidence']
                assert sha(evidence.read_bytes()) == record['evidence_sha256'], str(evidence)
                summary['review_evidence_hashes_checked'] += 1
    summary['status'] = 'PASS'
finally:
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
