"""Verify dev3 committed sources, installed builds and hardware evidence hashes.

python audit.py --repo REPO --build BUILD --rebuild REBUILD
    --hardware-wheel WHEEL --direction DIRECTION --output NEW_JSON
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


def payload(path):
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist()
                if n.startswith('flagrt_adapter/') and n.endswith('.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'build', 'rebuild', 'hardware-wheel', 'direction', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    commit = subprocess.check_output(['git','-C',str(args.repo),'rev-parse','40864f9^{commit}'], text=True).strip()
    builds = []
    original = payload(args.hardware_wheel)
    hardware_sha = sha(args.hardware_wheel.read_bytes())
    assert hardware_sha == '191887c6418a3ec09a03b4160778bdbadb11262d8b70473cd096e08ddb0e57ca'
    assert len(original) == 11
    for build in (args.build, args.rebuild):
        baseline = json.loads((build/'local.baseline.json').read_text())
        assert baseline['overall']=='PASS' and all(c['exit_code']==0 for c in baseline['commands'])
        count = sum(int(re.search(r'Ran (\d+) tests?', p.read_text()).group(1)) for p in build.glob('test_*.log'))
        assert count == 110, count
        for path, digest in baseline['source_sha256'].items():
            data = subprocess.check_output(['git','-C',str(args.repo),'show',f'{commit}:{path}'])
            assert sha(data)==digest==sha((build/'source'/path).read_bytes()), path
        wheel = build/'wheels'/baseline['wheel']['filename']
        assert sha(wheel.read_bytes())==baseline['wheel']['sha256']
        assert payload(wheel)==original
        builds.append(dict(package_controls=count, wheel_sha256=baseline['wheel']['sha256'],
            source_sha256=baseline['source_sha256'], dependencies=baseline['dependencies'],
            commands=[dict(name=c['name'],exit_code=c['exit_code']) for c in baseline['commands']]))
    assert builds[0]['source_sha256']==builds[1]['source_sha256']
    for name, data in original.items():
        assert subprocess.check_output(['git','-C',str(args.repo),'show',f'{commit}:src/{name}'])==data
    env = dict(os.environ, PYTHONPATH=str(args.build/'installed'), PYTHONDONTWRITEBYTECODE='1')
    controls = {}
    for case, pattern in [('ascend-official-unary','test_check_training.py'),
                          ('ascend-official-unary','test_public_audit.py'),
                          ('kunlun-unary','test_*.py')]:
        run = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
            str(args.direction/'cases'/case/'assets'),'-p',pattern], env=env, text=True, capture_output=True)
        assert run.returncode==0, run.stderr
        controls[case+'/'+pattern] = int(re.search(r'Ran (\d+) tests?',run.stderr).group(1))
    assert sorted(controls.values()) == [2,6,34], controls
    evidence = args.direction/'docs/evidence-public-entry-20261104'
    summary = json.loads((evidence/'summary.json').read_text())
    for name, digest in summary['files_sha256'].items():
        assert sha((evidence/name).read_bytes())==digest, name
    reviews = args.direction/'cases/ascend-official-unary/review-public-20261104'
    for row in json.loads((reviews/'model-reviews.json').read_text())['reviews']:
        assert sha((reviews/row['evidence']).read_bytes())==row['evidence_sha256']
    result = dict(overall='PASS', date='2026-11-04', source_commit=commit, version='0.3.0.dev3',
        builds=builds, case_controls=controls, hardware_evidence_files=len(summary['files_sha256']),
        hardware_tested_wheel_sha256=hardware_sha,
        identical_runtime_files={name:sha(data) for name,data in original.items()},
        hardware_report='../公共训练入口模型审核接入-20261104.md',
        scope='Committed dev3 build/install/CPU controls and 910C hardware payload/evidence identity')
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2); stream.write('\n')
    print('PASS: 110 package controls/build; 42 case controls; 11 identical runtime files;',len(summary['files_sha256']),'evidence hashes')


if __name__=='__main__':
    main()
