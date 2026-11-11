"""Verify copied immutable evidence and fail-closed model outcomes offline."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    collection, evidence = root / 'collection01', root / 'evidence'
    expected_archive = (collection / 'archive.sha256').read_text().split()[0]
    assert sha(collection / 'evidence.tar.gz') == expected_archive
    files = json.loads((collection / 'evidence-sha.json').read_text())
    for name, expected in files.items():
        path = (evidence / name).resolve()
        assert path.is_relative_to(evidence.resolve()), name
        assert sha(path) == expected, name
    summary = json.loads((collection / 'summary.json').read_text())
    for name, manifest in summary['inputs'].items():
        for relative, expected in manifest.items():
            assert sha(root / name / relative) == expected, (name, relative)
    checked = {}
    for model, count in [('qwen', 1863), ('onerec', 2397)]:
        qualification = f'{model}-qual-' + ('02' if model == 'qwen' else '01')
        q = json.loads((evidence / qualification / 'result.json').read_text())
        assert q['status'] == 'PASS' and len(q['cases']) == 3, qualification
        for row in q['cases']:
            assert row['status'] == 'pass' and row['restored'], qualification
            assert row['hits'] == {'GEMS SILU FORWARD': 1, 'GEMS SILU BACKWARD': 1}
        repeat = '02' if model == 'onerec' else '01'
        ids = {'native': '01', 'repeat': repeat, 'off': '01', 'unknown': '01'}
        for mode, suffix in ids.items():
            name = f'{model}-{mode}-{suffix}'
            run = summary['runs'][name]
            data = json.loads((evidence / name / 'result.json').read_text())
            assert run['host_status'] == 'PASS' and data['status'] == 'pass', name
            assert data['optimizer_steps'] == 3 and len(data['steps']) == 3, name
            assert len(data['comparisons']) == count, name
            assert data['dispatcher_restored'], name
            assert not any(data['provider_calls'].values()), name
            if mode != 'native':
                assert all(row['max_abs'] == 0 and row['mismatches'] == 0
                           for row in data['comparisons']), name
            if mode in ('off', 'unknown'):
                reason = 'experimental_disabled' if mode == 'off' else 'workload_unverified'
                checks = data['startup']['checks']
                assert checks and all(row['decision'] == 'native' and row['reason'] == reason
                                      for row in checks), (name, checks)
            checked[name] = {'status': 'PASS', 'steps': 3, 'comparisons': count}
        name = f'{model}-candidate-01'
        data = json.loads((evidence / name / 'result.json').read_text())
        assert data['status'] == 'fail' and data['optimizer_steps'] == 0, name
        assert data['dispatcher_restored'] and any(data['provider_calls'].values()), name
        assert any(row.get('mismatches', 0) for row in data['comparisons']), name
        checked[name] = {'status': 'NUMERICAL_FAILURE_RECORDED', 'optimizer_steps': 0}
        native = f'{model}-native-01'
        tensors = json.loads((collection / (native + '-tensor-sha.json')).read_text())
        assert len(tensors) == count, native
    for name, run in summary['runs'].items():
        if run['stop']:
            assert not run['stop']['Running'] and run['stop']['Pid'] == 0, name
            assert not run['stop']['OOMKilled'], name
    return dict(status='PASS', copied_files=len(files), checked_runs=checked,
                archive_sha256=expected_archive,
                tensor_backup='remote only; local manifests verified, not local tensor copies')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
