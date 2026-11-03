"""Bind schema-3 review to measured model files and the actual bounded workload."""
from dataclasses import asdict
import hashlib
from pathlib import Path


def workload_from_report(report, operators, bridge_sha256):
    from flagrt_adapter.workload import measured_workload
    settings = dict(workload=report['workload'], runtime_sources=report['runtime_sources'],
                    public_config=report.get('public_config', {}),
                    adapter_bridge_sha256=bridge_sha256,
                    execution='eager first-order single-device; autocast off; tf32 off')
    return measured_workload(report['workload']['model'], 'train', report['model_files'],
                             settings, operators)


def prepare_model(a, report, operator, shape, dtype='torch.float32'):
    import flagrt_adapter
    from flagrt_adapter.config import load_config
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.startup import prepare_policies, TargetBlocked
    bridge = Path(flagrt_adapter.__file__).parent / 'bridge.py'
    report['adapter_bridge_sha256'] = hashlib.sha256(bridge.read_bytes()).hexdigest()
    operators = {operator: dict(shapes=[shape], dtypes=[dtype])}
    context = workload_from_report(report, operators, report['adapter_bridge_sha256'])
    report['workload_identity'] = asdict(context)
    actual = environment_from_record(report['environment'])
    if a.mode in ('native', 'native_repeat'):
        config, records = dict(schema=3, enabled=False, records=None, workload_reviews=None,
                               operators=operators), ()
    else:
        config, records = load_config(a.adapter_config)
        if config['schema'] != 3 or config['operators'] != operators:
            raise ValueError('Model requires schema 3 and the exact measured operator range')
        expected_enabled = a.mode != 'off'
        expected_strict = a.mode in ('on', 'strict')
        if config['enabled'] != expected_enabled or config.get('require_target', False) != expected_strict:
            raise ValueError('Config switches do not match the requested test path')
        report['adapter_config_sha256'] = hashlib.sha256(a.adapter_config.read_bytes()).hexdigest()
    report.setdefault('model_loaded', False)
    report.setdefault('optimizer_created', False)
    report['provider_calls'] = {}
    try:
        policies, report['startup'] = prepare_policies(config, records, actual, workload=context)
    except TargetBlocked as exc:
        report['startup'] = exc.report
        raise
    expected_reason = {'denied':'workload_denied', 'unknown':'workload_unverified',
                       'off':'experimental_disabled', 'unqualified':'capability_unverified'}.get(a.mode)
    if expected_reason and any(c['reason'] != expected_reason for c in report['startup']['checks']):
        raise AssertionError('Unexpected startup rejection reason')
    if a.mode == 'strict':
        raise AssertionError('Expected strict startup to block')
    return actual, policies


def expected_native_reason(mode):
    return dict(off='experimental_disabled', unqualified='capability_unverified',
                denied='workload_denied', unknown='workload_unverified')[mode]


def record_expected_block(a, report, exc):
    """An expected block is a passing negative test, never a training pass."""
    import sys
    checks = (exc.report or {}).get('checks', [])
    if (a.mode != 'strict' or not checks or any(c['reason'] != 'workload_denied' for c in checks)
            or report.get('model_loaded') or report.get('optimizer_created')
            or report.get('steps') or report.get('optimizer_steps')
            or 'flag_gems' in sys.modules):
        raise AssertionError('Not a clean, expected pre-training workload denial') from exc
    report.update(status='blocked', block_reason='workload_denied',
                  provider_imported=False, acceptance='expected startup rejection')
