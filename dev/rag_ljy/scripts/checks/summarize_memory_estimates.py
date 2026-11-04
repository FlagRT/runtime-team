#!/usr/bin/env python3
"""Compare analytical estimates to attributed allocator peaks; never extrapolate coverage."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def summarize(observations):
    if not observations:
        raise ValueError('No per-forward measurements; cannot claim calibration')
    groups = defaultdict(list)
    fingerprints = set()
    for row in observations:
        for key in ('estimated_mib', 'allocator_incremental_peak_mib', 'analytical_components_mib'):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('Invalid measurement: '+key)
        if row['estimated_mib'] <= 0 or not row.get('request_id') or not row.get('model_fingerprint'):
            raise ValueError('Missing estimate, attribution or execution fingerprint')
        fingerprints.add(json.dumps(row['model_fingerprint'], sort_keys=True))
        groups[(row['tokens'], row['batch'])].append(row)
    if len(fingerprints) != 1:
        raise ValueError('Execution fingerprint changed; do not pool calibration samples')
    under = [r for r in observations if r['allocator_incremental_peak_mib'] > r['estimated_mib']]
    return {'status': 'underprediction_detected' if under else 'observed_cases_covered',
            'model_fingerprint': json.loads(next(iter(fingerprints))),
            'forward_count': len(observations), 'request_count': len({r['request_id'] for r in observations}),
            'underprediction_count': len(under),
            'max_observed_to_estimated_ratio': max(r['allocator_incremental_peak_mib']/r['estimated_mib'] for r in observations),
            'shapes': [{'tokens': t, 'batch': b, 'observations': len(rows),
                        'estimated_mib': max(r['estimated_mib'] for r in rows),
                        'max_allocator_incremental_peak_mib': max(r['allocator_incremental_peak_mib'] for r in rows),
                        'min_reserved_before_mib': min(r['reserved_before_mib'] for r in rows),
                        'max_reserved_before_mib': max(r['reserved_before_mib'] for r in rows)}
                       for (t,b),rows in sorted(groups.items())],
            'limitations': ['Only the listed shapes/requests are observed; no bound certified for untested shapes.',
                            'Torch allocator counters omit vendor allocations outside that allocator.',
                            'No automatic reduction of estimation coefficients; separate holdout validation is required.',
                            'Not the real-request 85% three-arrival-mode acceptance test.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-log', type=Path, required=True)
    p.add_argument('--request-summary', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=ROOT/'experiment_outputs/checks/estimate_calibration')
    a=p.parse_args()
    requests=json.loads(a.request_summary.read_text())
    ids={case['headers']['x-request-id'] for case in requests['cases'].values() if case['http_status']==200}
    rows=[]
    marker='event=forward_memory_observation observation='
    for line in a.server_log.read_text().splitlines():
        if marker in line:
            row=json.loads(line.split(marker,1)[1])
            if row['request_id'] in ids:rows.append(row)
    report=summarize(rows)
    if {r['request_id'] for r in rows} != ids:
        raise ValueError('Missing forward observations for a successful request')
    out=a.output_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    report['source_request_summary']=str(a.request_summary)
    report['source_log']=str(a.server_log)
    report['source_log_sha256']=hashlib.sha256(a.server_log.read_bytes()).hexdigest()
    (out/'observations.json').write_text(json.dumps(rows,indent=2))
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    shutil.copyfile(__file__,out/'summarizer.py')
    print(json.dumps({'output':str(out),'status':report['status'],'forward_count':report['forward_count'],
                      'request_count':report['request_count'],'max_observed_to_estimated_ratio':report['max_observed_to_estimated_ratio']},indent=2))
    return int(report['underprediction_count']>0)


if __name__=='__main__':raise SystemExit(main())
