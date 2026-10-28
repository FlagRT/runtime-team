"""Strict configuration front-end for existing bounded validation cases, not a trainer."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

PATHS = ('public_root', 'source', 'gems', 'deps', 'runtime', 'output', 'wheel', 'model', 'records')


def command(config):
    required = set(PATHS) | {'schema', 'model_kind', 'operator', 'mode', 'device', 'steps', 'seq_len'}
    allowed = required | {'reference', 'second_device'}
    if set(config) - allowed or required - set(config):
        raise ValueError('Missing or unknown configuration keys')
    if config['schema'] != 1 or config['operator'] != 'silu':
        raise ValueError('Only schema 1 and SiLU are supported by these model profiles')
    if type(config['device']) is not int or config['device'] < 0:
        raise ValueError('device must be a nonnegative physical device number')
    if type(config['steps']) is not int or config['steps'] not in (1, 2, 3):
        raise ValueError('steps must be 1..3')
    for key in PATHS + ('reference',):
        if key in config and (not isinstance(config[key], str) or not Path(config[key]).is_absolute()):
            raise ValueError(key + ' must be an absolute path')
    args = [sys.executable, str(Path(__file__).resolve().parents[1] / 'cases/ascend-official-unary/assets/run.py')]
    for key in PATHS:
        args += ['--' + key.replace('_', '-'), config[key]]
    args += ['--device', str(config['device']), '--steps', str(config['steps'])]
    if config['model_kind'] == 'onerec-8b':
        modes = {'native': 'native', 'off': 'disabled', 'unqualified': 'unqualified', 'on': 'target'}
        if config['mode'] not in modes or config['seq_len'] not in (16, 32):
            raise ValueError('OneRec supports native/off/unqualified/on and sequence 16/32')
        if 'second_device' in config:
            raise ValueError('OneRec profile is single-device')
        if config['mode'] != 'native' and 'reference' not in config:
            raise ValueError('A completed native reference is required')
        if config['mode'] == 'native' and 'reference' in config:
            raise ValueError('Native creates its own reference')
        args += ['--stage', 'onerec', '--onerec-mode', modes[config['mode']], '--seq-len', str(config['seq_len'])]
        if 'reference' in config:
            args += ['--reference', config['reference']]
    elif config['model_kind'] == 'qwen-embedding-0.6b':
        if config['mode'] != 'compare' or config['seq_len'] != 32:
            raise ValueError('Qwen paired probe supports compare mode and sequence 32 only')
        if 'second_device' in config or 'reference' in config:
            raise ValueError('Qwen paired probe owns its native comparison on one device')
        args += ['--stage', 'qwen']
    else:
        raise ValueError('Unsupported model profile; do not infer target eligibility')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        cmd = command(config)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    # Preserve the exact submitted configuration alongside outputs, including failures.
    output = Path(config['output'])
    if output.exists():
        parser.error('Output directory must be new')
    result = subprocess.run(cmd)
    if output.is_dir():
        (output / 'validation-config.json').write_text(json.dumps(config, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
