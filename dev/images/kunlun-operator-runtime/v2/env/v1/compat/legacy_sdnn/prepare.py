"""Generate a hash-pinned legacy SDK compiler COPY; never alter the SDK."""
import argparse
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASELINE = '7133710c0153c52dfe17822eb03cfd026a19c7c861b293dd3c9e818b6a64abef'
EXPECTED = {
    # Canonical LF; historical experimental copies had CRLF/mixed newlines.
    'gemm': '6091230312bd05ff97a29bc10c09146c57818ea1c5f668958332b59b3a008f1d',
    'epilogue': 'dbb2852ac6a9632c6d4d47b8aa958edad44066400489a89003698bab09394099',
    # Diagnostic/pass-control experiment, based on the intrinsic profile.
    'diagnostic': 'ba58858fa8f86023ff6c6c61a44142b833545163424fb97485236c6e9614e8d3',
}


def apply_exact(source, patch):
    """Apply unified hunks at their exact original offsets, with no fuzz."""
    lines = source.splitlines(keepends=True)
    delta = patch.splitlines(keepends=True)
    result, cursor, i = [], 0, 2
    while i < len(delta):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n', delta[i])
        if not match:
            raise ValueError('Malformed unified hunk')
        start = int(match[1]) - 1
        if start < cursor:
            raise ValueError('Overlapping patch hunks')
        result.extend(lines[cursor:start])
        cursor = start
        i += 1
        old_count = new_count = 0
        while i < len(delta) and not delta[i].startswith('@@ '):
            line = delta[i]
            if line[0] in ' -':
                if cursor >= len(lines) or lines[cursor] != line[1:]:
                    raise ValueError(f'Patch context mismatch at line {cursor + 1}')
                cursor += 1
                old_count += 1
            if line[0] in ' +':
                result.append(line[1:])
                new_count += 1
            if line[0] not in ' +-':
                raise ValueError('Unsupported patch line')
            i += 1
        if old_count != int(match[2] or 1) or new_count != int(match[4] or 1):
            raise ValueError('Patch hunk size mismatch')
    result.extend(lines[cursor:])
    return ''.join(result)


def prepare(source, profile):
    if hashlib.sha256(source).hexdigest() != BASELINE:
        raise ValueError('SDK compiler SHA256 mismatch; do not apply to a different SDK')
    text = source.decode('utf-8').replace('\r\n', '\n')
    patches = ['01-sdnn-index-intrinsics.patch']
    if profile == 'epilogue':
        patches.append('02-sdnn-scalar-relu-legacy.patch')
    elif profile == 'diagnostic':
        patches.append('03-sdnn-ttxir-diagnostics.patch')
    for name in patches:
        text = apply_exact(text, (ROOT / name).read_text(encoding='utf-8'))
    result = text.encode('utf-8')
    if hashlib.sha256(result).hexdigest() != EXPECTED[profile]:
        raise ValueError('Generated compiler SHA256 mismatch')
    compile(text, 'isolated_compiler.py', 'exec')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=EXPECTED, default='epilogue')
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        parser.error('Output must be a separate experiment copy')
    data = prepare(args.source.read_bytes(), args.profile)
    with args.output.open('xb') as handle:
        handle.write(data)
    print(f'GENERATED_SHA256={hashlib.sha256(data).hexdigest()}')


if __name__ == '__main__':
    main()
