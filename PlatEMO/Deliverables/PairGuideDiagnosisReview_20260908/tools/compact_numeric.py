"""Losslessly share repeated JSON arrays/objects; verify every expanded value.

Usage: python3 compact_numeric.py /path/to/numeric_evidence
References are relative to the JSON file containing them. No values are rounded.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def children(value, function):
    if isinstance(value, dict):
        return {k: function(v) for k, v in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [function(v) for v in value]
    return value


def main(root):
    files = sorted(root.glob('*.json'))
    counts = Counter()

    def count(value):
        if not isinstance(value, (dict, list)):
            return value
        content = encode(value)
        if len(content) >= 4096:
            counts[hashlib.sha256(content).hexdigest()] += 1
            children(value, count)
        return value

    for path in files:
        count(json.loads(path.read_text()))
    shared = root / 'shared'
    shared.mkdir(exist_ok=False)

    def transform(value, inside=False, top=False):
        if not isinstance(value, (dict, list)):
            return value
        content = encode(value)
        if len(content) < 4096:
            return value
        key = hashlib.sha256(content).hexdigest()
        if not top and counts[key] > 1:
            target = shared / (key + '.json')
            if not target.exists():
                body = children(value, lambda v: transform(v, True))
                target.write_bytes(encode(body) + b'\n')
            return {'$ref': ('' if inside else 'shared/') + target.name}
        return children(value, lambda v: transform(v, inside))

    @lru_cache(maxsize=256)
    def read(path):
        return resolve(json.loads(path.read_text()), path.parent)

    def resolve(value, directory):
        if isinstance(value, dict):
            if set(value) == {'$ref'}:
                path = (directory / value['$ref']).resolve()
                assert path.is_relative_to(root.resolve())
                return read(path)
            return {k: resolve(v, directory) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, directory) for v in value]
        return value

    checks = []
    for path in files:
        original = json.loads(path.read_text())
        compacted = transform(original, top=True)
        assert resolve(compacted, root) == original, path
        path.write_bytes(encode(compacted) + b'\n')
        checks.append({'path': path.name, 'expanded_sha256': digest(original)})
    (root / 'lossless_compaction.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f'Verified {len(checks)} exact round trips; {len(list(shared.glob("*.json")))} shared objects.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
