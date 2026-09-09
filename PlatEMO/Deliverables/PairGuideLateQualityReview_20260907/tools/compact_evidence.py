"""Share identical cached arrays/states; assert exact JSON round-trip equality."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'numeric_evidence'
SHARED = ROOT / 'shared'


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()


def resolve(value, directory):
    if isinstance(value, dict):
        if set(value) == {'$ref'}:
            path = (directory / value['$ref']).resolve()
            assert path.is_relative_to(ROOT.resolve())
            return resolve(json.loads(path.read_text()), path.parent)
        return {key: resolve(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, directory) for item in value]
    return value


def share(value):
    content = encoded(value)
    digest = hashlib.sha256(content).hexdigest()
    path = SHARED / (digest + '.json')
    if not path.exists():
        path.write_bytes(content + b'\n')
    return {'$ref': 'shared/' + path.name}


def main():
    SHARED.mkdir(exist_ok=True)
    checks = []
    for path in sorted(ROOT.glob('*.json')):
        if not (path.name.endswith('_plot_state.json') or path.name.endswith('_query.json')):
            continue
        original = resolve(json.loads(path.read_text()), ROOT)
        changed = json.loads(json.dumps(original))
        if path.name.endswith('_plot_state.json'):
            for field in ('searchState', 'trainingState', 'generationState'):
                if field in changed:
                    changed[field] = share(changed[field])
        else:
            # These are repeated copies in the native evidence schema.
            for parent, field in ((changed, 'rawDecs'), (changed['sample'], 'normalized'),
                                  (changed['pool'], 'candidateDecs')):
                parent[field] = share(parent[field])
        assert resolve(changed, ROOT) == original, path
        path.write_bytes(encoded(changed) + b'\n')
        checks.append({'path': path.name, 'expanded_sha256': hashlib.sha256(encoded(original)).hexdigest()})
    (ROOT / 'lossless_compaction.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Lossless reference round-trips:', len(checks))


if __name__ == '__main__':
    main()
