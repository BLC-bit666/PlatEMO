"""Read a packaged numeric JSON record and resolve all local shared objects."""
from functools import lru_cache
import json
from pathlib import Path


def load(path):
    path = Path(path).resolve()
    root = path.parent
    while root.name != 'numeric_evidence' and root != root.parent:
        root = root.parent
    if root.name != 'numeric_evidence':
        raise ValueError('Expected a file inside numeric_evidence')

    @lru_cache(maxsize=256)
    def read(target):
        if not target.is_relative_to(root):
            raise ValueError('Reference outside numeric_evidence')
        return resolve(json.loads(target.read_text()), target.parent)

    def resolve(value, directory):
        if isinstance(value, dict):
            if set(value) == {'$ref'}:
                return read((directory / value['$ref']).resolve())
            return {k: resolve(v, directory) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, directory) for v in value]
        return value

    return read(path)
