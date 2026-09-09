"""Preserve every numeric value and JSON number type; share repeated decision rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from array_codec import flatten,encode
from read_numeric import load

root=Path(sys.argv[1]).resolve();directory=root/'arrays';directory.mkdir(exist_ok=False)
files=sorted(root.glob('*.json'))+sorted((root/'shared').glob('*.json'))
files=[f for f in files if f.name!='lossless_compaction.json']
rows=Counter()
def key(row):return hashlib.sha256(json.dumps(row,separators=(',',':')).encode()).digest()
def count(v):
    if isinstance(v,dict):
        for x in v.values():count(x)
    elif isinstance(v,list):
        flat=flatten(v)
        if flat is not None:
            if len(flat[0])==2 and flat[0][1]>=16:
                for r in v:rows[key(r)]+=1
        else:
            for x in v:count(x)
for f in files:count(json.loads(f.read_text()))

def transform(v):
    if isinstance(v,dict):return {k:transform(x) for k,x in v.items()}
    if isinstance(v,list):
        flat=flatten(v)
        if flat is not None and len(flat[1])>=256:
            if len(flat[0])==2 and flat[0][1]>=16 and sum(rows[key(r)]>=3 for r in v)>=.5*len(v):
                return [encode(r,directory) if rows[key(r)]>=3 else r for r in v]
            return encode(v,directory)
        if flat is None:return [transform(x) for x in v]
    return v
for f in files:f.write_text(json.dumps(transform(json.loads(f.read_text())),ensure_ascii=False,separators=(',',':'))+'\n')
checks=json.loads((root/'lossless_compaction.json').read_text())
for r in checks:
    expanded=load(root/r['path'])
    canonical=json.dumps(expanded,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==r['expanded_sha256'],r['path']
print('BINARY_ARRAYS_VERIFIED',len(checks),'records,',len(list(directory.glob('*.bin'))),'array files; exact numeric values and JSON number types retained')
