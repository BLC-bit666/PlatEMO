"""Replace repeated long array keys/descriptors with compact integer IDs, losslessly."""
from pathlib import Path
import json
import sys
root=Path(sys.argv[1]).resolve();old=json.loads((root/'array_index.json').read_text());assert isinstance(old,dict)
keys=sorted(old);ids={key:i for i,key in enumerate(keys)};index=[None]*len(keys)
files=[f for f in root.rglob('*.json') if f.name not in ['array_index.json','lossless_compaction.json']]
def transform(v):
    if isinstance(v,dict):
        if '$array' in v:
            key=v['$array'];i=ids[key];offset,length,width=old[key]
            item={k:x for k,x in v.items() if k!='$array'};item.update(offset=offset,length=length,width=width)
            assert index[i] is None or index[i]==item
            index[i]=item
            return {'$array':i}
        return {k:transform(x) for k,x in v.items()}
    if isinstance(v,list):return [transform(x) for x in v]
    return v
for f in files:f.write_text(json.dumps(transform(json.loads(f.read_text())),ensure_ascii=False,separators=(',',':'))+'\n')
assert all(x is not None for x in index)
(root/'array_index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
print('SHORT_ARRAY_IDS',len(index))
