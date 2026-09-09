"""Pack newly generated array files into one byte-shuffled store without losing values."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve();widths={}
def visit(x):
    if isinstance(x,dict):
        if '$array' in x:
            widths[x['$array']]={'int64':8,'float32':4,'float64':8}[x['dtype']]
        else:
            for v in x.values():visit(v)
    elif isinstance(x,list):
        for v in x:visit(v)
for file in root.rglob('*.json'):visit(json.loads(file.read_text()))
index={};target=root/'arrays.bin';assert not target.exists()
with target.open('wb') as out:
    for name,width in sorted(widths.items()):
        file=root/'arrays'/name;data=file.read_bytes();assert len(data)%width==0
        shuffled=b''.join(data[i::width] for i in range(width))
        index[name]=[out.tell(),len(data),width]
        out.write(shuffled)
(root/'array_index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
# These are only this packager's generated temporary array files.
for name in index:(root/'arrays'/name).unlink()
(root/'arrays').rmdir()
print('PACKED_ARRAYS',len(index),'bytes',target.stat().st_size)
