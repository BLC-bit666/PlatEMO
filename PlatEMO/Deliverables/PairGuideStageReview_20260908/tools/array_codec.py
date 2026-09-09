"""Lossless numeric-array storage; standard library only, little-endian bytes."""
import array
import hashlib
from functools import lru_cache
import json
import math
import lzma
from pathlib import Path
import sys


def flatten(value):
    if not isinstance(value,list) or not value:return None
    if all(type(x) in (int,float) and math.isfinite(x) for x in value):return [len(value)],value
    if not all(isinstance(x,list) for x in value):return None
    parts=[flatten(x) for x in value]
    if any(x is None for x in parts) or any(x[0]!=parts[0][0] for x in parts):return None
    return [len(value)]+parts[0][0],[v for p in parts for v in p[1]]


def encode(value,directory):
    shape,values=flatten(value)
    integers=[i for i,x in enumerate(values) if type(x) is int]
    if len(integers)==len(values):
        code='q';a=array.array(code,values);integers=[]
    else:
        try:
            a=array.array('f',values)
            if not all(float(v)==x for v,x in zip(a,values)):raise ValueError('Needs float64')
            code='f'
        except (OverflowError,ValueError):code='d';a=array.array(code,values)
    if sys.byteorder!='little':a.byteswap()
    descriptor={'dtype':{'q':'int64','f':'float32','d':'float64'}[code],'shape':shape}
    if integers:descriptor['integerPositions']=integers
    raw=a.tobytes();key=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()+raw).hexdigest()+'.bin'
    path=directory/key
    if not path.exists():path.write_bytes(raw)
    descriptor['$array']=key
    return descriptor


@lru_cache(maxsize=4)
def packed_index(root):
    return json.loads((root/"array_index.json").read_text())


@lru_cache(maxsize=2)
def packed_bytes(root):
    compressed=root/'arrays.bin.xz'
    if compressed.exists():return lzma.decompress(compressed.read_bytes())
    return (root/'arrays.bin').read_bytes()


def decode(descriptor,root):
    file=descriptor['$array']
    compact=type(file) is int
    if compact:
        descriptor=packed_index(root)[file]
    else:
        assert Path(file).name==file
    code={'int64':'q','float32':'f','float64':'d'}[descriptor['dtype']]
    a=array.array(code)
    direct=None if compact else root/'arrays'/file
    if direct is not None and direct.exists():data=direct.read_bytes()
    else:
        if compact:offset,length,width=descriptor['offset'],descriptor['length'],descriptor['width']
        else:offset,length,width=packed_index(root)[file]
        shuffled=packed_bytes(root)[offset:offset+length]
        assert len(shuffled)==length and width==a.itemsize
        count=length//width;data=bytearray(length)
        for i in range(width):data[i::width]=shuffled[i*count:(i+1)*count]
    a.frombytes(data)
    if sys.byteorder!='little':a.byteswap()
    values=a.tolist()
    for i in descriptor.get('integerPositions',[]):values[i]=int(values[i])
    shape=descriptor['shape'];assert math.prod(shape)==len(values)
    def restore(flat,dims):
        if len(dims)==1:return flat
        stride=math.prod(dims[1:]);return [restore(flat[i*stride:(i+1)*stride],dims[1:]) for i in range(dims[0])]
    return restore(values,shape)
