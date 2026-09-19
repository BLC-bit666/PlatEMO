"""Python standard library only. Decode data; never execute bundled research code.
Usage: python decode_review.py /path/to/PairGuide_CGAN_Usage_Review_20260915.txt /tmp/pairguide_review
"""
from pathlib import Path, PurePosixPath
from io import BytesIO
import base64, hashlib, json, re, sys, tarfile

def decode(source, destination):
    raw=Path(source).read_text(encoding='utf-8')
    begin='----- BEGIN PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----'
    end='----- END PAIRGUIDE_EVIDENCE_TAR_XZ_BASE64 -----'
    encoded=raw.rsplit(begin,1)[1].split(end,1)[0]
    data=base64.b64decode(''.join(encoded.split()),validate=True)
    expected=re.search(r'^Payload TAR.XZ SHA256: ([0-9a-f]{64})$',raw,re.M).group(1)
    assert hashlib.sha256(data).hexdigest()==expected,'Payload hash mismatch'
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=BytesIO(data),mode='r:xz') as z:
        # Read each blob once in a forward pass, avoiding repeated XZ decompression.
        blobs={m.name:z.extractfile(m).read() for m in z if m.isfile()}
        index=json.loads(blobs.pop('FILE_INDEX.json'))
        for row in index['files']:
            rel=PurePosixPath(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts,'Unsafe path'
            target=(root/str(rel)).resolve();assert target.is_relative_to(root)
            b=blobs['blobs/'+row['sha256']]
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==b,'Refusing to overwrite a different existing file'
            else:target.write_bytes(b)
        (root/'FILE_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'verifiedFiles':len(index['files']),'directory':str(root),'payloadSHA256':expected},ensure_ascii=False))
    return index

if __name__=='__main__':
    decode(sys.argv[1],sys.argv[2])
