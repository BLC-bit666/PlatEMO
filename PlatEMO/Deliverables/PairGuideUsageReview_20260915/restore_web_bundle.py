"""Python standard library only. Decode data; never execute bundled research code.
Usage: python restore_web_bundle.py /path/to/PairGuide_CGAN_Usage_WEB_20260915.zip /tmp/pairguide_review
"""
from pathlib import Path, PurePosixPath
from io import BytesIO
import base64, hashlib, json, re, sys, tarfile, zipfile

def decode(source, destination):
    with zipfile.ZipFile(source) as outer:
        assert outer.testzip() is None
        data=outer.read('evidence.tar.xz')
        expected=json.loads(outer.read('BUNDLE.json'))['payloadSHA256']
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
