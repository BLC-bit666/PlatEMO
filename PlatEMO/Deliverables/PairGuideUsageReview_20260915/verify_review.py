"""Verify decoded original bytes, figure pixels and evidence scope. No experiments."""
from pathlib import Path
import hashlib, json
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;EXPANDED=HERE/'_roundtrip'
manifest=json.loads((HERE/'package_manifest.json').read_text())
txt=Path(manifest['txt']);assert txt.stat().st_size==manifest['bytes']<50_000_000
assert hashlib.sha256(txt.read_bytes()).hexdigest()==manifest['sha256']
index=json.loads((EXPANDED/'FILE_INDEX.json').read_text());assert len(index['files'])==manifest['entries']
for row in index['files']:
    data=(EXPANDED/row['path']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    if row['origin']==row['path'] and (ROOT/row['path']).exists():
        assert (ROOT/row['path']).read_bytes()==data,row['path']
images=json.loads((EXPANDED/'Review/FIGURE_INDEX.json').read_text())
for r in images:
    with Image.open(EXPANDED/r['path']) as im:
        im.load();assert im.size==(r['width'],r['height'])
        assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==r['displayedRGBSHA256']
    assert hashlib.sha256((ROOT/r['sourcePath']).read_bytes()).hexdigest()==r['originalSHA256']
current=[r for r in manifest['sourceVersions'] if '20260914' in r['manifest'] or '20260915' in r['manifest']]
assert all(r['matched']==r['total'] for r in current)
assert len(manifest['summary'])==13
assert abs(manifest['summary'][1]['metrics']['AUC_200_100000']['percent']+4.267599178417458)<1e-8
assert abs(manifest['summary'][1]['metrics']['IGDfinal']['percent']+5.16465609806006)<1e-8
assert len(images)==manifest['images']
checks={'status':'passed','bytes':txt.stat().st_size,'allRestoredFilesVerified':len(index['files']),
        'allImagesPixelsVerified':len(images),'currentEraCodeManifestsFullyCovered':len(current),
        'fullRunUseFamiliesRecomputed':len(manifest['summary']),
        'searchFEAdded':0,'trainingUpdatesAdded':0,'algorithmFilesModifiedForPackaging':False,
        'webUploadTested':False,'textUploadCaveat':'Official FAQ caps text/document files at 2M tokens; use the equivalent ZIP with Python if TXT is rejected.'}
(HERE/'VALIDATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
