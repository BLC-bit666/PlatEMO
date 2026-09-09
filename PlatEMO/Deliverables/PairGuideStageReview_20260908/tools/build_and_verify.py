"""Build/verify this review package, using cached source files only. No experiments run."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import zipfile
from read_numeric import load

root=Path.cwd();package=Path(__file__).resolve().parents[1]
assert package.name=='PairGuideStageReview_20260908'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

# Make remaining references to excluded historical figures explicit, not broken links.
source_rows=json.loads((package/'tools/copy_sources.json').read_text());source_lookup={r['file']:r for r in source_rows}
omitted=json.loads((package/'tools/omitted_historical_links.json').read_text())
for doc in (package/'experiments').rglob('*.md'):
    text=doc.read_text()
    def clean(m):
        t=urllib.parse.unquote(m.group(2).strip('<>').split('#')[0])
        if not t or re.match(r'^\w+://',t):return m.group(0)
        path=(doc.parent/t).resolve()
        if path.exists() and path.is_relative_to(package):return m.group(0)
        omitted.append({'document':str(doc.relative_to(package)),'target':m.group(2)})
        return m.group(1)+'（历史原包文件：'+m.group(2)+'）'
    fixed=re.sub(r'\[([^\]]*)\]\(([^)]+)\)',clean,text)
    if fixed!=text:
        doc.write_text(fixed);source_lookup[str(doc.relative_to(package))]['transformation']='Only portable document links; missing historical targets explicitly identified.'
(package/'tools/copy_sources.json').write_text(json.dumps(source_rows,ensure_ascii=False,indent=2)+'\n')
(package/'tools/omitted_historical_links.json').write_text(json.dumps(omitted,ensure_ascii=False,indent=2)+'\n')

production=json.loads((root/'Data/PairGuideStageDiagnosis_20260908/source_manifest.json').read_text())
for r in production:
    assert sha(root/r['path'])==r['sha256']
    assert sha(package/'code'/r['path'])==r['sha256']
provenance=[]
for r in source_rows:
    source=root/r['source'];target=package/r['file'];assert source.is_file() and target.is_file()
    if r['transformation']=='none':assert sha(source)==sha(target),r['file']
    provenance.append([r['file'],r['source'],sha(source),sha(target),r['transformation']])

latest=package/'experiments/PairGuideStageDiagnosis_20260908'
for file,count in [('gradient_records.csv',288),('low_lr_runs_gradient_records.csv',144),('gradient_pair_verification.csv',36),
                   ('low_lr_runs_pair_verification.csv',18),('suffix_records.csv',84),('suffix_verification.csv',84),
                   ('suffix_boundary_witnesses.csv',252),('suffix_trajectories.csv',2184)]:
    assert len(read_csv(latest/file))==count,file
assert json.loads((latest/'final_validation.json').read_text())['status']=='passed'
assert len(list((latest/'figures').glob('*.png')))==31
assert len(list(package.rglob('*.png')))==39
assert (package/'PROMPT_TO_GPT.md').read_bytes()==(root/'Deliverables/PairGuideStageReview_20260908_PROMPT.md').read_bytes()

numeric=package/'numeric_evidence'
checks=json.loads((numeric/'lossless_compaction.json').read_text());assert len(checks)==254
stage_sources=json.loads((package/'tools/stage_numeric_sources.json').read_text());assert len(stage_sources)==244
history_sources=json.loads((package/'tools/history_numeric_sources.json').read_text());assert len(history_sources)==10
cases=0
for r in checks:
    value=load(numeric/r['path'])
    assert hashlib.sha256(canonical(value)).hexdigest()==r['expanded_sha256'],r['path']
    if r['path'].startswith(('runs__','low_lr_runs__')):
        assert value['Complete'] and len(value['Snapshots'])==4 and len(value['Table'])==4
        assert [x['addedUpdates'] for x in value['Table']]==[0,20,200,2000]
        for s in value['Snapshots']:
            assert 'netG' in s['Model'] and 'netC' in s['Model']
            assert len(s['Samples']['X'])==200 and len(s['Samples']['Y'])==200
        cases+=1
assert cases==108
print('ALL254_NUMERIC_RECORDS_EXACTLY_VERIFIED',flush=True)

links=0
for doc in [package/'README_FIRST.md',package/'PROMPT_TO_GPT.md',*list((package/'experiments').rglob('*.md'))]:
    for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
        target=urllib.parse.unquote(target.strip('<>').split('#')[0])
        if not target or re.match(r'^\w+://',target):continue
        path=(doc.parent/target).resolve();assert path.is_relative_to(package) and path.exists(),(doc,target)
        links+=1
with (package/'SOURCE_PROVENANCE.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['file','source','source_sha256','package_sha256','transformation']);w.writerows(sorted(provenance))
validation={'status':'passed','currentTrainingCases':108,'currentSearchWindows':84,'latestOriginalPNGs':31,'historicalOriginalPNGs':8,
            'currentNumericRecords':244,'historicalNumericRecords':10,'allModelCheckpointsRetained':[0,20,200,2000],
            'exactExpandedNumericRecordsVerified':254,'productionSourceHashesVerified':len(production),
            'localDocumentLinksVerified':links,'matFiles':0,'newTrainingForPackaging':0,'newOracleCallsForPackaging':0,
            'numberEncoding':'Exact float32 only if every value round-trips; otherwise float64; integer number types restored. Little-endian byte-shuffled XZ arrays.bin.xz with array_index.json. Reader uses Python standard library.',
            'limitBytes':50000000,'zipFormat':'DEFLATE',
            'note':'These are packaging checks, not fresh experiments. ZIP byte-size/CRC/SHA checks execute after writing. MANIFEST excludes itself and this validation file.'}
(package/'BUNDLE_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
files=sorted(f for f in package.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc')
for f in files:assert f.suffix.lower() not in {'.mat','.fig','.zip'},f
with (package/'MANIFEST.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['file','bytes','sha256'])
    for file in files:
        if file.name not in ['MANIFEST.csv','BUNDLE_VALIDATION.json']:w.writerow([str(file.relative_to(package)),file.stat().st_size,sha(file)])
files=sorted(f for f in package.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc')
target=package.with_suffix('.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in files:z.write(f,str(Path(package.name)/f.relative_to(package)))
assert target.stat().st_size<50000000,target.stat().st_size
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None and len(z.infolist())==len(files)
    assert not any(Path(i.filename).suffix.lower()=='.mat' for i in z.infolist())
    for r in read_csv(package/'MANIFEST.csv'):
        b=z.read(package.name+'/'+r['file']);assert len(b)==int(r['bytes']) and hashlib.sha256(b).hexdigest()==r['sha256']
checksum=sha(target);target.with_suffix('.zip.sha256').write_text(f'{checksum}  {target.name}\n')
result={'zip':str(target),'bytes':target.stat().st_size,'files':len(files),'png':39,'mat':0,'CRC':'passed','SHA256':checksum}
(root/'Data/PairGuideStageDiagnosis_20260908/package_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
