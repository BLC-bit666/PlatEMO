"""Build the review ZIP from the staged package; verify evidence and provenance.

Run from the original PlatEMO root. Only writes this package's generated
manifests and its sibling ZIP/checksum. Requires only the Python standard library.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import zipfile

from read_numeric import load

root = Path.cwd()
package = Path(__file__).resolve().parents[1]
assert package.name == 'PairGuideDiagnosisReview_20260908'
assert (root / 'Data/PairGuideResetAt99800_20260908/source_manifest.json').is_file()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


source_checks = json.loads((root / 'Data/PairGuideResetAt99800_20260908/source_manifest.json').read_text())
for row in source_checks:
    assert sha(root / row['path']) == row['sha256']
    assert sha(package / 'code' / row['path']) == row['sha256']

provenance = []
for row in json.loads((package / 'tools/copy_sources.json').read_text()):
    source = root / row['source']
    target = package / row['file']
    if row['transformation'] == 'none':
        assert sha(source) == sha(target), target
    provenance.append([row['file'], row['source'], sha(source), sha(target), row['transformation']])

numeric_sources = json.loads((package / 'tools/numeric_sources.json').read_text())
compaction = {row['path']: row['expanded_sha256'] for row in
              json.loads((package / 'numeric_evidence/lossless_compaction.json').read_text())}
assert len(numeric_sources) == len(compaction) == 88
for row in numeric_sources:
    target = package / 'numeric_evidence' / row['file']
    expanded = load(target)
    assert hashlib.sha256(canonical(expanded)).hexdigest() == compaction[row['file']]
    source = root / row['source']
    original = json.loads(source.read_text())
    if row['file'].startswith('validation__') and row['file'].endswith('fixture.json'):
        original.pop('Prefixes')
    if row['file'].startswith('reset__') and 'Snapshots' in original:
        original['Snapshots'] = [original['Snapshots'][-1]]
    assert expanded == original, row['file']
    provenance.append([str(target.relative_to(package)), row['source'], sha(source), sha(target),
                       row['selection'] + ' Repeated arrays use exact local JSON references.'])

important_documents = [package / name for name in ['README_FIRST.md', 'PROMPT_TO_GPT.md', 'LATEST_SYNTHESIS.md']]
important_documents += list((package / 'experiments').glob('*/REPORT.md'))
important_documents += list((package / 'experiments').glob('*/REPRODUCE.md'))
important_documents += [package / 'experiments/PairGuideSinglePoint_20260907/RESULTS.md']
link_count = 0
for path in important_documents:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        target = urllib.parse.unquote(target.strip('<>').split('#')[0])
        if not target or re.match(r'^\w+://', target):
            continue
        resolved = (path.parent / target).resolve()
        assert resolved.is_relative_to(package), (path, target)
        assert resolved.exists(), (path, target)
        link_count += 1

reset = package / 'experiments/PairGuideResetAt99800_20260908'
validation = package / 'experiments/PairGuideGPTValidation_20260908'
for path, count in [(reset / 'verification.csv', 24),
                    (reset / 'mismatch_verification_10000.csv', 2),
                    (reset / 'reset_records.csv', 120),
                    (validation / 'confirmation_search_metrics.csv', 80),
                    (validation / 'suffix_records.csv', 48)]:
    assert len(csv_rows(path)) == count, path
pngs = list(package.rglob('*.png'))
assert len(pngs) == 149
assert len(list((validation).rglob('*.png'))) == 118
assert len(list((reset).rglob('*.png'))) == 17

with (package / 'SOURCE_PROVENANCE.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['packaged_file', 'source_file', 'source_sha256', 'packaged_sha256', 'transformation'])
    writer.writerows(sorted(provenance))

checks = {
    'status': 'passed', 'production_source_hashes_verified': len(source_checks),
    'numeric_records_exactly_verified': len(numeric_sources),
    'unmodified_png_files': len(pngs), 'current_two_round_png_files': 135,
    'primary_document_local_links_verified': link_count,
    'mat_files': 0, 'new_training_runs_for_packaging': 0, 'new_oracle_calls_for_packaging': 0,
    'zip_format': 'standard DEFLATE', 'zip_limit_bytes': 50000000,
    'note': 'Packaging checks, not a new execution of all experiment scripts. '
            'MANIFEST excludes itself and this validation record to avoid self hashes. '
            'ZIP CRC, byte size and SHA-256 are verified after writing.'
}
(package / 'BUNDLE_VALIDATION.json').write_text(json.dumps(checks, indent=2) + '\n')
files = sorted(f for f in package.rglob('*') if f.is_file()
               and '__pycache__' not in f.parts and f.suffix != '.pyc')
for path in files:
    assert path.suffix.lower() not in {'.mat', '.fig', '.zip'}, path
with (package / 'MANIFEST.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['file', 'bytes', 'sha256'])
    for path in files:
        if path.name not in {'MANIFEST.csv', 'BUNDLE_VALIDATION.json'}:
            writer.writerow([str(path.relative_to(package)), path.stat().st_size, sha(path)])
files = sorted(f for f in package.rglob('*') if f.is_file()
               and '__pycache__' not in f.parts and f.suffix != '.pyc')
target = package.with_suffix('.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, arcname=str(Path(package.name) / path.relative_to(package)))
assert target.stat().st_size < 50000000, target.stat().st_size
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert len(archive.infolist()) == len(files)
    assert not any(Path(info.filename).suffix.lower() == '.mat' for info in archive.infolist())
    for row in csv_rows(package / 'MANIFEST.csv'):
        data = archive.read(package.name + '/' + row['file'])
        assert len(data) == int(row['bytes'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
checksum = sha(target)
target.with_suffix('.zip.sha256').write_text(f'{checksum}  {target.name}\n')
print(json.dumps({'zip': str(target), 'bytes': target.stat().st_size, 'files': len(files),
                  'png': len(pngs), 'mat': 0, 'crc': 'passed', 'sha256': checksum}, indent=2))
