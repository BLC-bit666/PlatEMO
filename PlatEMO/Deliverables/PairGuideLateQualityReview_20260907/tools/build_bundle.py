"""Build the review ZIP from the original workspace; never include MAT files."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
from urllib.parse import unquote

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent.parent
CAMPAIGNS = [
    'PairGuideSinglePoint_20260907',
    'PairGuideFilteredArchive_5to8_R3_20260907',
    'PairGuideTrainingDiagnosis_20260907',
    'PairGuideDominanceDrift_20260907',
]
ALLOWED = {'.m', '.py', '.md', '.txt', '.csv', '.json', '.log', '.png'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build():
    sources = {}
    for name in CAMPAIGNS:
        for source in (REPO / 'Data' / name).rglob('*'):
            if source.is_file() and source.suffix.lower() in ALLOWED:
                sources[source] = PACKAGE / 'experiments' / name / source.relative_to(REPO / 'Data' / name)
    manifest = REPO / 'Data' / CAMPAIGNS[0] / 'full_source_manifest.json'
    for item in json.loads(manifest.read_text()):
        source = REPO / item['path']
        sources[source] = PACKAGE / 'code' / item['path']
    base = REPO / 'Algorithms' / 'Multi-objective optimization' / 'CBS-CGAN'
    for source in base.rglob('*'):
        if source.is_file() and source.suffix.lower() in {'.m', '.md'}:
            sources[source] = PACKAGE / 'code' / source.relative_to(REPO)
    sources[REPO / 'Agent.md'] = PACKAGE / 'review_notes' / 'Agent_current.md'
    provenance = []
    for source, target in sorted(sources.items()):
        original = source.read_bytes()
        content = original
        frozen = any(part in {'before', 'full_source', 'offline_source', 'first_use_source',
                              'mismatch_source', 'source_snapshot'} for part in source.parts)
        if source.suffix == '.md' and not frozen:
            def link(match):
                label, destination = match.groups()
                absolute = unquote(destination.strip('<>'))
                absolute = re.sub(r':\d+$', '', absolute)
                if not absolute.startswith(str(REPO) + '/'):
                    return match.group(0)
                actual = Path(absolute)
                mapped = sources.get(actual)
                if mapped is None and actual.is_dir():
                    relative = actual.relative_to(REPO)
                    if relative.parts[0] == 'Data' and relative.parts[1] in CAMPAIGNS:
                        mapped = PACKAGE / 'experiments' / Path(*relative.parts[1:])
                if mapped is not None:
                    return label + '(<'+os.path.relpath(mapped, target.parent)+'>)'
                return label.lstrip('!').strip('[]') + '（原工作区文件，未打包）'
            content = re.sub(r'(!?\[[^\]\n]*\])\(([^\n]*?)\)', link,
                             original.decode('utf-8')).encode('utf-8')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        provenance.append({'package_path': target.relative_to(PACKAGE).as_posix(),
                           'source_path': source.relative_to(REPO).as_posix(),
                           'source_sha256': digest(original), 'package_sha256': digest(content),
                           'operation': 'copy' if content == original else 'markdown_links_only'})
    with (PACKAGE / 'SOURCE_PROVENANCE.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(provenance[0]))
        writer.writeheader(); writer.writerows(provenance)
    # This probe was produced solely while preparing this review package.
    probe = PACKAGE / 'tools' / 'probe.log'
    if probe.exists(): probe.unlink()
    files = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and
                   p.name not in {'MANIFEST.csv', 'BUNDLE_VALIDATION.json'} and
                   '__pycache__' not in p.parts)
    assert not any(p.suffix.lower() in {'.mat', '.fig', '.zip'} or p.name == '.DS_Store' for p in files)
    with (PACKAGE / 'MANIFEST.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'bytes', 'sha256'])
        for path in files:
            data = path.read_bytes()
            writer.writerow([path.relative_to(PACKAGE).as_posix(), len(data), digest(data)])
    for name in CAMPAIGNS:
        original_images = list((REPO / 'Data' / name).rglob('*.png'))
        for source in original_images:
            assert sources[source].read_bytes() == source.read_bytes()
    images = sum(path.suffix.lower() == '.png' for path in files)
    assert images == 117, images
    archive = PACKAGE.with_suffix('.zip')
    assert not archive.exists(), 'Refuse to overwrite an existing ZIP.'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in files + [PACKAGE / 'MANIFEST.csv']:
            output.write(path, (Path(PACKAGE.name) / path.relative_to(PACKAGE)).as_posix())
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        assert not any(Path(name).suffix.lower() in {'.mat', '.fig'} for name in saved.namelist())
        for row in csv.DictReader((PACKAGE / 'MANIFEST.csv').open()):
            raw = saved.read(PACKAGE.name + '/' + row['path'])
            assert digest(raw) == row['sha256'] and len(raw) == int(row['bytes'])
    assert archive.stat().st_size < 50_000_000, archive.stat().st_size
    validation = {'zip_integrity': 'passed', 'manifest_hashes': 'all passed',
                  'mat_files': 0, 'fig_files': 0, 'original_png_files': images,
                  'copied_source_and_experiment_files': len(sources),
                  'numeric_export': json.loads((PACKAGE / 'numeric_evidence' / 'export_status.json').read_text()),
                  'limit_bytes': 50_000_000,
                  'zip_bytes_before_validation_record': archive.stat().st_size,
                  'note': 'MANIFEST covers payload except itself and this validation record.'}
    path = PACKAGE / 'BUNDLE_VALIDATION.json'
    path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        output.write(path, PACKAGE.name + '/' + path.name)
    assert archive.stat().st_size < 50_000_000
    with zipfile.ZipFile(archive) as saved:
        assert saved.testzip() is None
        count = len(saved.namelist())
    sha = digest(archive.read_bytes())
    archive.with_suffix('.zip.sha256').write_text(sha + '  ' + archive.name + '\n')
    print(json.dumps({'zip': str(archive), 'bytes': archive.stat().st_size,
                      'MB': archive.stat().st_size/1_000_000, 'files': count,
                      'png': images, 'sha256': sha}, indent=2))


if __name__ == '__main__':
    build()
