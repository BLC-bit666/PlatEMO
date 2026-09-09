"""Select diagnosis snapshots for the <50 MB review package, without rounding.

Run from the original PlatEMO root after the two read-only MATLAB exporters.
Only rebuilds this package's own generated numeric_evidence directory.
"""
import json
from pathlib import Path
import shutil

root = Path.cwd()
package = Path(__file__).resolve().parents[1]
out = package / 'numeric_evidence'
assert (root / 'Data/PairGuideResetAt99800_20260908/review_numeric').is_dir()
assert out.parent == package and package.name == 'PairGuideDiagnosisReview_20260908'
if out.exists():
    shutil.rmtree(out)
out.mkdir()
records = []


def save(source, name, value, selection):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    records.append({'file': name, 'source': str(source.relative_to(root)), 'selection': selection})


for source in sorted((root / 'Data/PairGuideGPTValidation_20260908/review_numeric').glob('*.json')):
    name = source.name
    if not (name.endswith(('plot01.json', 'plot06.json', 'snapshot5.json', 'fixture.json'))):
        continue
    value = json.loads(source.read_text())
    selection = 'Complete exported record.'
    if name.endswith('fixture.json'):
        value = {k: v for k, v in value.items() if k != 'Prefixes'}
        selection = 'Data, W and full Spans retained; large 50000/70000 FE suffix Prefixes omitted.'
    save(source, 'validation__' + name, value, selection)

for source in sorted((root / 'Data/PairGuideResetAt99800_20260908/review_numeric').glob('*.json')):
    value = json.loads(source.read_text())
    selection = 'Complete exported fixture, including original G/D and archive state.'
    if 'Snapshots' in value:
        final = value['Snapshots'][-1]
        # Initial fixture plus fresh_model(seed) and verification records document resets.
        # Keep complete final network/Adam state and final generated coordinates/objectives.
        value['Snapshots'] = [final]
        selection = ('Final snapshot only, including complete G/D/Adam state and generated X/Y; '
                     'all checkpoint metrics remain in Table and CSV; intermediate samples/models omitted.')
    save(source, 'reset__' + source.name, value, selection)

(package / 'tools/numeric_sources.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'Selected {len(records)} numeric records; values not rounded.')
