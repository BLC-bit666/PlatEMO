"""Summarize completed/partial checkpoints without training or oracle calls."""
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

folder = Path(__file__).resolve().parent
directory = sys.argv[1] if len(sys.argv) > 1 else 'runs'
prefix = '' if directory == 'runs' else directory + '_'


def read(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def write(name, rows):
    if not rows:
        return
    with (folder / (prefix + name)).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


records = []
steps = []
complete = 0
for path in sorted((folder / directory).glob('*.csv')):
    if path.name.endswith('_steps.csv'):
        continue
    rows = read(path)
    records.extend(rows)
    complete += any(r['addedUpdates'] == 2000 for r in rows)
    diagnostic = path.with_name(path.stem + '_steps.csv')
    if diagnostic.exists():
        for row in read(diagnostic):
            row.update({k: rows[-1][k] for k in ['problem', 'stage', 'stream', 'loss']})
            steps.append(row)
write('gradient_records.csv', records)
write('actual_step_records.csv', steps)

lookup = {(r['problem'], r['stage'], r['stream'], r['loss'], r['addedUpdates']): r for r in records}
pairs = []
for key, b in lookup.items():
    problem, stage, stream, loss, updates = key
    if loss != 'distance' or updates == 0:
        continue
    a = lookup.get((problem, stage, stream, 'adversarial', updates))
    if a is None:
        continue
    row = {'problem': problem, 'stage': stage, 'stream': stream, 'addedUpdates': updates}
    for metric in ['exactDistance', 'seenObjectiveGap', 'singletonObjectiveGap', 'conditionalSeenGap',
                   'objectiveGap', 'reverseCoverageGap', 'directionError', 'weightedDecisionMSE']:
        row[metric + 'Ratio'] = b[metric] / a[metric] if a[metric] > 0 else math.nan
    row['supportCriterion'] = int(row['exactDistanceRatio'] < 1 and row['seenObjectiveGapRatio'] <= .8
                                  and row['reverseCoverageGapRatio'] <= 1.05
                                  and row['singletonObjectiveGapRatio'] < 1)
    pairs.append(row)
write('gradient_paired.csv', pairs)

groups = defaultdict(list)
for row in pairs:
    groups[(row['problem'], row['stage'], row['addedUpdates'])].append(row)
summary = []
for (problem, stage, updates), group in sorted(groups.items()):
    row = {'problem': problem, 'stage': stage, 'addedUpdates': updates, 'pairedStreams': len(group),
           'supportCount': sum(r['supportCriterion'] for r in group)}
    for metric in [k for k in group[0] if k.endswith('Ratio')]:
        values = [r[metric] for r in group if math.isfinite(r[metric]) and r[metric] > 0]
        row[metric + 'GeoMean'] = math.exp(sum(map(math.log, values)) / len(values)) if values else math.nan
    summary.append(row)
write('gradient_summary.csv', summary)

step_groups = defaultdict(list)
for row in steps:
    step_groups[(row['problem'], row['stage'], row['loss'])].append(row)
step_summary = []
for (problem, stage, loss), group in sorted(step_groups.items()):
    step_summary.append({'problem': problem, 'stage': stage, 'loss': loss, 'observedSteps': len(group),
                         'negativeAdversarialCosineRate': sum(r['fullVsAdversarialCosine'] < 0 for r in group) / len(group),
                         'positiveAdamDirectionalDerivativeRate': sum(r['adamDirectionalDerivative'] > 0 for r in group) / len(group),
                         'distanceIncreaseRate': sum(r['distanceChange'] > 1e-7 for r in group) / len(group),
                         'finiteStepReversalRate': sum(r['adamDirectionalDerivative'] < -1e-7 and r['distanceChange'] > 1e-7 for r in group) / len(group)})
write('actual_step_summary.csv', step_summary)
status = {'completedRuns': complete, 'plannedRuns': 72 if directory == 'runs' else 36, 'recordedCheckpoints': len(records),
          'matchedNonzeroCheckpoints': len(pairs), 'actualStepsObserved': len(steps),
          'note': 'Partial summaries are exploratory; final decision requires complete paired streams.'}
(folder / (prefix + 'progress_summary.json')).write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status))
for row in summary:
    if row['addedUpdates'] == 2000:
        print(row['problem'], row['stage'], 'streams', row['pairedStreams'],
              'distance', round(row['exactDistanceRatioGeoMean'], 3),
              'seen', round(row['seenObjectiveGapRatioGeoMean'], 3),
              'coverage', round(row['reverseCoverageGapRatioGeoMean'], 3), 'support', row['supportCount'])
