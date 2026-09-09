"""Descriptive summaries of the exact frozen-data reset experiment."""
from pathlib import Path
import csv, statistics, json

P = Path(__file__).resolve().parent

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    if not rows:
        return
    with (P / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

rows = [r for p in sorted((P / 'runs').glob('*.csv')) for r in read(p)]
write('reset_records.csv', rows)
summary = []
for arm in ['original', 'stable_side', 'stable_no_side']:
    for problem in ['LIRCMOP7_BC', 'LIRCMOP8_BC']:
        panel = [r for r in rows if r['arm'] == arm and r['problem'] == problem]
        base = next((r for r in panel if r['reset'] == '0' and r['addedUpdates'] == '0'), None)
        if base is None:
            continue
        for step in ['1000', '2000', '5000', '10000']:
            warm = [r for r in panel if r['reset'] == '0' and r['addedUpdates'] == step]
            fresh = [r for r in panel if r['reset'] != '0' and r['addedUpdates'] == step]
            if len(warm) != 1 or len(fresh) != 3:
                continue
            w = warm[0]
            item = dict(arm=arm, problem=problem, addedUpdates=int(step), freshInitializations=3,
                        baselineObjectiveGap=float(base['objectiveGap']),
                        warmObjectiveGap=float(w['objectiveGap']),
                        freshObjectiveGapMean=statistics.mean(float(r['objectiveGap']) for r in fresh),
                        freshObjectiveGapMin=min(float(r['objectiveGap']) for r in fresh),
                        freshObjectiveGapMax=max(float(r['objectiveGap']) for r in fresh),
                        freshBetterThanBaseline=sum(float(r['objectiveGap']) < float(base['objectiveGap']) for r in fresh),
                        freshBetterThanWarm=sum(float(r['objectiveGap']) < float(w['objectiveGap']) for r in fresh),
                        baselineEndpointRMSE=float(base['endpointRMSE']), warmEndpointRMSE=float(w['endpointRMSE']),
                        freshEndpointRMSEMean=statistics.mean(float(r['endpointRMSE']) for r in fresh),
                        baselineSeenGap=float(base['seenObjectiveGap']), warmSeenGap=float(w['seenObjectiveGap']),
                        freshSeenGapMean=statistics.mean(float(r['seenObjectiveGap']) for r in fresh),
                        baselineCoverageGap=float(base['realCoverageGap']), warmCoverageGap=float(w['realCoverageGap']),
                        freshCoverageGapMean=statistics.mean(float(r['realCoverageGap']) for r in fresh),
                        seenConditions=int(base['seenConditions']), queryConditions=int(base['queryConditions']))
            summary.append(item)
write('reset_summary.csv', summary)
capacity = [r for p in sorted((P / 'capacity').glob('*.csv')) for r in read(p)]
write('capacity_records.csv', capacity)
wide = [r for p in sorted((P / 'capacity_wide').glob('*.csv')) for r in read(p)]
write('capacity_wide_records.csv', wide)
print(json.dumps({'resetRecords': len(rows), 'complete10000Runs': sum(r['addedUpdates'] == '10000' for r in rows),
                  'complete10000Panels': sum(r['addedUpdates'] == 10000 for r in summary), 'capacityRecords': len(capacity)}))
