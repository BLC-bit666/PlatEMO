"""Summarize cached held-out usage and boundary proxies; no oracle calls."""
from pathlib import Path
import csv, math, statistics

P = Path(__file__).resolve().parent

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (P / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

usage = [row for path in sorted((P / 'confirmation_usage').glob('*.csv')) for row in read(path)]
assert len(usage) == 40
write('confirmation_usage.csv', usage)
boundary = [row for path in sorted((P / 'confirmation_boundary_proxies').glob('*.csv')) for row in read(path)]
assert len(boundary) == 80 * 51 * 3
write('confirmation_boundary_proxy_records.csv', boundary)
summary = []
for problem in ['ALL'] + sorted({r['problem'] for r in boundary}):
    for threshold in ['0.01', '0.02', '0.04']:
        for arm in sorted({r['arm'] for r in boundary}):
            rows = [r for r in boundary if r['arm'] == arm and r['threshold'] == threshold
                    and (problem == 'ALL' or r['problem'] == problem)]
            final = [r for r in rows if r['FE'] == '100000']
            summary.append(dict(problem=problem, threshold=threshold, arm=arm, n=len(final),
                finalNovelWitnessesMean=statistics.mean(float(r['novelSeparatedWitnesses']) for r in final),
                finalCoveredWitnessesMean=statistics.mean(float(r['witnessesCoveredByP1']) for r in final),
                lateSnapshotNovelWitnessesMean=statistics.mean(float(r['novelSeparatedWitnesses']) for r in rows)))
write('confirmation_boundary_proxy_summary.csv', summary)

trajectories = read(P / 'confirmation_search_trajectories.csv')
points = {(r['problem'], r['seed'], r['arm']): float(r['IGD'])
          for r in trajectories if r['targetFE'] == '50000'}
prefix = []
for problem in ['ALL'] + sorted({r['problem'] for r in trajectories}):
    ratios = [value / points[p, seed, 'original'] for (p, seed, arm), value in points.items()
              if arm == 'stable_no_side' and (problem == 'ALL' or p == problem)]
    prefix.append(dict(problem=problem, n=len(ratios),
        IGDAt50000PairedGeomean=math.exp(statistics.mean(math.log(x) for x in ratios))))
write('confirmation_prefix_comparison.csv', prefix)
for arm in ['original', 'stable_no_side']:
    rows = [r for r in usage if r['arm'] == arm]
    totals = {key: sum(int(r[key]) for r in rows) for key in
              ['lateSelected', 'lateFeasible', 'lateRetainedP1', 'lateRetainedP2', 'lateRetainedEither', 'lateRetainedNeither']}
    print(arm, totals)
print('SECONDARY_SUMMARY_COMPLETE', len(boundary), 'boundary rows')
