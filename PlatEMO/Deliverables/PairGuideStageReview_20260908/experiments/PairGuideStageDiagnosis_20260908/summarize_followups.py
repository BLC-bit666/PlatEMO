"""Cached-data summaries; ratios are paired geometric means, not significance tests."""
import csv
import math
from collections import defaultdict
from pathlib import Path

folder = Path(__file__).resolve().parent

def read(name):
    with (folder / name).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
    return rows

def write(name, rows):
    with (folder/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def key(r):return tuple(r[k] for k in ['problem','stage','loss','stream','addedUpdates'])
metrics=['exactDistance','seenObjectiveGap','singletonObjectiveGap','conditionalSeenGap','reverseCoverageGap','directionError']
original={key(r):r for r in read('gradient_records.csv')}
groups=defaultdict(list);paired=[]
for low in read('low_lr_runs_gradient_records.csv'):
    if not low['addedUpdates']:continue
    high=original[key(low)]
    r={k:low[k] for k in ['problem','stage','loss','stream','addedUpdates']}
    for m in metrics:r[m+'Ratio']=low[m]/high[m]
    paired.append(r);groups[tuple(r[k] for k in ['problem','stage','loss','addedUpdates'])].append(r)
write('learning_rate_paired.csv',paired)
summary=[]
for (problem,stage,loss,updates), group in sorted(groups.items()):
    r=dict(problem=problem,stage=stage,loss=loss,addedUpdates=updates,streams=len(group))
    for m in metrics:r[m+'RatioGeoMean']=math.exp(sum(math.log(x[m+'Ratio']) for x in group)/len(group))
    summary.append(r)
write('learning_rate_summary.csv',summary)

suffix=read('suffix_records.csv');lookup={(r['problem'],r['prefixFE'],r['mode']):r for r in suffix};pairs=[]
for r in suffix:
    if r['mode'] in ['raw','de','pair_only']:continue
    row={k:r[k] for k in ['problem','prefixFE','mode','queryAvailable','firstSelected','firstSurvivedP1']}
    for control in ['raw','de','pair_only']:
        b=lookup[(r['problem'],r['prefixFE'],control)]
        for m in ['finalIGD','IGDAUC','finalHV']:
            row[m+'RatioVs_'+control]=r[m]/b[m] if b[m] else math.nan
            row[m+'DifferenceVs_'+control]=r[m]-b[m]
    pairs.append(row)
write('suffix_paired.csv',pairs)
print('FOLLOWUP_SUMMARIES_COMPLETE:108 low-rate paired checkpoints,48 search-window comparisons')
for arm in ['adversarial20','distance20','adversarial2000','distance2000']:
    rows=[r for r in pairs if r['mode']==arm and r['queryAvailable']]
    for control in ['raw','de']:
        ratios=[r['finalIGDRatioVs_'+control] for r in rows]
        auc=[r['IGDAUCRatioVs_'+control] for r in rows]
        print(arm,'vs',control,'IGD wins/ties/losses at1%',sum(x<.99 for x in ratios),sum(.99<=x<=1.01 for x in ratios),sum(x>1.01 for x in ratios),'AUC wins/ties/losses',sum(x<.99 for x in auc),sum(.99<=x<=1.01 for x in auc),sum(x>1.01 for x in auc))
