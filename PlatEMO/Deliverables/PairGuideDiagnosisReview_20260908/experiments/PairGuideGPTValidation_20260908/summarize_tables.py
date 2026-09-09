"""Paired descriptive summaries; no new objective or constraint evaluation."""
import csv, json, math, statistics, sys
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent
STAGE=sys.argv[1] if len(sys.argv)>1 else 'development'
PREFIX='confirmation_' if STAGE=='confirmation' else ''

def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(name,rows):
    if not rows:return
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geomean(v):return math.exp(statistics.mean(math.log(x) for x in v))

rows=read(P/(PREFIX+'search_metrics.csv')); lookup={(r['problem'],r['seed'],r['arm']):r for r in rows}
summary=[]; pairs=[]
for r in rows:
    base=lookup[r['problem'],r['seed'],'original']
    pairs.append(dict(problem=r['problem'],seed=r['seed'],arm=r['arm'],
        lateIGDAUCRatio=float(r['lateIGDAUC'])/float(base['lateIGDAUC']),
        finalIGDRatio=float(r['finalIGD'])/float(base['finalIGD']),
        lateHVAUCChange=float(r['lateHVAUC'])-float(base['lateHVAUC'])))
write(PREFIX+'search_paired.csv',pairs)
for problem in ['ALL']+sorted(set(r['problem'] for r in rows)):
 for arm in sorted(set(r['arm'] for r in rows)):
    x=[r for r in rows if r['arm']==arm and (problem=='ALL' or r['problem']==problem)]
    y=[r for r in pairs if r['arm']==arm and (problem=='ALL' or r['problem']==problem)]
    if not x:continue
    summary.append(dict(problem=problem,arm=arm,n=len(x),
        finalIGDMean=statistics.mean(float(r['finalIGD']) for r in x),
        finalIGDSD=statistics.stdev(float(r['finalIGD']) for r in x) if len(x)>1 else 0,
        lateIGDAUCMean=statistics.mean(float(r['lateIGDAUC']) for r in x),
        finalHVMean=statistics.mean(float(r['finalHV']) for r in x),
        pairedLateIGDAUCGeomean=geomean(r['lateIGDAUCRatio'] for r in y),
        pairedFinalIGDGeomean=geomean(r['finalIGDRatio'] for r in y),
        lateIGDAUCWins=sum(r['lateIGDAUCRatio']<1 for r in y)))
write(PREFIX+'search_summary.csv',summary)
if STAGE=='confirmation':
    print(json.dumps({'confirmationRuns':len(rows),'summary':[r for r in summary if r['problem']=='ALL']},indent=2))
    sys.exit(0)
all12=[r for r in summary if r['problem']=='ALL' and r['arm'].startswith('stable_') and r['n']==12]
if len(all12)==2:
 best=min(all12,key=lambda r:r['pairedLateIGDAUCGeomean'])
 (P/'development_selection.json').write_text(json.dumps(dict(
    criterion='Mean paired log lateIGDAUC ratio, require below 1',
    arms=all12,selected=best['arm'] if best['pairedLateIGDAUCGeomean']<1 else None,
    independentConfirmationRequired=best['pairedLateIGDAUCGeomean']<1),indent=2)+'\n')

fixed=[]
for p in sorted((P/'fixed_conditions').glob('*.csv')):fixed+=read(p)
write('fixed_condition_records.csv',fixed); fsum=[]; fp=[]
for n in ['LIRCMOP7_BC','LIRCMOP8_BC']:
 for u in ['0','20','200','1000','2000']:
  for m in ['0','1']:
   x=[r for r in fixed if r['problem']==n and r['addedUpdates']==u and r['moving']==m]
   fsum.append(dict(problem=n,addedUpdates=int(u),moving=int(m),n=len(x),**{
      metric:statistics.mean(float(r[metric]) for r in x)
      for metric in ['endpointRMSE','directionError','nearestTrainingObjective','labelAccuracy']}))
 for seed in ['1','2','3']:
  x={r['moving']:r for r in fixed if r['problem']==n and r['seed']==seed and r['addedUpdates']=='2000'}
  fp.append(dict(problem=n,seed=int(seed),**{metric+'FixedOverMoving':float(x['0'][metric])/float(x['1'][metric])
       for metric in ['endpointRMSE','directionError','nearestTrainingObjective']}))
write('fixed_condition_summary.csv',fsum);write('fixed_condition_pairs.csv',fp)

suffix=[]
for p in sorted((P/'suffix').glob('*.csv')):suffix+=read(p)
write('suffix_records.csv',suffix); su=[]; sp=[]
base={(r['problem'],r['seed'],r['prefixFE']):r for r in suffix if r['mode']=='raw'}
for r in suffix:
 b=base[r['problem'],r['seed'],r['prefixFE']]
 sp.append(dict(problem=r['problem'],seed=r['seed'],prefixFE=r['prefixFE'],mode=r['mode'],
    IGDAUCRatio=float(r['IGDAUC'])/float(b['IGDAUC']),finalIGDRatio=float(r['finalIGD'])/float(b['finalIGD'])))
write('suffix_paired.csv',sp)
for mode in ['raw','midpoint','de','pair_only']:
 x=[r for r in suffix if r['mode']==mode]; y=[r for r in sp if r['mode']==mode]
 su.append(dict(mode=mode,n=len(x),pairedIGDAUCGeomean=geomean(r['IGDAUCRatio'] for r in y),
    pairedFinalIGDGeomean=geomean(r['finalIGDRatio'] for r in y),wins=sum(r['IGDAUCRatio']<1 for r in y),
    firstSelected=sum(float(r['firstSelected']) for r in x),
    firstSurvivedP1=sum(float(r['firstSurvivedP1']) for r in x)))
write('suffix_summary.csv',su)
proxy=[]
for p in sorted((P/'boundary_proxies').glob('*.csv')):proxy+=read(p)
write('boundary_proxy_records.csv',proxy)
print(json.dumps({'searchRuns':len(rows),'fixedRecords':len(fixed),'suffixRuns':len(suffix),
    'boundaryRecords':len(proxy),'completeStableArms':all12},indent=2))
