"""Locked candidate versus each held-out control; cached metrics only."""
import csv,json,math,statistics,random
from pathlib import Path
P=Path(__file__).resolve().parent
selected=json.loads((P/'confirmation_lock.json').read_text())['selected']
with (P/'confirmation_search_metrics.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==80
lookup={(r['problem'],int(r['seed']),r['arm']):r for r in rows}
output=[];details=[]
for baseline in ['original','fallback_only','pair_only']:
 for problem in ['ALL']+[f'LIRCMOP{n}_BC' for n in range(5,9)]:
  pairs=[]
  for r in rows:
   if r['arm']!=selected or (problem!='ALL' and r['problem']!=problem):continue
   b=lookup[r['problem'],int(r['seed']),baseline]
   v=dict(problem=r['problem'],seed=int(r['seed']),baseline=baseline,
      lateIGDAUCRatio=float(r['lateIGDAUC'])/float(b['lateIGDAUC']),
      finalIGDRatio=float(r['finalIGD'])/float(b['finalIGD']),
      lateHVAUCChange=float(r['lateHVAUC'])-float(b['lateHVAUC']))
   pairs.append(v)
   if problem=='ALL':details.append(v)
  logs=[math.log(r['lateIGDAUCRatio']) for r in pairs]
  # Five seed blocks, retaining all four problems of a sampled seed together.
  blocks=[statistics.mean(math.log(r['lateIGDAUCRatio']) for r in pairs if r['seed']==seed)
          for seed in range(11,16)]
  rng=random.Random(20260908);samples=sorted(math.exp(statistics.mean(rng.choices(blocks,k=5))) for _ in range(10000))
  output.append(dict(problem=problem,baseline=baseline,n=len(pairs),
    lateIGDAUCRatio=math.exp(statistics.mean(logs)),
    seedBlockBootstrapLow=samples[249],seedBlockBootstrapHigh=samples[9749],
    lateIGDAUCWins=sum(r['lateIGDAUCRatio']<1 for r in pairs),
    finalIGDRatio=math.exp(statistics.mean(math.log(r['finalIGDRatio']) for r in pairs)),
    finalIGDWins=sum(r['finalIGDRatio']<1 for r in pairs),
    meanLateHVAUCChange=statistics.mean(r['lateHVAUCChange'] for r in pairs)))
for name,data in [('confirmation_candidate_comparisons.csv',output),('confirmation_candidate_pairs.csv',details)]:
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps([r for r in output if r['problem']=='ALL'],indent=2))
