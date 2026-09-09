"""Final cached artifact checks and cost ledger; no MATLAB/oracle calls."""
import csv
import hashlib
import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

folder=Path(__file__).resolve().parent
root=folder.parent.parent

def read(name):
    with (folder/name).open(newline='') as f:return list(csv.DictReader(f))

def number(x):return float(x)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

sources=json.loads((folder/'source_manifest.json').read_text())
for item in sources:
    assert sha(root/item['path'])==item['sha256'],item['path']
assert len(list((folder/'fixtures').glob('*.mat')))==20
assert len(list((folder/'baselines').glob('*.mat')))==20
assert len(read('stage_baselines.csv'))==20
for prefix,expected in [('',72),('low_lr_runs_',36)]:
    rows=read(prefix+'gradient_records.csv')
    assert len(rows)==expected*4
    groups={}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in ['problem','stage','loss','stream']),[]).append(r)
        assert number(r['offlineFE'])==200 and number(r['offlineObjRows'])==400 and number(r['offlineConRows'])==200
    assert len(groups)==expected
    for values in groups.values():assert sorted(number(x['addedUpdates']) for x in values)==[0,20,200,2000]
    audit_name='gradient_pair_verification.csv' if not prefix else 'low_lr_runs_pair_verification.csv'
    audit=read(audit_name);assert len(audit)==expected//2
    for r in audit:
        assert all(number(r[k])==1 for k in ['complete','sameInitialState','sameRNGTrajectory','frozenData','matchedGradientNorm'])
        assert number(r['GUpdatesPerArm'])==2000 and number(r['DUpdatesPerArm'])==10000
    assert len(read(prefix+'actual_step_records.csv'))==expected*105

suffix=read('suffix_records.csv');verify=read('suffix_verification.csv');assert len(suffix)==len(verify)==84
assert sum(number(r['rawExactReplay']) for r in verify)==12
assert all(number(r['FE'])==5000 and number(r['prefixVerified'])==1 and number(r['queryGateVerified'])==1 and number(r['oneBatchOnly'])==1 for r in verify)
assert sum(number(r['suffixFE']) for r in suffix)==420000
assert len(read('suffix_boundary_witnesses.csv'))==252
assert len(read('suffix_trajectories.csv'))==2184
assert len(read('runs_geometry.csv'))==288 and len(read('low_lr_runs_geometry.csv'))==144
assert len(read('learning_rate_paired.csv'))==108 and len(read('suffix_paired.csv'))==48
assert len(read('adam_step_size.csv'))==108 and len(read('source_gradient_reproduction.csv'))==10
neutral=read('probe_neutrality.csv');assert len(neutral)==2
assert all(number(r[k])==1 for r in neutral for k in ['equalG','equalD','equalAdam','equalRNG'])
for n in range(5,9):
    r=read(f'LIRCMOP{n}_BC_seed01_capture_check.csv')[0]
    assert all(number(r[k])==1 for k in ['sameP1','sameP2','sameG','sameD','sameLastQuery'])
    assert number(r['searchFE'])==100000

capture=read('capture_training_cost.csv');assert len(capture)==4
native_g=sum(int(r['GUpdates']) for r in capture);native_d=sum(int(r['DUpdates']) for r in capture)
ledger={'newTrueFE':888800,'newCalObjRows':1777600,'newCalConRows':888800,
        'components':[{'name':'unchanged full-search capture','cases':4,'FE':400000,'GUpdates':native_g,'DUpdates':native_d},
                      {'name':'unique frozen baselines','cases':20,'FE':4000},
                      {'name':'72 primary cases,three nonzero geometry checkpoints','cases':216,'FE':43200,'GUpdates':144000,'DUpdates':720000},
                      {'name':'36 lower-G-LR cases,three nonzero geometry checkpoints','cases':108,'FE':21600,'GUpdates':72000,'DUpdates':360000},
                      {'name':'one-batch search suffixes','cases':84,'FE':420000},
                      {'name':'two20-step native/probe identity comparisons','FE':0,'GUpdates':80,'DUpdates':400},
                      {'name':'36 actual-Adam one-step probes','FE':0,'GUpdates':36,'DUpdates':180}],
        'totalNewGParameterUpdates':216116+native_g,'totalNewDParameterUpdates':1080580+native_d,
        'failedSuffixAttempts':44,'failedSuffixFE':0,
        'accountingNotes':['Zero-step samples reused across arms; only20 baseline evaluations counted.',
                           'Static gradients,geometry decomposition,audits and plots use cached data and add no oracle FE.',
                           'BC Evaluation calls CalObj twice including its constraint path; these rows are counted.',
                           'Offline oracle results never feed training or online search.',
                           'Gradient diagnostic overhead remains in measured wall time; matched gradient norm is not matched computational cost.',
                           'Copied-state coordinate/gradient computations are not additional parameter updates.']}
assert sum(r['FE'] for r in ledger['components'])==ledger['newTrueFE']
(folder/'cost_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')

pngs=list((folder/'figures').glob('*.png'));assert len(pngs)==31
for f in pngs:
    data=f.read_bytes();assert data[:8]==b'\x89PNG\r\n\x1a\n'
    w,h=struct.unpack('>II',data[16:24]);assert w>=1000 and h>=700
for name in ['REPORT.md','REPRODUCE.md','SOURCE_REVIEW.md']:
    text=(folder/name).read_text();assert len(text)>1000
    for target in re.findall(r'\]\((/[^)]+)\)',text):
        # This validator creates its own linked result only after all checks pass.
        if Path(target)!=folder/'final_validation.json':assert Path(target).exists(),target

validation={'status':'passed','checkedAtUTC':datetime.now(timezone.utc).isoformat(),
            'completePrimaryCases':72,'completeLowerGRateCases':36,'completeSearchWindows':84,
            'exactUnchangedFullSearchReplays':4,'exactNativeFirstSteps':12,'pairedTrainingStreams':54,
            'stageObservationFE':[20000,50000,80000],'searchSeeds':[1],'trainingStreams':[1,2,3],
            'scientificPNGs':31,'productionFilesUnchanged':len(sources),
            'newTrueFE':ledger['newTrueFE'],'reportedLimitations':['One search trajectory per problem;training streams are not independent search runs.',
            '5/6 mid/late historical training data reported separately;6 late original query gate closed.',
            'Suffixes change one batch then freeze further CGAN service;not continuous modified online training.',
            'No validated production fix or proof of a unique root cause.']}
(folder/'final_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
# Fingerprint all reviewable code/numeric summaries/figures plus final model snapshots.
paths=sorted(f for f in folder.rglob('*') if f.is_file() and f.suffix in {'.m','.py','.md','.csv','.json','.png','.mat','.patch'}
             and f.name not in {'artifact_manifest.json','WORK_STATE.json'})
manifest=[{'path':str(f.relative_to(folder)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in paths]
(folder/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False))
