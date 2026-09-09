"""Stop only this task's redundant, read-only legacy report after 60 searches.
The unified held-out report is built from the sealed results by separate scripts.
"""
from pathlib import Path
import os,signal,subprocess,sys,time,json,re
P=Path(__file__).resolve().parent
pid=int(sys.argv[1]);deadline=time.monotonic()+1800
expected={f'LIRCMOP{n}_BC_seed{s:02d}_{m}.mat' for n in range(5,9)
          for s in range(11,16) for m in ['cgan','fallback_only','pair_only']}
while time.monotonic()<deadline:
 args=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
 if not args:raise SystemExit('Parent already exited; no signal sent.')
 assert 'MATLAB' in args and 'run_confirmation(8)' in args and str(P) in args
 log=(P/'confirmation.log').read_text(errors='replace')
 if (P/'confirmation_selected'/'protocol.mat').exists():raise SystemExit('Selected search already started; no signal sent.')
 done=len(re.findall(r'^DONE LIRCMOP',log,re.M))
 if done==60 and not re.search(r'^FAILED LIRCMOP',log,re.M):
  actual={f.name for f in (P/'confirmation_controls').glob('LIRCMOP*_seed*_*.mat')}
  assert actual==expected
  procs=subprocess.run(['ps','-axo','pid=,ppid=,comm='],capture_output=True,text=True).stdout.splitlines()
  workers=[]
  for line in procs:
   v=line.strip().split(None,2)
   if len(v)==3 and int(v[1])==pid and v[2].endswith('/MATLAB'):workers.append(int(v[0]))
  if not workers:
   record=dict(controlSearchRunsComplete=60,failedSearchRuns=0,sealedFiles=len(actual),
       parentPID=pid,stoppedPhase='redundant legacy report, after worker pool closed',
       reportReplacement='unified confirmation CSV, plots, per-run audits and final report',
       signal='SIGINT',unixTime=time.time())
   (P/'control_report_orchestration.json').write_text(json.dumps(record,indent=2)+'\n')
   os.kill(pid,signal.SIGINT)
   print('CONTROL_SEARCH_COMPLETE_LEGACY_REPORT_INTERRUPTED',flush=True)
   break
 time.sleep(1)
else:raise SystemExit('Guard timed out; no signal sent.')
