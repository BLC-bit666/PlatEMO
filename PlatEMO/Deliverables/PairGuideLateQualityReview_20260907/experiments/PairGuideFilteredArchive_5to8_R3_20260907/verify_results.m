function verify_results
% Verify this completed campaign without training or new objective evaluation.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
folder=fullfile(root,'Data','PairGuideFilteredArchive_5to8_R3_20260907');
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
S=load(fullfile(folder,'state.mat'),'State'); P=load(fullfile(folder,'protocol.mat'),'Protocol');
assert(S.State.status=="complete" && S.State.completed==12 && S.State.failed==0);
O=P.Protocol.options;
assert(isequal(O.problems,["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC"]) && ...
    isequal(O.seeds,1:3) && isequal(O.modes,"cgan") && O.N==100 && O.maxFE==100000);
assert(P.Protocol.workers==10 && P.Protocol.threadsPerWorker==1);
D=P.Protocol.algorithmDefaults;
assert(D.selectionPool=="ccmo" && D.archiveFrontDepth==1 && D.pairArchiveCapacity==500 && ...
    D.pairInitialEpoch==200 && D.pairRetrainEpoch==20 && D.nCritic==5 && D.ganMiniBatch==32);
expectedStages=sort(["first_true_use","FE010000","FE030000","FE050000","FE070000","FE100000"]);
rows=0; missingFirst=0; figureCount=0; plottedStates=0; missingGeneration=0;
for problem=O.problems
 for seed=O.seeds
  name=sprintf('%s_seed%02d_cgan.mat',problem,seed);
  R=load(fullfile(folder,name),'Record','Checkpoints'); C=R.Checkpoints;
  assert(R.Record.finalFE==100000 && R.Record.N==100 && R.Record.D==30 && R.Record.defaultDimension);
  assert(R.Record.sourceHash==P.Protocol.sourceHash && R.Record.mode=="cgan" && R.Record.seed==seed);
  assert(height(C)==6 && isequal(sort(C.stage)',expectedStages));
  assert(all(C.actualFE(~C.missing)<=C.targetFE(~C.missing)) && C.actualFE(C.stage=="FE100000")==100000);
  assert(all(C.archivePairs(~C.missing)<=500) && all(C.activePairs(~C.missing)<=100));
  assert(all(isfinite(C.eligibilityRemoved(~C.missing))) && all(isfinite(C.infeasiblePoolEligible(~C.missing))));
  rows=rows+height(C); missingFirst=missingFirst+nnz(C.missing & C.stage=="first_true_use");
 end
 cache=load(fullfile(folder,'analysis','figures',char(problem),'run_01','plot_data.mat'),'Report','States');
 assert(cache.Report.sourceHash==P.Protocol.sourceHash && cache.Report.seed==1 && ...
     cache.Report.offlineFullFE==0 && cache.Report.oracleCalls.CalConRows==0);
 assert(all(isfile(cache.Report.files)) && all(isfile(cache.Report.focusFiles)));
 assert(numel(cache.Report.files)==numel(cache.States) && ...
     cache.Report.figureCount==2*numel(cache.States));
 figureCount=figureCount+cache.Report.figureCount;
 for k=1:numel(cache.States)
  V=cache.States{k}; plottedStates=plottedStates+1;
  assert(V.observationFE<=V.targetFE && size(V.p1,1)==100 && size(V.p2,1)==100);
  if V.missingGeneration, missingGeneration=missingGeneration+1; continue; end
  assert(isequal(V.pending,V.children) && size(V.xf,1)==V.trainingPairs);
  assert(V.trainingFE<=V.productionFE && V.productionFE<V.consumptionFE && V.consumptionFE<=V.observationFE);
 end
 fprintf('VERIFIED %s runs=3 figures=%d\n',problem,cache.Report.figureCount);
end
T=readtable(fullfile(folder,'analysis','checkpoints_per_run.csv'),'TextType','string');
assert(height(T)==72 && rows==72);
G=readtable(fullfile(folder,'analysis','problem_summary.csv'),'TextType','string');
assert(height(G)==4 && all(G.runs==3));
assert(isempty(dir(fullfile(folder,'*.failure.txt'))));
V=struct('status',"verified",'completedRuns',12,'failedRuns',0,'checkpointRows',rows, ...
    'missingFirstUse',missingFirst,'distributionFigures',figureCount,'plottedStates',plottedStates, ...
    'plottedStatesWithoutGeneration',missingGeneration,'metricCurves',4, ...
    'sourceHash',P.Protocol.sourceHash,'verifiedAt',string(datetime('now')));
fid=fopen(fullfile(folder,'verification.json'),'w'); assert(fid>=0);
cleanup=onCleanup(@()fclose(fid)); fprintf(fid,'%s\n',jsonencode(V,PrettyPrint=true)); clear cleanup;
fprintf('CAMPAIGN_RESULTS_VERIFIED %s\n',jsonencode(V));
end
