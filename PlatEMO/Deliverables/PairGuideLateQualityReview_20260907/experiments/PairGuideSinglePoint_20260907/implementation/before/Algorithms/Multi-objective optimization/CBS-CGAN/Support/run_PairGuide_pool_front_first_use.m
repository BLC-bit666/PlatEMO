function run_PairGuide_pool_front_first_use(number,arm)
%RUN_PAIRGUIDE_POOL_FRONT_FIRST_USE B/C/D, 800 epochs, one true CGAN use.
warning('off','all'); maxNumCompThreads(1);
root=fileparts(which('platemo')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuidePoolFrontFirstUse_20260907');
arm=string(arm); assert(isscalar(arm) && ismember(arm,["B","C","D"]));
assert(ismember(number,[5 7 8 10 12]));
depth=find(arm==["B","C","D"])-1;
BoundaryOptions=struct('selectionPool',"ccmo",'archiveFrontDepth',depth);
name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name);
stem=sprintf('%s_%s_epoch0800',name,arm); target=fullfile(folder,[stem,'.mat']);
if isfile(target)
    Old=load(target,'M'); assert(Old.M.epoch==800 && Old.M.arm==arm);
    fprintf('EXISTS %s\n',stem); return;
end
% This untrained prefix determines the epoch budget without changing search.
rng(1,'twister'); P=ctor('N',100,'maxFE',100000,'maxRuntime',Inf);
G=PairGuide('save',1,'outputFcn',@stopAtEligible);
G.configureComparison('fallback_only'); G.configureBoundaryExperiment(BoundaryOptions);
G.Solve(P); S=G.guideExperimentSnapshot(); Prefix=S.evidence.population{end}; W=S.evidence.W;
[Data,Gate]=PairBoundaryArchive_RC('trainingdata',Prefix.archive,W,P,struct());
assert(P.D==30 && Gate.eligible && Data.count>=8 && P.FE<100000, ...
    'CBSPairGuide:NoFirstTraining','No eligible first training set before the FE limit.');
batches=ceil(Data.count/min(Data.count,16)); productionFE=P.FE; epochs=800;
fprintf('PREFIX %s FE=%d pairs=%d batches=%d\n',stem,productionFE,Data.count,batches);
updates=epochs*batches;
rng(1,'twister'); P=ctor('N',100,'maxFE',100000,'maxRuntime',Inf);
G=PairGuide('parameter',{500,6,updates,32,5,8,.3},'save',1,'run',1,'outputFcn',@stopAfterFirstUse);
G.configureBoundaryExperiment(BoundaryOptions);
timer=tic; G.Solve(P); wall=toc(timer); S=G.guideExperimentSnapshot(); E=S.evidence;
trained=E.training(cellfun(@(x)x.trained,E.training));
assert(isscalar(trained) && isscalar(E.queries),'Expected exactly one training and one query.');
T=trained{1}; Q=E.queries{1}; Pop=E.population{Q.generation+1}; Use=E.generations{end}.use;
assert(isequaln(Pop,Prefix),'The trained trajectory must match its untrained prefix.');
assert(T.epochs==800 && T.updates==updates && T.pairVisits==800*Data.count);
assert(T.criticUpdates==5*updates && T.batchesPerEpoch==batches);
assert(P.FE==productionFE+200 && Q.productionFE==productionFE && Q.modelVersion==1);
assert(sum(cellfun(@(g)g.use.selected>0,E.generations))==1 && Use.selected>0);
assert(isequal(Q.rawDecs,Q.pool.candidateDecs) && ~isfield(Q.pool,'constructedDecs'));
assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:)) && isequal(Use.childDecs,Q.pending.decs));
assert(size(Q.rawDecs,1)==500 && T.trainingPairs==Data.count && Use.selected<=20);
assert(all(arrayfun(@(id)nnz(Q.pending.ids==id)<=2,unique(Q.pending.ids))));
verifyFrontTrajectory(E,P,depth);
% Only after stopping: all unselected native points and network endpoints are
% evaluated by a separate problem instance, never available to the search.
Eval=ctor(); PairGuideCost_RC('start',Eval); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
n=size(Q.rawDecs,1); span=Eval.upper-Eval.lower;
X=[Q.rawDecs;Eval.lower+Q.sample.generatedF.*span;Eval.lower+Q.sample.generatedI.*span];
allY=Eval.CalObj(X); allC=Eval.CalCon(X); Y=allY(1:n,:); C=allC(1:n,:);
k=Q.pool.keepIdx; assert(isequal(Y(k,:),Use.childObjs) && isequal(C(k,:),Use.childCons));
f=(Q.pool.xf-Eval.lower)./span; i=(Q.pool.xi-Eval.lower)./span;
ef=sum((Q.sample.generatedF-f).^2,2); ei=sum((Q.sample.generatedI-i).^2,2);
F=Pop.p1Objs(all(Pop.p1Cons<=0,2),:); Ar=Pop.archive; active=Ar.active;
Trace=E.generations{Q.generation}.archive;
firstInfeasibleFE=inf;
for j=1:numel(E.evaluations)
    V=E.evaluations{j}; row=find(any(V.constraints>0,2),1);
    if ~isempty(row); firstInfeasibleFE=min(firstInfeasibleFE,V.firstEvalID+row-1); end
end
M=struct('problem',string(name),'arm',arm,'epoch',epochs,'updates',T.updates, ...
    'batchesPerEpoch',batches,'trainingPairs',T.trainingPairs,'archivePairs',numel(Ar.id), ...
    'productionFE',productionFE,'firstUseFE',P.FE,'postRMSE',T.postDiagnostics.allEndpointRMSE, ...
    'queryEndpointRMSE',sqrt(mean([ef;ei])/P.D), ...
    'relativeErrorMedian',median(sqrt((ef+ei)/2)./Q.pool.gaps), ...
    'generatedFfeasible',mean(all(allC(n+(1:n),:)<=0,2)), ...
    'generatedIinfeasible',mean(any(allC(2*n+(1:n),:)>0,2)), ...
    'nativeBandRate',mean(Q.pool.nativeInBand),'nativeFeasible',mean(all(C<=0,2)), ...
    'nativeDominated',mean(dominated(F,Y)),'selectedCount',numel(k), ...
    'selectedFeasible',mean(all(C(k,:)<=0,2)),'selectedDominated',mean(dominated(F,Y(k,:))), ...
    'survivedP1',nnz(Use.survivedP1),'survivedP2',nnz(Use.survivedP2), ...
    'firstUseIGD',P.CalMetric('IGD',G.result{end,2}),'trainingSeconds',T.trainingSeconds, ...
    'wallSeconds',wall,'eligiblePairs',Trace.eligiblePairs,'archiveFrontDepth',depth, ...
    'activeFront1',nnz(Trace.retainedFrontRanks(active)==1), ...
    'activeFront2',nnz(Trace.retainedFrontRanks(active)==2), ...
    'p1DominatedActive',nnz(dominated(F,Ar.yf(active,:))), ...
    'populationOverlap',nnz(ismember(Pop.p1Decs,Pop.p2Decs,'rows')), ...
    'firstInfeasibleFE',firstInfeasibleFE);
assert(Eval.FE==0); OfflineCost=PairGuideCost_RC('snapshot'); clear cleanup;
AlgorithmCost=E.oracleCalls; FinalPop=E.population{end};
assert(AlgorithmCost.CalConRows==P.FE && AlgorithmCost.CalObjRows==2*P.FE);
assert(OfflineCost.CalConRows==1500 && OfflineCost.CalObjRows==3000);
save(target,'M','Q','T','Pop','Use','Y','C','allY','allC','Data','Trace', ...
    'OfflineCost','AlgorithmCost','FinalPop','BoundaryOptions','E','-v7.3');
writetable(struct2table(M),fullfile(folder,[stem,'.csv']));
fprintf('RESULT %s %s\n',stem,jsonencode(M));
end

function stopAtEligible(G,~)
S=G.guideExperimentSnapshot(); E=S.evidence;
if numel(E.population)>1 && nnz(E.population{end}.archive.active)>=8
    error('PlatEMO:Termination','First eligible prefix captured.');
end
end

function stopAfterFirstUse(G,P)
S=G.guideExperimentSnapshot(); E=S.evidence;
if ~isempty(E.queries)
    assert(~isempty(E.queries{1}.pending.decs),'CBSPairGuide:NoFirstUse','First query selected no usable native candidates.');
    P.maxFE=E.queries{1}.productionFE+2*P.N;
elseif any(cellfun(@(t)t.reason=="numerical_failure",E.training))
    error('CBSPairGuide:FirstTrainingFailed','First training failed; no retraining in this experiment.');
end
end

function verifyFrontTrajectory(E,P,depth)
for j=1:numel(E.generations)
    Before=E.population{j}; After=E.population{j+1}; A=After.archive; V=E.generations{j}.archive;
    assert(numel(A.id)<=500 && nnz(A.active)<=100);
    assert(numel(unique(A.ref(A.active)))==nnz(A.active));
    assert(V.evaluatedInputPoints==400 && V.candidateInputPoints==2*numel(Before.archive.id)+400);
    % Qualified history may shrink; capacity is an upper bound, not a quota.
    if depth==0
        F=After.p1Objs(all(After.p1Cons<=0,2),:);
        assert(~any(dominated(F,A.yf(A.active,:))));
    else
        X=[Before.p1Decs;Before.p2Decs]; Y=[Before.p1Objs;Before.p2Objs]; C=[Before.p1Cons;Before.p2Cons];
        for z=2*j:2*j+1
            U=E.evaluations{z}; X=[X;U.decisions]; Y=[Y;U.objectives]; C=[C;U.constraints]; %#ok<AGROW>
        end
        valid=all(C<=0,2); X=[X(valid,:);Before.archive.xf]; Y=[Y(valid,:);Before.archive.yf];
        [X,u]=unique(X,'rows','stable'); Y=Y(u,:); F=NDSort(Y,inf);
        [found,u]=ismember(A.xf,X,'rows'); assert(all(found)); ranks=reshape(F(u),[],1);
        assert(isequal(ranks,V.retainedFrontRanks) && all(ranks<=depth));
        assert(V.eligiblePairs==nnz(ranks<=depth));
        assert(~any(dominated(Y(F==1,:),A.yi)), ...
            'Every retained infeasible endpoint must pass the complete feasible front.');
    end
end
assert(P.FE==E.fullFE);
end

function d=dominated(F,Y)
if isempty(F); d=false(size(Y,1),1); return; end
weak=all(reshape(F,1,size(F,1),2)<=reshape(Y,size(Y,1),1,2)+1e-12,3);
strict=any(reshape(F,1,size(F,1),2)<reshape(Y,size(Y,1),1,2)-1e-12,3);
d=any(weak & strict,2);
end
