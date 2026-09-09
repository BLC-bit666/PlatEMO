function audit_single_campaign(selectedFiles)
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907');
protocol=load(fullfile(folder,'full_run','protocol.mat'),'Protocol');
files=dir(fullfile(folder,'full_run','LIRCMOP*_seed*_*.mat'));
stem="";
if nargin>0
    files=files(ismember(string({files.name}),string(selectedFiles)));
    assert(numel(files)==numel(selectedFiles)); stem="pilot_";
else
    assert(numel(files)==36);
end
checks=cell(numel(files),1); quality=struct([]); costs=struct([]);
for f=1:numel(files)
    R=load(fullfile(files(f).folder,files(f).name),'Record','Audit'); E=R.Audit.evidence;
    assert(R.Record.sourceHash==protocol.Protocol.sourceHash);
    assert(E.schema=="PairGuide-single-v3" && R.Record.finalFE==100000 && R.Record.D==30);
    assert(E.fullFE==100000 && E.objectiveOnlyFE==0 && E.constraintOnlyFE==0);
    assert(E.oracleCalls.CalObjRows==200000 && E.oracleCalls.CalConRows==100000);
    assert(all(cellfun(@(g)g.archive.eligibilityRejectedPairs==0 && ...
        g.archive.archiveFrontDepth==1 && all(g.archive.retainedFrontRanks==1),E.generations)));
    assert(all(cellfun(@(p)numel(p.archive.id)<=500,E.population)));
    initialReplay=false; updates=0; trained=E.training(cellfun(@(t)t.trained,E.training));
    if R.Record.mode=="cgan"
        assert(~isempty(trained) && trained{1}.updates==1000);
        assert(all(cellfun(@(t)t.updates==20,trained(2:end))));
        updates=sum(cellfun(@(t)t.updates,trained));
        assert(E.lastModel.iterG==updates && E.lastModel.iterC==5*updates);
        assert(~isempty(E.lastModel.avgG) && ~isempty(E.lastModel.avgC));
        assert(E.networkEndpointRows==500*numel(E.queries));
        first=load(fullfile(folder,'first_use',sprintf('%s_seed%02d_lrzero1000.mat', ...
            R.Record.problem,R.Record.seed)),'Q','Prefix');
        assert(isequaln(E.population{E.queries{1}.generation+1},first.Prefix));
        assert(isequal(E.queries{1}.rawDecs,first.Q.rawDecs) && ...
            isequal(E.queries{1}.pending.decs,first.Q.pending.decs)); initialReplay=true;
        for j=1:numel(E.queries)
            Q=E.queries{j}; U=E.generations{Q.generation+1}.use;
            assert(size(Q.rawDecs,1)==500 && all(Q.pending.ids==0));
            assert(isequal(Q.rawDecs(Q.pool.keepIdx,:),Q.pending.decs));
            [present,selectedIndex]=ismember(U.childDecs,Q.pending.decs,'rows');
            assert(all(present) && all(U.matchedPairIds==0));
            if U.selected>0
                assert(isequal(U.requestedRefs,Q.pending.refs(selectedIndex)) && ...
                    isequal(U.requestedSides,Q.pending.sides(selectedIndex)));
            end
            assert(U.selected+U.fallback==U.requested);
            assert(nnz(Q.sides==0)==250 && nnz(Q.sides==1)==250);
            assert(numel(unique(Q.refs))==size(E.W,1));
        end
        ctor=str2func(char(R.Record.problem)); Eval=ctor();
        PairGuideCost_RC('start',Eval); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
        times=cellfun(@(p)p.observationFE,E.population);
        firstUse=find(cellfun(@(g)g.use.selected>0,E.generations),1)+1;
        indices=[firstUse,arrayfun(@(fe)find(times<=fe,1,'last'),[10000 30000 50000 70000 100000])];
        stages=["first_true_use","FE010000","FE030000","FE050000","FE070000","FE100000"];
        for j=1:numel(indices)
            P=E.population{indices(j)}; U=E.generations{indices(j)-1}.use;
            q=find(cellfun(@(v)v.productionFE==U.productionFE,E.queries),1);
            if isempty(q); continue; end
            Q=E.queries{q}; T=trained{Q.modelVersion}; Train=E.population{T.generation+1};
            [Data,~]=PairBoundaryArchive_RC('trainingdata',Train.archive,E.W,Eval, ...
                struct('referenceScale',Train.referenceScale));
            [X,~,map]=unique(Q.rawDecs,'rows'); Yunique=Eval.CalObj(X); Cunique=Eval.CalCon(X);
            Y=Yunique(map,:); feasible=all(Cunique(map,:)<=0,2); correct=feasible==Q.sides;
            scale=E.population{Q.generation+1}.referenceScale;
            [actual,~,Yn]=AssignReferenceVectors_CBS(Y,E.W,scale);
            [~,~,YtrainFrame]=AssignReferenceVectors_CBS(Y,E.W,Train.referenceScale);
            target=Q.conditions(:,1:end-1); target=target./max(vecnorm(target,2,2),eps);
            angles=directionError(Yn,target); trainAngles=directionError(YtrainFrame,target);
            known=ismember(Q.conditions,[Data.cF;Data.cI],'rows');
            Prefix=E.population{Q.generation+1}; F=Prefix.p1Objs(all(Prefix.p1Cons<=0,2),:);
            dominated=false(500,1);
            for k=1:size(F,1)
                dominated=dominated | (all(F(k,:)<=Y+1e-12,2) & any(F(k,:)<Y-1e-12,2));
            end
            TrainY=[Train.archive.yf(Train.archive.active,:);Train.archive.yi(Train.archive.active,:)];
            nearest=sqrt(min(pdist2(Y,TrainY,'squaredeuclidean'),[],2));
            row=struct('problem',R.Record.problem,'seed',R.Record.seed,'stage',stages(j), ...
                'observationFE',P.observationFE,'productionFE',Q.productionFE,'trainingFE',Train.observationFE, ...
                'modelVersion',Q.modelVersion,'uniqueRaw',size(X,1),'trainingPairs',Data.count, ...
                'labelAccuracy',mean(correct),'feasibleRequestAccuracy',mean(feasible(Q.sides==1)), ...
                'infeasibleRequestAccuracy',mean(~feasible(Q.sides==0)), ...
                'directionError',mean(angles),'trainingFrameDirectionError',mean(trainAngles), ...
                'knownDirectionError',mean(angles(known)),'unseenDirectionError',mean(angles(~known)), ...
                'knownFraction',mean(known),'referenceHitRate',mean(actual==Q.refs), ...
                'jointRate',mean(correct & angles<=5 & ~dominated),'rawUseful',mean(feasible & ~dominated), ...
                'nearestObjectiveDistance',mean(nearest),'selected',U.selected, ...
                'selectedLabelAccuracy',mean(all(U.childCons<=0,2)==U.requestedSides), ...
                'survivedP1',nnz(U.survivedP1));
            quality=[quality;row]; %#ok<AGROW>
        end
        Cost=PairGuideCost_RC('snapshot'); assert(Eval.FE==0); clear cleanup;
        costs=[costs;struct('problem',R.Record.problem,'seed',R.Record.seed, ...
            'searchFEAdded',0,'CalObjRows',Cost.CalObjRows,'CalConRows',Cost.CalConRows)]; %#ok<AGROW>
    else
        assert(isempty(trained) && E.networkEndpointRows==0);
    end
    checks{f}=struct('problem',R.Record.problem,'seed',R.Record.seed,'mode',R.Record.mode, ...
        'sourceHash',R.Record.sourceHash,'finalFE',E.fullFE,'initialExactReplay',initialReplay, ...
        'generatorUpdates',updates,'trainingEvents',numel(trained),'queries',numel(E.queries), ...
        'archiveEligibilityVerified',true,'nativeCoordinatesVerified',true);
    fprintf('AUDIT %s seed=%d %s passed\n',R.Record.problem,R.Record.seed,R.Record.mode);
end
writetable(struct2table(vertcat(checks{:})),fullfile(folder,stem+'full_verification.csv'));
writetable(struct2table(quality),fullfile(folder,stem+'conditional_quality.csv'));
writetable(struct2table(costs),fullfile(folder,stem+'conditional_quality_offline_cost.csv'));
Q=struct2table(quality);
S=groupsummary(Q,{'problem','stage'},'mean', ...
    {'labelAccuracy','directionError','trainingFrameDirectionError','jointRate','rawUseful','survivedP1','knownFraction'});
writetable(S,fullfile(folder,stem+'conditional_quality_mean.csv'));
fprintf('%sVERIFICATION_PASS %d runs\n',stem,numel(files));
end

function a=directionError(Y,target)
Y=Y./max(vecnorm(Y,2,2),eps);
a=acosd(max(-1,min(1,sum(Y.*target,2))));
end
